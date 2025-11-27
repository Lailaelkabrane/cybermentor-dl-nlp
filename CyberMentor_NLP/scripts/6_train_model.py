# 6_train_model_mix_speed_accuracy.py
"""
Entraînement mix vitesse/qualité (Option C)
- DistilBERT
- max_length=128, batch_size=16, epochs=3
- stratified train/val (90/10)
- class weights, scheduler, AMP, early save best-val
- tokenization en amont pour accélérer l'entraînement
- safe cap par classe pour garantir temps d'entraînement raisonnable
"""

import os
import time
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "./data/nlp_train.csv"   # Doit contenir 'text_features' et 'multi_class_label'
MODEL_OUTPUT_DIR = "./models/cybermentor_distilbert_mix"
MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_PCT = 0.06
NUM_WORKERS = 2
PIN_MEMORY = True

# Cap par classe pour contenu trop grand (évite entraînements trop longs)
MAX_SAMPLES_PER_CLASS = 1200

# Early save tolerance (si val n'améliore pas après N epochs on arrête)
EARLY_STOPPING_PATIENCE = 2

# -------------------------
# HELPERS
# -------------------------
def load_and_check(df_path=DATA_PATH):
    print(f"📥 Chargement: {df_path}")
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"{df_path} introuvable. Place nlp_train.csv dans ./data/")
    df = pd.read_csv(df_path)
    if 'text_features' not in df.columns or 'multi_class_label' not in df.columns:
        raise ValueError("Le fichier doit contenir 'text_features' et 'multi_class_label'.")
    df = df.dropna(subset=['text_features', 'multi_class_label']).reset_index(drop=True)
    df['multi_class_label'] = df['multi_class_label'].astype(int)
    print(f"🔎 Total samples: {len(df)} | Num labels: {df['multi_class_label'].nunique()}")
    print(df['multi_class_label'].value_counts().sort_index())
    return df

def cap_per_class(df, max_per_class=MAX_SAMPLES_PER_CLASS):
    counts = df['multi_class_label'].value_counts().to_dict()
    if max(counts.values()) <= max_per_class:
        print("✅ Pas de cap nécessaire par classe.")
        return df
    print(f"⚠️ Application d'un cap par classe: {max_per_class} échantillons max / classe.")
    frames = []
    for cls, cnt in counts.items():
        cls_df = df[df['multi_class_label'] == cls]
        if cnt > max_per_class:
            cls_df = cls_df.sample(n=max_per_class, random_state=42)
        frames.append(cls_df)
    new_df = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"🔽 Nouveau total: {len(new_df)}")
    print(new_df['multi_class_label'].value_counts().sort_index())
    return new_df

def prepare_tokenized_tensors(df, tokenizer, max_length=MAX_LENGTH):
    texts = df['text_features'].astype(str).tolist()
    labels = df['multi_class_label'].tolist()
    # Tokenize in batch (faster)
    print("🔁 Tokenization (batch)...")
    enc = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = enc['input_ids']
    attention_mask = enc['attention_mask']
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_mask, labels_tensor

def stratified_split_tensors(input_ids, attention_mask, labels, test_size=0.10):
    # Convert to numpy for stratified split by labels
    labels_np = labels.numpy()
    idx = np.arange(len(labels_np))
    train_idx, val_idx = train_test_split(
        idx, test_size=test_size, random_state=42, stratify=labels_np
    )
    train_ds = TensorDataset(input_ids[train_idx], attention_mask[train_idx], labels[train_idx])
    val_ds = TensorDataset(input_ids[val_idx], attention_mask[val_idx], labels[val_idx])
    print(f"📊 Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_ds, val_ds

def compute_class_weights(labels):
    counts = Counter(labels.tolist())
    classes = sorted(counts.keys())
    freqs = np.array([counts[c] for c in classes], dtype=np.float32)
    weights = 1.0 / (freqs + 1e-12)
    weights = weights / np.mean(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    print(f"🔧 Class weights (normalized): {weights_tensor.numpy()}")
    return weights_tensor

# -------------------------
# TRAIN / EVAL
# -------------------------
def evaluate(model, dataloader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            trues.extend(labels.cpu().numpy().tolist())
    acc = (np.array(preds) == np.array(trues)).mean() if len(trues) > 0 else 0.0
    return acc

def train():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")

    df = load_and_check(DATA_PATH)
    df = cap_per_class(df, MAX_SAMPLES_PER_CLASS)

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # Tokenize all data (faster at training time)
    input_ids, attention_mask, labels = prepare_tokenized_tensors(df, tokenizer, MAX_LENGTH)

    # Stratified split into train/val (tensors)
    train_ds, val_ds = stratified_split_tensors(input_ids, attention_mask, labels, test_size=0.10)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    num_labels = df['multi_class_label'].nunique()
    print(f"🤖 Num labels: {num_labels}")

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(device)

    # compute class weights on training labels
    train_labels_list = [int(x[2]) for x in train_ds]  # quick extraction
    class_weights = compute_class_weights(torch.tensor(train_labels_list))
    class_weights = class_weights.to(device)

    # replace loss function: use weighted CrossEntropyLoss inside training loop
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_PCT))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"▶️ Config -> epochs: {EPOCHS}, batch_size: {BATCH_SIZE}, max_len: {MAX_LENGTH}, total_steps: {total_steps}, AMP: {use_amp}")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        all_preds = []
        all_trues = []

        for step, batch in enumerate(train_loader, start=1):
            input_ids_b, attention_mask_b, labels_b = [b.to(device) for b in batch]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(input_ids=input_ids_b, attention_mask=attention_mask_b)
                logits = outputs.logits
                loss = loss_fn(logits, labels_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(labels_b.detach().cpu().numpy().tolist())

            if step % 30 == 0 or step == len(train_loader):
                acc_so_far = (np.array(all_preds) == np.array(all_trues)).mean() if len(all_trues) > 0 else 0.0
                avg_loss = running_loss / step
                elapsed = int(time.time() - epoch_start)
                print(f"Epoch {epoch}/{EPOCHS} - Step {step}/{len(train_loader)} | Loss: {avg_loss:.4f} | Acc_so_far: {acc_so_far:.4f} | Elapsed: {elapsed}s")

        # End epoch: evaluate on validation
        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"✅ Epoch {epoch} done. Avg Loss: {avg_epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {int(epoch_time)}s")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(MODEL_OUTPUT_DIR)
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
            meta = {
                'model': MODEL_NAME,
                'num_labels': num_labels,
                'epoch_saved': epoch,
                'val_acc': val_acc,
                'batch_size': BATCH_SIZE,
                'max_length': MAX_LENGTH
            }
            with open(os.path.join(MODEL_OUTPUT_DIR, "training_metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"💾 Modèle sauvegardé -> {MODEL_OUTPUT_DIR}")
        else:
            epochs_no_improve += 1
            print(f"ℹ️ Pas d'amélioration (patience {epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")

        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("⛔ Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"\n🎉 Training terminé. Best val acc: {best_val_acc:.4f}")
    print(f"⏱️ Temps total: {int(total_time)}s ({total_time/60:.1f} min)")

# -------------------------
# EVALUATION RAPIDE (après entraînement)
# -------------------------
def evaluate_saved_model(test_csv_path="./data/nlp_test.csv", model_dir=MODEL_OUTPUT_DIR):
    if not os.path.exists(model_dir):
        print("❌ Aucun modèle sauvegardé trouvé.")
        return None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    df = pd.read_csv(test_csv_path)
    df = df.dropna(subset=['text_features', 'multi_class_label']).reset_index(drop=True)
    texts = df['text_features'].astype(str).tolist()
    labels = df['multi_class_label'].astype(int).tolist()

    print(f"🔎 Test samples: {len(texts)}")

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            out = model(**enc)
            p = torch.argmax(out.logits, dim=1).cpu().numpy()
            preds.extend(p.tolist())
    acc = (np.array(preds) == np.array(labels)).mean()
    print(f"📊 Test Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, digits=4))
    return acc

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    try:
        train()
        # Après training, évaluer automatiquement si on a un modèle sauvegardé
        print("\n--- ÉVALUATION DU MODELE SAUVÉ ---")
        evaluate_saved_model()
    except Exception as ex:
        print("❌ Erreur:", ex)
        import traceback
        traceback.print_exc()
