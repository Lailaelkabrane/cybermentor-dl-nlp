"""
MODEL EVALUATION SCRIPT (MULTI-CLASS) - CYBERMENTOR AI
------------------------------------------------------
Evaluates the trained DistilBERT model:
    ./models/cybermentor_distilbert_mix

Generates:
 - Confusion Matrix (PNG)
 - Per-class Metrics Bar Chart (PNG)
 - F1/Precision/Recall Comparison (PNG)
 - Classification Report (text + json)
 - Full structured evaluation report
"""

import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import json
import os


class MultiClassEvaluator:
    def __init__(self, model_dir="./models/cybermentor_distilbert_mix"):
        self.model_dir = model_dir
        
        print(f"📥 Loading model from: {model_dir}")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"🖥️ Device: {self.device}")
        print("✅ Model loaded successfully.\n")

    # -----------------------------------------------------
    def load_test_data(self, path="./data/nlp_test.csv"):
        print("📥 Loading test dataset...")
        df = pd.read_csv(path)

        if "multi_class_label" not in df.columns:
            raise KeyError("❌ Column 'multi_class_label' not found in test CSV.")

        print(f"📊 Test samples: {len(df)}")
        print("📈 Label distribution:")
        print(df["multi_class_label"].value_counts().sort_index())

        return df

    # -----------------------------------------------------
    def predict_batch(self, texts, batch_size=32):
        predictions = []

        print("🔮 Running predictions...")

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            encoding = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoding).logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            predictions.extend(preds)

            if i % (batch_size * 10) == 0:
                print(f"  Processed: {min(i + batch_size, len(texts))}/{len(texts)}")

        return predictions

    # -----------------------------------------------------
    def evaluate(self):
        df = self.load_test_data()

        texts = df["text_features"].astype(str).tolist()
        labels = df["multi_class_label"].astype(int).tolist()

        predictions = self.predict_batch(texts)

        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)

        print("\n📊 FINAL EVALUATION")
        print("=" * 50)
        print(f"🔥 Accuracy: {accuracy:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(labels, predictions))

        return labels, predictions, report, accuracy

    # -----------------------------------------------------
    def plot_confusion_matrix(self, true_labels, preds, class_names, filename="./results/confusion_matrix.png"):
        cm = confusion_matrix(true_labels, preds)

        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix - CyberMentor AI")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"📌 Confusion matrix saved: {filename}")

    # -----------------------------------------------------
    def plot_class_metrics(self, report, class_names, filename="./results/class_metrics.png"):
        precision = [report[str(i)]["precision"] for i in range(len(class_names))]
        recall = [report[str(i)]["recall"] for i in range(len(class_names))]
        f1 = [report[str(i)]["f1-score"] for i in range(len(class_names))]

        x = np.arange(len(class_names))
        width = 0.25

        plt.figure(figsize=(12, 6))
        plt.bar(x - width, precision, width, label="Precision")
        plt.bar(x, recall, width, label="Recall")
        plt.bar(x + width, f1, width, label="F1-Score")

        plt.xticks(x, class_names, rotation=45)
        plt.ylabel("Score")
        plt.title("Per-Class Evaluation Metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"📌 Class metrics chart saved: {filename}")

    # -----------------------------------------------------
    def save_report(self, report, accuracy, true_labels, preds, filename="./results/evaluation_report.json"):
        class_count = pd.Series(true_labels).value_counts().sort_index().to_dict()

        full_report = {
            "accuracy": float(accuracy),
            "class_counts": class_count,
            "classification_report": report,
            "model_path": self.model_dir
        }

        with open(filename, "w") as f:
            json.dump(full_report, f, indent=4)

        print(f"📄 Report saved: {filename}")

    # -----------------------------------------------------
    def run_full_evaluation(self):
        os.makedirs("./results", exist_ok=True)

        true_labels, preds, report, accuracy = self.evaluate()

        class_names = [str(i) for i in sorted(report.keys()) if i.isdigit()]

        # Plots
        self.plot_confusion_matrix(true_labels, preds, class_names)
        self.plot_class_metrics(report, class_names)

        # Save JSON report
        self.save_report(report, accuracy, true_labels, preds)

        print("\n🎉 Evaluation completed!")
        print("📁 All results saved in ./results/")


# -----------------------------------------------------
def main():
    print("=" * 60)
    print("📌 CYBERMENTOR NLP — MULTI-CLASS MODEL EVALUATION")
    print("=" * 60)

    evaluator = MultiClassEvaluator(
        model_dir="./models/cybermentor_distilbert_mix"
    )

    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
