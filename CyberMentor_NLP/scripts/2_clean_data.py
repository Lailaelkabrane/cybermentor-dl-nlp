import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def find_correct_label_column(df):
    """
    Trouve la VRAIE colonne de label - VERSION CORRIGÉE
    """
    print("\n🔎 RECHERCHE INTELLIGENTE DE LA BONNE COLONNE LABEL")
    print("=" * 60)
    
    # PRIORITÉ ABSOLUE: Colonnes avec 'Label' ou 'label' dans le nom
    label_cols = [col for col in df.columns if col.lower() == 'label']
    
    if label_cols:
        best_col = label_cols[0]
        dist = df[best_col].value_counts()
        print(f"✅ COLONNE 'Label' TROUVÉE: {best_col}")
        print(f"   Distribution: {dist.to_dict()}")
        print(f"   Taux d'attaque: {(dist.get(1, 0)/len(df))*100:.2f}%")
        return best_col
    
    # Fallback: autres colonnes binaires
    binary_cols = []
    for col in df.columns:
        if df[col].nunique() == 2:
            binary_cols.append(col)
    
    if binary_cols:
        # Préférer les colonnes avec 'attack' dans le nom
        attack_cols = [col for col in binary_cols if 'attack' in col.lower()]
        if attack_cols:
            best_col = attack_cols[0]
        else:
            best_col = binary_cols[0]
        
        print(f"⚠️  Colonne binaire sélectionnée: {best_col}")
        return best_col
    
    # Dernier recours
    return df.columns[-1]

def analyze_real_labels(df):
    """
    Analyse la VRAIE distribution des labels
    """
    print("\n🎯 ANALYSE DE LA VÉRITABLE DISTRIBUTION")
    print("=" * 50)
    
    # VÉRITABLE colonne Label
    label_col = 'Label'
    attack_cat_col = 'attack_cat'
    
    if label_col in df.columns:
        label_dist = df[label_col].value_counts()
        print(f"📊 DISTRIBUTION RÉELLE DE '{label_col}':")
        total = len(df)
        
        for val, count in label_dist.items():
            percentage = (count / total) * 100
            status = "NORMAL" if val == 0 else "ATTAQUE"
            print(f"   {status}: {count:>8} ({percentage:6.2f}%)")
        
        attack_rate = (label_dist.get(1, 0) / total) * 100
        print(f"   → Taux d'attaque réel: {attack_rate:.2f}%")
    
    if attack_cat_col in df.columns:
        attack_cat_dist = df[attack_cat_col].value_counts()
        print(f"\n🔥 TYPES D'ATTAQUES DÉTECTÉS:")
        for attack_type, count in attack_cat_dist.items():
            print(f"   - {attack_type}: {count}")
    
    return label_col


def clean_and_prepare_for_nlp(df, label_col):
    """
    Nettoie et prépare pour NLP avec la BONNE colonne label
    """
    print("\n🧹 NETTOYAGE AVEC BON LABEL")
    print("=" * 35)
    
    df_clean = df.copy()
    
    # 1. Supprimer doublons
    initial_size = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"✅ Doublons supprimés: {initial_size - len(df_clean)}")
    
    # 2. Remplir valeurs manquantes
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col].fillna('unknown', inplace=True)
            else:
                df_clean[col].fillna(0, inplace=True)
    
    # 3. Encoder colonnes catégorielles
    categorical_cols = ['proto', 'service', 'state']
    encoders = {}
    
    for col in categorical_cols:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le
    
    print(f"✅ Colonnes encodées: {categorical_cols}")
    
    # 4. Créer features texte pour NLP
    print("\n📝 CRÉATION DES FEATURES TEXTE POUR NLP...")
    
    # Combinaison intelligente des features
    text_features = []
    for idx, row in df_clean.iterrows():
        text_parts = []
        
        # Features principales
        if 'proto' in df_clean.columns:
            text_parts.append(f"proto_{row['proto']}")
        if 'service' in df_clean.columns:
            text_parts.append(f"service_{row['service']}")
        if 'state' in df_clean.columns:
            text_parts.append(f"state_{row['state']}")
        if 'attack_cat' in df_clean.columns and pd.notna(row['attack_cat']):
            text_parts.append(f"attack_{str(row['attack_cat']).strip()}")
        
        text_features.append(" ".join(text_parts))
    
    df_clean['text_features'] = text_features
    
    print(f"✅ Features texte créées. Exemple:")
    print(f"   {text_features[0][:100]}...")
    
    return df_clean, encoders

def main():
    """
    EXÉCUTION PRINCIPALE - VERSION CORRIGÉE
    """
    print("=" * 70)
    print("ÉTAPE 2: NETTOYAGE - VERSION CORRIGÉE (BON LABEL)")
    print("=" * 70)
    
    # 1. Charger les données
    print("📥 Chargement des données...")
    df = pd.read_csv('./data/UNSW-NB15_combined.csv', low_memory=False)
    print(f"📊 Taille originale: {df.shape}")
    
    # 2. Trouver la VRAIE colonne label
    label_col = analyze_real_labels(df)
    
    # 3. Distribution FINALE CORRECTE
    final_dist = df[label_col].value_counts()
    print(f"\n🎯 DISTRIBUTION OFFICIELLE:")
    total = len(df)
    normal_count = final_dist.get(0, 0)
    attack_count = final_dist.get(1, 0)
    
    print(f"   NORMAL (0):  {normal_count:>8} ({(normal_count/total)*100:6.2f}%)")
    print(f"   ATTACK (1):  {attack_count:>8} ({(attack_count/total)*100:6.2f}%)")
    print(f"   → Taux d'attaque: {(attack_count/total)*100:.2f}%")
    
    # 5. Nettoyer avec BON label
    df_clean, encoders = clean_and_prepare_for_nlp(df, label_col)
    
    # 6. Sauvegarder
    df_clean.to_csv('./data/UNSW-NB15_cleaned.csv', index=False)
    print("💾 Données nettoyées sauvegardées: UNSW-NB15_cleaned.csv")
    
    # 7. Rapport final CORRECT
    report = {
        'dataset_info': {
            'original_shape': list(df.shape),
            'cleaned_shape': list(df_clean.shape),
            'correct_label_column': label_col,
            'attack_rate_percentage': round((attack_count/total)*100, 2)
        },
        'label_distribution': {
            'normal_count': int(normal_count),
            'attack_count': int(attack_count),
            'normal_percentage': round((normal_count/total)*100, 2),
            'attack_percentage': round((attack_count/total)*100, 2)
        },
        'attack_types': df['attack_cat'].value_counts().to_dict() if 'attack_cat' in df.columns else {}
    }
    
    with open('./results/cleaning_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📊 Rapport CORRECT sauvegardé: cleaning_report.json")
    
    # 8. Message final
    print(f"\n{'✅'*20}")
    print("ÉTAPE 2 TERMINÉE AVEC SUCCÈS!")
    print(f"📈 Taux d'attaque réel: {(attack_count/total)*100:.2f}%")
    print(f"📁 Fichier nettoyé: UNSW-NB15_cleaned.csv")
    print(f"{'✅'*20}")

if __name__ == "__main__":
    main()