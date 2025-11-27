import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from collections import Counter
import json
import os

def analyze_imbalance(df):
    """
    Analyse détaillée du déséquilibre des classes
    """
    print("=" * 60)
    print("ÉTAPE 3: GESTION DU DÉSÉQUILIBRE DES CLASSES")
    print("=" * 60)
    
    label_col = 'Label'
    label_dist = df[label_col].value_counts()
    
    print("📊 ANALYSE DU DÉSÉQUILIBRE:")
    total = len(df)
    normal_count = label_dist[0]
    attack_count = label_dist[1]
    
    print(f"   Normal (0):  {normal_count:>8} ({(normal_count/total)*100:6.2f}%)")
    print(f"   Attack (1):  {attack_count:>8} ({(attack_count/total)*100:6.2f}%)")
    print(f"   Ratio déséquilibre: {normal_count/attack_count:.1f}:1")
    
    return label_dist

def apply_smote_balancing(df):
    """
    Applique SMOTE pour équilibrer les classes
    """
    print("\n🔄 APPLICATION DE SMOTE (Over-sampling)...")
    
    # Séparer features et target
    X = df.drop(['Label', 'text_features', 'attack_cat'], axis=1, errors='ignore')
    y = df['Label']
    
    # Appliquer SMOTE
    smote = SMOTE(random_state=42, sampling_strategy='auto')
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"✅ SMOTE appliqué:")
    print(f"   Avant: {Counter(y)}")
    print(f"   Après: {Counter(y_balanced)}")
    
    # Recréer le DataFrame équilibré
    df_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    df_balanced['Label'] = y_balanced
    
    # Réintégrer les colonnes texte si elles existent
    if 'text_features' in df.columns:
        # Pour SMOTE, on duplique les text_features des samples minoritaires
        minority_texts = df[df['Label'] == 1]['text_features'].values
        n_needed = len(df_balanced) - len(df)
        repeated_texts = np.random.choice(minority_texts, size=n_needed, replace=True)
        
        all_texts = np.concatenate([df['text_features'].values, repeated_texts])
        df_balanced['text_features'] = all_texts
    
    return df_balanced

def apply_undersampling(df):
    """
    Applique l'under-sampling pour équilibrer les classes
    """
    print("\n🔄 APPLICATION DE L'UNDER-SAMPLING...")
    
    # Séparer les classes
    df_majority = df[df['Label'] == 0]
    df_minority = df[df['Label'] == 1]
    
    # Under-sample la classe majoritaire
    df_majority_undersampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_minority),
        random_state=42
    )
    
    # Combiner
    df_balanced = pd.concat([df_majority_undersampled, df_minority])
    
    print(f"✅ Under-sampling appliqué:")
    print(f"   Normal: {len(df_majority_undersampled)}")
    print(f"   Attack: {len(df_minority)}")
    print(f"   Total: {len(df_balanced)}")
    
    return df_balanced

def apply_class_weights_strategy(df):
    """
    Stratégie avec poids de classes (sans modifier les données)
    """
    print("\n⚖️  CALCUL DES POIDS DE CLASSES...")
    
    label_dist = df['Label'].value_counts()
    total = len(df)
    
    # Calcul des poids (inverse proportionnel à la fréquence)
    weight_attack = total / (2 * label_dist[1])
    weight_normal = total / (2 * label_dist[0])
    
    class_weights = {
        0: weight_normal,
        1: weight_attack
    }
    
    print(f"✅ Poids de classes calculés:")
    print(f"   Poids Normal (0): {weight_normal:.2f}")
    print(f"   Poids Attack (1): {weight_attack:.2f}")
    
    return class_weights

def compare_balancing_methods(df):
    """
    Compare les différentes méthodes d'équilibrage
    """
    print("\n🔍 COMPARAISON DES MÉTHODES D'ÉQUILIBRAGE")
    print("=" * 45)
    
    original_size = len(df)
    original_ratio = df['Label'].value_counts()[0] / df['Label'].value_counts()[1]
    
    methods = {}
    
    # 1. SMOTE
    try:
        df_smote = apply_smote_balancing(df.copy())
        methods['smote'] = {
            'df': df_smote,
            'size': len(df_smote),
            'ratio': 1.0,  # Parfaitement équilibré
            'description': 'SMOTE (Over-sampling) - Crée de nouvelles instances synthétiques'
        }
    except Exception as e:
        print(f"❌ SMOTE échoué: {e}")
    
    # 2. Under-sampling
    df_undersample = apply_undersampling(df.copy())
    methods['undersampling'] = {
        'df': df_undersample,
        'size': len(df_undersample),
        'ratio': 1.0,
        'description': 'Under-sampling - Réduit la classe majoritaire'
    }
    
    # 3. Class Weights
    class_weights = apply_class_weights_strategy(df)
    methods['class_weights'] = {
        'weights': class_weights,
        'description': 'Poids de classes - Pénalise plus les erreurs sur la classe minoritaire'
    }
    
    # Affichage comparatif
    print(f"\n📈 COMPARAISON:")
    print(f"   Original:     {original_size:>6} samples, ratio {original_ratio:.1f}:1")
    
    for method, info in methods.items():
        if 'df' in info:
            print(f"   {method:12} {info['size']:>6} samples, ratio 1:1")
        else:
            print(f"   {method:12} {'N/A':>6} (poids seulement)")
    
    return methods

def save_balancing_report(original_df, balanced_methods):
    """
    Sauvegarde un rapport détaillé de l'équilibrage
    """
    report = {
        'original_dataset': {
            'total_samples': len(original_df),
            'normal_samples': int(original_df['Label'].value_counts()[0]),
            'attack_samples': int(original_df['Label'].value_counts()[1]),
            'imbalance_ratio': float(original_df['Label'].value_counts()[0] / original_df['Label'].value_counts()[1]),
            'attack_rate_percentage': float((original_df['Label'].value_counts()[1] / len(original_df)) * 100)
        },
        'balancing_methods': {}
    }
    
    for method_name, method_info in balanced_methods.items():
        if 'df' in method_info:
            df_balanced = method_info['df']
            report['balancing_methods'][method_name] = {
                'total_samples': len(df_balanced),
                'normal_samples': int(df_balanced['Label'].value_counts()[0]),
                'attack_samples': int(df_balanced['Label'].value_counts()[1]),
                'balance_ratio': 1.0,
                'description': method_info['description']
            }
        else:
            report['balancing_methods'][method_name] = {
                'class_weights': method_info['weights'],
                'description': method_info['description']
            }
    
    # Recommandations
    report['recommendations'] = {
        'for_large_datasets': 'SMOTE (conserve toute l\'information)',
        'for_speed': 'Under-sampling (plus rapide)',
        'for_deep_learning': 'Class Weights (pas de modification des données)',
        'warning': 'SMOTE peut créer du bruit avec des données très déséquilibrées'
    }
    
    with open('./results/balancing_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📊 Rapport d'équilibrage sauvegardé")

def main():
    """
    EXÉCUTION PRINCIPALE - GESTION DU DÉSÉQUILIBRE
    """
    # Charger les données nettoyées
    print("📥 Chargement des données nettoyées...")
    df = pd.read_csv('./data/UNSW-NB15_cleaned.csv')
    print(f"📊 Dataset nettoyé: {df.shape}")
    
    # Analyser le déséquilibre
    label_dist = analyze_imbalance(df)
    
    # Comparer les méthodes
    balancing_methods = compare_balancing_methods(df)
    
    # Sauvegarder rapport
    save_balancing_report(df, balancing_methods)
    
    # Sauvegarder le dataset SMOTE (recommandé)
    if 'smote' in balancing_methods and 'df' in balancing_methods['smote']:
        df_smote = balancing_methods['smote']['df']
        df_smote.to_csv('./data/UNSW-NB15_balanced.csv', index=False)
        print("💾 Dataset équilibré (SMOTE) sauvegardé: UNSW-NB15_balanced.csv")
    
    # Sauvegarder aussi l'under-sampling (alternative)
    if 'undersampling' in balancing_methods and 'df' in balancing_methods['undersampling']:
        df_undersample = balancing_methods['undersampling']['df']
        df_undersample.to_csv('./data/UNSW-NB15_undersampled.csv', index=False)
        print("💾 Dataset under-sampled sauvegardé: UNSW-NB15_undersampled.csv")
    
    print(f"\n{'✅'*20}")
    print("ÉTAPE 3 TERMINÉE AVEC SUCCÈS!")
    print("📈 Données maintenant équilibrées pour l'entraînement")
    print(f"{'✅'*20}")

if __name__ == "__main__":
    main()