import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def prepare_nlp_features(df):
    """
    Prépare des features texte RICHES et INFORMATIVES pour l'entraînement NLP
    """
    print("=" * 60)
    print("ÉTAPE 4: PRÉPARATION DES FEATURES NLP AVANCÉES")
    print("=" * 60)
    
    print("📝 Création des features texte enrichies...")
    
    # Créer des features texte détaillées pour DistilBERT
    text_features = []
    attack_descriptions = []
    
    for idx, row in df.iterrows():
        # PARTIE 1: Features techniques détaillées
        text_parts = []
        
        # 1. Informations de protocole et service (CRITIQUE)
        if 'proto' in df.columns:
            proto_mapping = {1: 'TCP', 2: 'UDP', 3: 'ICMP', 4: 'Other'}
            proto_name = proto_mapping.get(row['proto'], f'Protocol_{row["proto"]}')
            text_parts.append(f"Protocol:{proto_name}")
        
        if 'service' in df.columns:
            service_mapping = {0: 'HTTP', 1: 'FTP', 2: 'SSH', 3: 'DNS', 4: 'SMTP', 5: 'Other'}
            service_name = service_mapping.get(row['service'], f'Service_{row["service"]}')
            text_parts.append(f"Service:{service_name}")
        
        if 'state' in df.columns:
            state_mapping = {0: 'ESTABLISHED', 1: 'SYN_SENT', 2: 'SYN_RECV', 3: 'FIN_WAIT', 4: 'CLOSED'}
            state_name = state_mapping.get(row['state'], f'State_{row["state"]}')
            text_parts.append(f"Connection_State:{state_name}")
        
        # 2. Patterns de trafic réseau (TRÈS IMPORTANT)
        if 'sbytes' in df.columns and 'dbytes' in df.columns:
            sbytes, dbytes = row['sbytes'], row['dbytes']
            
            # Détection de patterns suspects
            if sbytes > 10000:
                text_parts.append("High_Outbound_Traffic")
            if dbytes > 10000:
                text_parts.append("High_Inbound_Traffic")
            if sbytes == 0 and dbytes > 0:
                text_parts.append("One_Way_Communication")
            if sbytes > 0 and dbytes == 0:
                text_parts.append("Outbound_Only")
            if 50 <= sbytes <= 1000 and 50 <= dbytes <= 1000:
                text_parts.append("Balanced_Communication")
        
        # 3. Analyse de durée et timing
        if 'dur' in df.columns:
            duration = row['dur']
            if duration < 0.1:
                text_parts.append("Very_Short_Connection")
            elif duration > 60:
                text_parts.append("Long_Lived_Connection")
            elif duration > 300:
                text_parts.append("Suspicious_Long_Session")
            else:
                text_parts.append(f"Duration:{duration:.1f}s")
        
        # 4. Patterns de packets (DÉTECTION D'ANOMALIES)
        if 'spkts' in df.columns and 'dpkts' in df.columns:
            spkts, dpkts = row['spkts'], row['dpkts']
            
            if spkts > dpkts * 10:
                text_parts.append("Heavy_Outbound_Packets")
            elif dpkts > spkts * 10:
                text_parts.append("Heavy_Inbound_Packets")
            elif spkts == 1 and dpkts == 0:
                text_parts.append("Single_Probe_Packet")
        
        # 5. Flags TCP (ANALYSE COMPORTEMENTALE)
        if 'Sload' in df.columns and 'Dload' in df.columns:
            sload, dload = row['Sload'], row['Dload']
            
            if sload > 1000000:  # 1 Mbps
                text_parts.append("High_Source_Load")
            if dload > 1000000:
                text_parts.append("High_Destination_Load")
        
        # PARTIE 2: Description comportementale (pour le contexte NLP)
        behavior_parts = []
        
        # Détection de scans
        if 'ct_srv_src' in df.columns and row['ct_srv_src'] > 10:
            behavior_parts.append("Multiple_Services_From_Source")
        
        if 'ct_src_dport_ltm' in df.columns and row['ct_src_dport_ltm'] > 5:
            behavior_parts.append("Multiple_Destination_Ports")
        
        # Détection de comportements DoS-like
        if 'ct_dst_src_ltm' in df.columns and row['ct_dst_src_ltm'] > 20:
            behavior_parts.append("Multiple_Connections_To_Same_Destination")
        
        # Analyse de la charge
        if 'Sload' in df.columns and row['Sload'] > 5000000:  # 5 Mbps
            behavior_parts.append("Potential_DoS_Outbound")
        if 'Dload' in df.columns and row['Dload'] > 5000000:
            behavior_parts.append("Potential_DoS_Inbound")
        
        # COMBINAISON FINALE
        technical_context = " ".join(text_parts)
        behavioral_context = " ".join(behavior_parts) if behavior_parts else "Normal_Behavior"
        
        # Feature texte finale très riche
        final_text = f"Network_Activity: {technical_context}. Behavioral_Pattern: {behavioral_context}"
        text_features.append(final_text)
        
        # Description d'attaque pour contexte supplémentaire
        if row['Label'] == 1:  # Si c'est une attaque
            attack_desc = "MALICIOUS_ACTIVITY: " + final_text
        else:
            attack_desc = "NORMAL_ACTIVITY: " + final_text
        
        attack_descriptions.append(attack_desc)
    
    df['text_features'] = text_features
    df['attack_context'] = attack_descriptions
    
    # ANALYSE DES FEATURES CRÉÉES
    print(f"✅ Features texte enrichies créées.")
    print(f"📊 Statistiques des features:")
    
    feature_lengths = [len(text.split()) for text in text_features]
    print(f"   Mots par échantillon: {np.mean(feature_lengths):.1f} (min: {np.min(feature_lengths)}, max: {np.max(feature_lengths)})")
    
    # Afficher des exemples selon le label
    normal_examples = df[df['Label'] == 0]['text_features'].head(2)
    attack_examples = df[df['Label'] == 1]['text_features'].head(2)
    
    print(f"\n🔍 Exemples NORMaux:")
    for i, example in enumerate(normal_examples):
        print(f"   {i+1}. {example[:120]}...")
    
    print(f"\n🚨 Exemples ATTAQUE:")
    for i, example in enumerate(attack_examples):
        print(f"   {i+1}. {example[:120]}...")
    
    return df

def analyze_feature_importance(df):
    """
    Analyse l'importance des différentes features pour la détection
    """
    print("\n🔍 Analyse de l'importance des features...")
    
    # Features critiques pour la détection
    critical_features = [
        'proto', 'service', 'state', 'sbytes', 'dbytes', 'dur',
        'spkts', 'dpkts', 'Sload', 'Dload', 'ct_srv_src', 'ct_dst_src_ltm'
    ]
    
    available_features = [f for f in critical_features if f in df.columns]
    
    print(f"📋 Features disponibles pour l'analyse: {available_features}")
    
    # Analyser la corrélation avec le label
    if available_features:
        correlation_data = []
        for feature in available_features[:8]:  # Prendre les 8 premières
            if pd.api.types.is_numeric_dtype(df[feature]):
                corr = np.corrcoef(df[feature], df['Label'])[0, 1]
                correlation_data.append({'Feature': feature, 'Correlation': abs(corr)})
        
        if correlation_data:
            corr_df = pd.DataFrame(correlation_data).sort_values('Correlation', ascending=False)
            print(f"📈 Corrélation avec le label (top {len(corr_df)}):")
            for _, row in corr_df.head(5).iterrows():
                print(f"   {row['Feature']:15}: {row['Correlation']:.3f}")

def prepare_train_test_split(df):
    """
    Prépare la division train/validation/test avec stratification
    """
    print("\n🎯 Préparation des splits train/validation/test...")
    
    # Vérifier la distribution des labels
    label_dist = df['Label'].value_counts()
    print(f"Distribution des labels: {label_dist.to_dict()}")
    
    # Division stratifiée
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3, 
        random_state=42, 
        stratify=df['Label']
    )
    
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42, 
        stratify=temp_df['Label']
    )
    
    print(f"✅ Division terminée:")
    print(f"   Train:      {len(train_df)} échantillons")
    print(f"   Validation: {len(val_df)} échantillons")
    print(f"   Test:       {len(test_df)} échantillons")
    
    # Vérifier la distribution dans chaque split
    print(f"\n📊 Distribution dans chaque split:")
    for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
        dist = split_df['Label'].value_counts()
        print(f"   {split_name:12} - Normal: {dist.get(0, 0):>5}, Attack: {dist.get(1, 0):>5}")
    
    return train_df, val_df, test_df

def save_nlp_data(train_df, val_df, test_df):
    """
    Sauvegarde les données préparées pour NLP
    """
    print("\n💾 Sauvegarde des données NLP enrichies...")
    
    # Créer le dossier si nécessaire
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # Sauvegarder les splits avec les nouvelles features
    columns_to_save = ['text_features', 'attack_context', 'Label']
    available_columns = [col for col in columns_to_save if col in train_df.columns]
    
    train_df[available_columns].to_csv('./data/nlp_train.csv', index=False)
    val_df[available_columns].to_csv('./data/nlp_val.csv', index=False)
    test_df[available_columns].to_csv('./data/nlp_test.csv', index=False)
    
    print("✅ Données NLP enrichies sauvegardées:")
    print(f"   nlp_train.csv: {len(train_df)} échantillons")
    print(f"   nlp_val.csv:   {len(val_df)} échantillons")
    print(f"   nlp_test.csv:  {len(test_df)} échantillons")
    
    # Sauvegarder les métadonnées détaillées
    metadata = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': {
            'train': train_df['Label'].value_counts().to_dict(),
            'val': val_df['Label'].value_counts().to_dict(),
            'test': test_df['Label'].value_counts().to_dict()
        },
        'text_feature_stats': {
            'average_words_per_sample': np.mean([len(text.split()) for text in train_df['text_features']]),
            'average_chars_per_sample': np.mean([len(text) for text in train_df['text_features']]),
            'vocabulary_size': len(set(' '.join(train_df['text_features']).split())),
            'feature_quality': 'HIGH'  # Nos features sont maintenant très informatives
        },
        'feature_engineering': {
            'technical_features_included': True,
            'behavioral_patterns_included': True,
            'traffic_analysis_included': True,
            'attack_context_included': True
        }
    }
    
    with open('./results/nlp_preparation_report.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("📊 Métadonnées détaillées sauvegardées")

def main():
    """
    EXÉCUTION PRINCIPALE - PRÉPARATION NLP AVANCÉE
    """
    # Charger les données équilibrées
    print("📥 Chargement des données équilibrées...")
    df = pd.read_csv('./data/UNSW-NB15_undersampled.csv')
    print(f"📊 Dataset équilibré: {df.shape}")
    
    # Analyser les features disponibles
    print(f"\n📋 Colonnes disponibles: {df.columns.tolist()}")
    
    # Préparer les features NLP enrichies
    df_nlp = prepare_nlp_features(df)
    
    # Analyser l'importance des features
    analyze_feature_importance(df_nlp)
    
    # Préparer les splits
    train_df, val_df, test_df = prepare_train_test_split(df_nlp)
    
    # Sauvegarder les données NLP
    save_nlp_data(train_df, val_df, test_df)
    
    # Sauvegarder le dataset complet NLP-ready
    df_nlp.to_csv('./data/UNSW-NB15_nlp_ready.csv', index=False)
    print("💾 Dataset NLP-ready enrichi sauvegardé: UNSW-NB15_nlp_ready.csv")
    
    print(f"\n{'✅'*20}")
    print("ÉTAPE 4 TERMINÉE AVEC SUCCÈS!")
    print("🤖 Données ENRICHIES prêtes pour l'entraînement DistilBERT!")
    print("🎯 Features techniques + comportementales + contextuelles")
    print(f"{'✅'*20}")

if __name__ == "__main__":
    main()