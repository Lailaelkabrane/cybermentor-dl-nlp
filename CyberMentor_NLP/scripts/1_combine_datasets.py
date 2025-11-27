import pandas as pd
import numpy as np
import os

def combine_unsw_datasets():
    """
    Étape 1: Combine les 4 fichiers UNSW-NB15 en un seul dataset
    """
    print("=" * 60)
    print("ÉTAPE 1: COMBINAISON DES DATASETS UNSW-NB15")
    print("=" * 60)
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs('./data', exist_ok=True)
    
    # Noms de colonnes standards UNSW-NB15
    standard_columns = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
        'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
    ]
    
    print("📥 Chargement des 4 fichiers...")
    dataframes = []
    
    for i in range(1, 5):
        file_path = f'./data/UNSW-NB15_{i}.csv'
        try:
            # Essayer avec les noms de colonnes standards
            df = pd.read_csv(file_path, names=standard_columns, low_memory=False)
            
            # Vérifier si la première ligne est un header
            first_row = df.iloc[0]
            if any(isinstance(x, str) and any(keyword in x.lower() for keyword in ['srcip', 'proto', 'service']) for x in first_row):
                df = df.iloc[1:].reset_index(drop=True)  # Supprimer le header
                
            dataframes.append(df)
            print(f"✅ Fichier {i} chargé: {df.shape}")
            
        except Exception as e:
            print(f"❌ Erreur avec le fichier {i}: {e}")
            # Essayer sans noms de colonnes
            try:
                df = pd.read_csv(file_path, low_memory=False)
                dataframes.append(df)
                print(f"✅ Fichier {i} chargé (sans noms standards): {df.shape}")
            except Exception as e2:
                print(f"❌ Échec complet du fichier {i}: {e2}")
    
    if not dataframes:
        print("❌ Aucun fichier chargé avec succès!")
        return None
    
    # Combiner tous les dataframes
    print("\n🔗 Combinaison des datasets...")
    df_combined = pd.concat(dataframes, ignore_index=True)
    
    # Si nous n'avons pas pu utiliser les noms standards, utiliser les colonnes existantes
    if len(df_combined.columns) != len(standard_columns):
        print(f"⚠️  Nombre de colonnes différent: {len(df_combined.columns)} au lieu de {len(standard_columns)}")
        print(f"Colonnes actuelles: {df_combined.columns.tolist()}")
    
    print(f"📊 Dataset combiné: {df_combined.shape}")
    print(f"🔢 Mémoire utilisée: {df_combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Sauvegarder
    df_combined.to_csv('./data/UNSW-NB15_combined.csv', index=False)
    print("💾 Dataset combiné sauvegardé: UNSW-NB15_combined.csv")
    
    return df_combined

if __name__ == "__main__":
    combine_unsw_datasets()