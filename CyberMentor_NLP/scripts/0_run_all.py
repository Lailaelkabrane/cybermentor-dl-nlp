"""
PIPELINE COMPLET CYBERMENTOR NLP
================================

Script principal pour exécuter tout le pipeline NLP de CyberMentor :
1. Combinaison des datasets UNSW-NB15
2. Nettoyage des données
3. Équilibrage des classes
4. Préparation des features NLP avancées
5. Entraînement du modèle DistilBERT
6. Évaluation du modèle

Auteur: CyberMentor AI Team
Version: 2.0
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, description):
    """
    Exécute un script Python et gère les erreurs
    """
    print(f"\n{'='*60}")
    print(f"🚀 EXÉCUTION: {description}")
    print(f"📄 Script: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Exécuter le script
        result = subprocess.run(
            [sys.executable, f"./scripts/{script_name}"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )

        execution_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ SUCCÈS: {script_name} terminé en {execution_time:.2f}s")
            print("📝 Sortie:")
            print(result.stdout[-500:])  # Derniers 500 caractères
        else:
            print(f"❌ ÉCHEC: {script_name} a échoué (code: {result.returncode})")
            print("📝 Erreur:")
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ ERREUR: Impossible d'exécuter {script_name}")
        print(f"   Détails: {str(e)}")
        return False

    return True

def check_dependencies():
    """
    Vérifie que toutes les dépendances sont installées
    """
    print("🔍 VÉRIFICATION DES DÉPENDANCES...")

    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'torch', 'transformers', 'imbalanced-learn'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MANQUANT")

    if missing_packages:
        print(f"\n⚠️  PACKAGES MANQUANTS: {', '.join(missing_packages)}")
        print("Installez-les avec: pip install " + ' '.join(missing_packages))
        return False

    print("✅ Toutes les dépendances sont installées")
    return True

def create_directories():
    """
    Crée les dossiers nécessaires s'ils n'existent pas
    """
    print("📁 CRÉATION DES DOSSIERS...")

    directories = ['./data', './models', './results', './logs']

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Créé: {directory}")
        else:
            print(f"✅ Existe: {directory}")

def main():
    """
    Fonction principale - exécute tout le pipeline
    """
    print("=" * 80)
    print("🤖 CYBERMENTOR NLP - PIPELINE COMPLET")
    print("=" * 80)
    print(f"📅 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Répertoire: {os.getcwd()}")
    print("=" * 80)

    # Vérifications préalables
    if not check_dependencies():
        print("❌ Arrêt du pipeline - dépendances manquantes")
        sys.exit(1)

    create_directories()

    # Pipeline des scripts à exécuter
    pipeline_steps = [
        {
            'script': '1_combine_datasets.py',
            'description': 'Étape 1: Combinaison des datasets UNSW-NB15'
        },
        {
            'script': '2_clean_data.py',
            'description': 'Étape 2: Nettoyage et préparation des données'
        },
        {
            'script': '3_balance_data.py',
            'description': 'Étape 3: Équilibrage des classes'
        },
        {
            'script': '5_nlp_features_advanced.py',
            'description': 'Étape 4: Préparation des features NLP avancées'
        },
        {
            'script': '6_train_model.py',
            'description': 'Étape 5: Entraînement du modèle DistilBERT'
        },
        {
            'script': '7_evaluate_model.py',
            'description': 'Étape 6: Évaluation du modèle'
        }
    ]

    # Statistiques
    total_steps = len(pipeline_steps)
    successful_steps = 0
    failed_steps = []

    # Exécution du pipeline
    for i, step in enumerate(pipeline_steps, 1):
        print(f"\n🎯 ÉTAPE {i}/{total_steps}")

        if run_script(step['script'], step['description']):
            successful_steps += 1
        else:
            failed_steps.append(step['script'])
            print(f"⚠️  Échec de l'étape {i}, continuation du pipeline...")

    # Rapport final
    print(f"\n{'='*80}")
    print("📊 RAPPORT FINAL DU PIPELINE")
    print(f"{'='*80}")

    end_time = datetime.now()
    print(f"🏁 Fin: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n📈 STATISTIQUES:")
    print(f"   Étapes totales: {total_steps}")
    print(f"   Réussies: {successful_steps}")
    print(f"   Échouées: {len(failed_steps)}")

    if failed_steps:
        print(f"   Scripts échoués: {', '.join(failed_steps)}")

    success_rate = (successful_steps / total_steps) * 100
    print(f"   Taux de succès: {success_rate:.1f}%")

    if success_rate == 100:
        print(f"\n🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
        print("🤖 CyberMentor NLP est prêt pour la production")
        print("📁 Résultats disponibles dans ./results/")
        print("🧠 Modèles entraînés dans ./models/")
    else:
        print(f"\n⚠️  PIPELINE PARTIELLEMENT RÉUSSI")
        print("Vérifiez les erreurs ci-dessus et relancez les étapes échouées")

    print(f"{'='*80}")

if __name__ == "__main__":
    main()
