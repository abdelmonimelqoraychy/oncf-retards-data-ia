import subprocess
import sys
from pathlib import Path


# =========================
# CONFIGURATION
# =========================

EXECUTER_POSTGRES = True
# Mets True si PostgreSQL fonctionne correctement chez toi.


SCRIPTS = [
    "src/00_generate_sample_data.py",
    "src/01_read_data.py",
    "src/02_clean_data.py",
    "src/03_analyze_data.py",
    "src/04_visualize_data.py",
    "src/06_train_model.py"
]


SCRIPT_POSTGRES = "src/05_load_to_postgres.py"


def executer_script(script_path):
    """
    Exécute un script Python et arrête le pipeline en cas d'erreur.
    """

    script = Path(script_path)

    if not script.exists():
        raise FileNotFoundError(f"Script introuvable : {script_path}")

    print("\n" + "=" * 70)
    print(f"EXÉCUTION : {script_path}")
    print("=" * 70)

    resultat = subprocess.run(
        [sys.executable, script_path],
        text=True
    )

    if resultat.returncode != 0:
        raise RuntimeError(f"Erreur pendant l'exécution de : {script_path}")

    print(f"TERMINÉ : {script_path}")


def verifier_dossiers():
    """
    Vérifie et crée les dossiers nécessaires au projet.
    """

    dossiers = [
        "data/raw",
        "data/processed",
        "reports",
        "reports/figures",
        "models",
        "sql",
        "dashboard"
    ]

    for dossier in dossiers:
        Path(dossier).mkdir(parents=True, exist_ok=True)


def main():
    print("===== PIPELINE PROJET ONCF DATA & IA =====")

    verifier_dossiers()

    for script in SCRIPTS:
        executer_script(script)

    if EXECUTER_POSTGRES:
        executer_script(SCRIPT_POSTGRES)
    else:
        print("\nPostgreSQL ignoré.")
        print("Pour l'activer, mets EXECUTER_POSTGRES = True dans ce fichier.")

    print("\n" + "=" * 70)
    print("PIPELINE TERMINÉ AVEC SUCCÈS")
    print("=" * 70)

    print("\nFichiers générés :")
    print("- data/raw/circulations_oncf_sample.csv")
    print("- data/processed/circulations_clean.csv")
    print("- reports/kpi_report.txt")
    print("- reports/ml_report.txt")
    print("- reports/model_comparison.csv")
    print("- reports/feature_importance.csv")
    print("- models/model_retard_train.pkl")
    print("- reports/figures/*.png")

    print("\nPour lancer le dashboard :")
    print("py -m streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()