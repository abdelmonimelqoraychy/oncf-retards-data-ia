import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("data/raw/circulations_oncf_sample.csv")
PROCESSED_PATH = Path("data/processed/circulations_clean.csv")


def nettoyer_texte(valeur):
    """
    Nettoie une valeur textuelle :
    - supprime les espaces
    - met la première lettre de chaque mot en majuscule
    """
    if pd.isna(valeur):
        return "Non renseigné"
    return str(valeur).strip().title()


def charger_donnees(path):
    """
    Charge les données brutes depuis un fichier CSV.
    """
    df = pd.read_csv(path)
    return df


def verifier_colonnes(df):
    """
    Vérifie que les colonnes nécessaires existent.
    """
    colonnes_obligatoires = [
        "date_trajet",
        "train_id",
        "type_train",
        "ligne",
        "gare_depart",
        "gare_arrivee",
        "heure_depart_prevue",
        "heure_depart_reelle",
        "heure_arrivee_prevue",
        "heure_arrivee_reelle",
        "cause_retard"
    ]

    colonnes_manquantes = []

    for colonne in colonnes_obligatoires:
        if colonne not in df.columns:
            colonnes_manquantes.append(colonne)

    if len(colonnes_manquantes) > 0:
        raise ValueError(f"Colonnes manquantes : {colonnes_manquantes}")


def creer_datetime(df, colonne_heure):
    """
    Combine date_trajet + heure pour créer une colonne datetime.
    """
    return pd.to_datetime(
        df["date_trajet"].astype(str) + " " + df[colonne_heure].astype(str),
        errors="coerce"
    )


def calculer_retard_minutes(date_prevue, date_reelle):
    """
    Calcule le retard en minutes.
    Si le résultat est négatif, on considère le retard comme 0.
    """
    retard = (date_reelle - date_prevue).dt.total_seconds() / 60
    retard = retard.fillna(0)
    retard = retard.clip(lower=0)
    return retard.astype(int)


def nettoyer_donnees(df):
    """
    Nettoie et transforme les données ONCF.
    """

    # Nettoyage des colonnes texte
    df["train_id"] = df["train_id"].astype(str).str.strip().str.upper()
    df["type_train"] = df["type_train"].apply(nettoyer_texte)
    df["ligne"] = df["ligne"].apply(nettoyer_texte)
    df["gare_depart"] = df["gare_depart"].apply(nettoyer_texte)
    df["gare_arrivee"] = df["gare_arrivee"].apply(nettoyer_texte)
    df["cause_retard"] = df["cause_retard"].apply(nettoyer_texte)

    # Conversion de la date
    df["date_trajet"] = pd.to_datetime(df["date_trajet"], errors="coerce")

    # Création des colonnes datetime
    df["depart_prevu_dt"] = creer_datetime(df, "heure_depart_prevue")
    df["depart_reel_dt"] = creer_datetime(df, "heure_depart_reelle")
    df["arrivee_prevue_dt"] = creer_datetime(df, "heure_arrivee_prevue")
    df["arrivee_reelle_dt"] = creer_datetime(df, "heure_arrivee_reelle")

    # Calcul des retards
    df["retard_depart_minutes"] = calculer_retard_minutes(
        df["depart_prevu_dt"],
        df["depart_reel_dt"]
    )

    df["retard_arrivee_minutes"] = calculer_retard_minutes(
        df["arrivee_prevue_dt"],
        df["arrivee_reelle_dt"]
    )

    # Variable cible pour le futur modèle Machine Learning
    df["train_en_retard"] = np.where(df["retard_arrivee_minutes"] >= 10, 1, 0)

    # Statut lisible pour le dashboard
    df["statut"] = np.where(df["train_en_retard"] == 1, "En retard", "À l'heure")

    # Variables temporelles utiles pour l'analyse
    df["jour_semaine_num"] = df["date_trajet"].dt.dayofweek

    jours = {
        0: "Lundi",
        1: "Mardi",
        2: "Mercredi",
        3: "Jeudi",
        4: "Vendredi",
        5: "Samedi",
        6: "Dimanche"
    }

    df["jour_semaine"] = df["jour_semaine_num"].map(jours)
    df["heure_depart"] = df["depart_prevu_dt"].dt.hour

    # Suppression des doublons
    df = df.drop_duplicates()

    return df


def sauvegarder_donnees(df, path):
    """
    Sauvegarde les données nettoyées.
    """
    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    print("===== CHARGEMENT DES DONNÉES =====")
    df = charger_donnees(RAW_PATH)

    print("Dimensions avant nettoyage :", df.shape)

    verifier_colonnes(df)

    print("\n===== NETTOYAGE EN COURS =====")
    df_clean = nettoyer_donnees(df)

    print("Dimensions après nettoyage :", df_clean.shape)

    sauvegarder_donnees(df_clean, PROCESSED_PATH)

    print("\n===== FICHIER NETTOYÉ CRÉÉ =====")
    print(f"Fichier sauvegardé dans : {PROCESSED_PATH}")

    print("\n===== APERÇU DES DONNÉES PROPRES =====")
    print(df_clean.head())

    print("\n===== COLONNES DISPONIBLES =====")
    print(df_clean.columns.tolist())


if __name__ == "__main__":
    main()