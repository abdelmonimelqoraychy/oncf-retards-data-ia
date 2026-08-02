import random
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


OUTPUT_PATH = Path("data/raw/circulations_oncf_sample.csv")


def ajouter_minutes(heure_str, minutes):
    """
    Ajoute des minutes à une heure au format HH:MM.
    """
    heure = datetime.strptime(heure_str, "%H:%M")
    nouvelle_heure = heure + timedelta(minutes=minutes)
    return nouvelle_heure.strftime("%H:%M")


def generer_retard(cause):
    """
    Génère un retard selon la cause.
    Les valeurs sont simulées pour rendre le dataset plus réaliste.
    """
    if cause == "Aucune":
        return random.choice([0, 0, 0, 2, 3, 5])

    if cause == "Trafic":
        return random.randint(5, 35)

    if cause == "Technique":
        return random.randint(15, 90)

    if cause == "Exploitation":
        return random.randint(10, 60)

    if cause == "Météo":
        return random.randint(10, 75)

    if cause == "Incident voyageur":
        return random.randint(20, 100)

    return random.randint(0, 20)


def generer_dataset(nombre_lignes=1000):
    random.seed(42)

    types_train = ["Al Atlas", "Al Boraq", "Train Navette Rapide"]

    lignes = [
        ("Fes-Rabat", "Fes", "Rabat"),
        ("Rabat-Fes", "Rabat", "Fes"),
        ("Casa-Fes", "Casa", "Fes"),
        ("Fes-Casa", "Fes", "Casa"),
        ("Casa-Marrakech", "Casa", "Marrakech"),
        ("Marrakech-Casa", "Marrakech", "Casa"),
        ("Tanger-Casa", "Tanger", "Casa"),
        ("Casa-Tanger", "Casa", "Tanger"),
        ("Rabat-Casa", "Rabat", "Casa"),
        ("Casa-Rabat", "Casa", "Rabat")
    ]

    causes = [
        "Aucune",
        "Trafic",
        "Technique",
        "Exploitation",
        "Météo",
        "Incident voyageur"
    ]

    probabilites_causes = [
        0.45,
        0.20,
        0.12,
        0.12,
        0.07,
        0.04
    ]

    heures_depart_possibles = [
        "06:00", "06:30", "07:00", "07:30", "08:00", "08:30",
        "09:00", "10:00", "11:00", "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00", "18:00", "19:00", "20:00"
    ]

    date_debut = datetime(2026, 1, 1)

    donnees = []

    for i in range(1, nombre_lignes + 1):
        date_trajet = date_debut + timedelta(days=random.randint(0, 180))

        type_train = random.choice(types_train)
        ligne, gare_depart, gare_arrivee = random.choice(lignes)

        heure_depart_prevue = random.choice(heures_depart_possibles)

        duree_trajet = random.randint(60, 260)
        heure_arrivee_prevue = ajouter_minutes(heure_depart_prevue, duree_trajet)

        cause_retard = random.choices(causes, weights=probabilites_causes, k=1)[0]

        retard_depart = generer_retard(cause_retard)

        retard_arrivee = retard_depart + random.randint(-5, 20)
        retard_arrivee = max(0, retard_arrivee)

        heure_depart_reelle = ajouter_minutes(heure_depart_prevue, retard_depart)
        heure_arrivee_reelle = ajouter_minutes(heure_arrivee_prevue, retard_arrivee)

        train_id = f"T{i:04d}"

        donnees.append({
            "date_trajet": date_trajet.strftime("%Y-%m-%d"),
            "train_id": train_id,
            "type_train": type_train,
            "ligne": ligne,
            "gare_depart": gare_depart,
            "gare_arrivee": gare_arrivee,
            "heure_depart_prevue": heure_depart_prevue,
            "heure_depart_reelle": heure_depart_reelle,
            "heure_arrivee_prevue": heure_arrivee_prevue,
            "heure_arrivee_reelle": heure_arrivee_reelle,
            "cause_retard": cause_retard
        })

    df = pd.DataFrame(donnees)
    return df


def main():
    print("===== GÉNÉRATION DU DATASET SIMULÉ =====")

    df = generer_dataset(nombre_lignes=1000)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Dataset généré : {OUTPUT_PATH}")
    print("Dimensions :", df.shape)

    print("\nAperçu :")
    print(df.head())


if __name__ == "__main__":
    main()