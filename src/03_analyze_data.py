import pandas as pd
from pathlib import Path


DATA_PATH = Path("data/processed/circulations_clean.csv")
REPORT_PATH = Path("reports/kpi_report.txt")


def charger_donnees(path):
    """
    Charge les données nettoyées.
    """
    df = pd.read_csv(path)
    return df


def calculer_kpi_globaux(df):
    """
    Calcule les indicateurs globaux du projet.
    """
    nombre_total_trains = len(df)
    nombre_trains_retard = df["train_en_retard"].sum()
    nombre_trains_heure = nombre_total_trains - nombre_trains_retard

    taux_ponctualite = (nombre_trains_heure / nombre_total_trains) * 100
    taux_retard = (nombre_trains_retard / nombre_total_trains) * 100

    retard_moyen_depart = df["retard_depart_minutes"].mean()
    retard_moyen_arrivee = df["retard_arrivee_minutes"].mean()
    retard_max_arrivee = df["retard_arrivee_minutes"].max()

    kpi = {
        "nombre_total_trains": nombre_total_trains,
        "nombre_trains_retard": int(nombre_trains_retard),
        "nombre_trains_heure": int(nombre_trains_heure),
        "taux_ponctualite": round(taux_ponctualite, 2),
        "taux_retard": round(taux_retard, 2),
        "retard_moyen_depart": round(retard_moyen_depart, 2),
        "retard_moyen_arrivee": round(retard_moyen_arrivee, 2),
        "retard_max_arrivee": int(retard_max_arrivee)
    }

    return kpi


def analyser_par_ligne(df):
    """
    Analyse les retards par ligne ferroviaire.
    """
    analyse = (
        df.groupby("ligne")
        .agg(
            nombre_trains=("train_id", "count"),
            retard_moyen_arrivee=("retard_arrivee_minutes", "mean"),
            retard_max_arrivee=("retard_arrivee_minutes", "max"),
            nombre_retards=("train_en_retard", "sum")
        )
        .reset_index()
    )

    analyse["taux_retard"] = (
        analyse["nombre_retards"] / analyse["nombre_trains"] * 100
    ).round(2)

    analyse["retard_moyen_arrivee"] = analyse["retard_moyen_arrivee"].round(2)

    analyse = analyse.sort_values(
        by="retard_moyen_arrivee",
        ascending=False
    )

    return analyse


def analyser_par_gare_depart(df):
    """
    Analyse les retards selon la gare de départ.
    """
    analyse = (
        df.groupby("gare_depart")
        .agg(
            nombre_trains=("train_id", "count"),
            retard_moyen_arrivee=("retard_arrivee_minutes", "mean"),
            nombre_retards=("train_en_retard", "sum")
        )
        .reset_index()
    )

    analyse["taux_retard"] = (
        analyse["nombre_retards"] / analyse["nombre_trains"] * 100
    ).round(2)

    analyse["retard_moyen_arrivee"] = analyse["retard_moyen_arrivee"].round(2)

    analyse = analyse.sort_values(
        by="retard_moyen_arrivee",
        ascending=False
    )

    return analyse


def analyser_causes(df):
    """
    Analyse les causes principales des retards.
    """
    analyse = (
        df.groupby("cause_retard")
        .agg(
            nombre_trains=("train_id", "count"),
            retard_moyen_arrivee=("retard_arrivee_minutes", "mean")
        )
        .reset_index()
    )

    analyse["retard_moyen_arrivee"] = analyse["retard_moyen_arrivee"].round(2)

    analyse = analyse.sort_values(
        by="nombre_trains",
        ascending=False
    )

    return analyse


def analyser_par_jour(df):
    """
    Analyse les retards par jour de semaine.
    """
    analyse = (
        df.groupby("jour_semaine")
        .agg(
            nombre_trains=("train_id", "count"),
            retard_moyen_arrivee=("retard_arrivee_minutes", "mean"),
            nombre_retards=("train_en_retard", "sum")
        )
        .reset_index()
    )

    analyse["retard_moyen_arrivee"] = analyse["retard_moyen_arrivee"].round(2)

    return analyse


def analyser_par_heure(df):
    """
    Analyse les retards selon l'heure prévue de départ.
    """
    analyse = (
        df.groupby("heure_depart")
        .agg(
            nombre_trains=("train_id", "count"),
            retard_moyen_arrivee=("retard_arrivee_minutes", "mean"),
            nombre_retards=("train_en_retard", "sum")
        )
        .reset_index()
    )

    analyse["retard_moyen_arrivee"] = analyse["retard_moyen_arrivee"].round(2)

    analyse = analyse.sort_values(by="heure_depart")

    return analyse


def sauvegarder_rapport(kpi, par_ligne, par_gare, par_cause, par_jour, par_heure):
    """
    Sauvegarde un rapport texte dans le dossier reports.
    """
    with open(REPORT_PATH, "w", encoding="utf-8") as fichier:
        fichier.write("===== RAPPORT KPI - RETARDS FERROVIAIRES =====\n\n")

        fichier.write("===== KPI GLOBAUX =====\n")
        fichier.write(f"Nombre total de trains : {kpi['nombre_total_trains']}\n")
        fichier.write(f"Nombre de trains en retard : {kpi['nombre_trains_retard']}\n")
        fichier.write(f"Nombre de trains à l'heure : {kpi['nombre_trains_heure']}\n")
        fichier.write(f"Taux de ponctualité : {kpi['taux_ponctualite']} %\n")
        fichier.write(f"Taux de retard : {kpi['taux_retard']} %\n")
        fichier.write(f"Retard moyen au départ : {kpi['retard_moyen_depart']} minutes\n")
        fichier.write(f"Retard moyen à l'arrivée : {kpi['retard_moyen_arrivee']} minutes\n")
        fichier.write(f"Retard maximum à l'arrivée : {kpi['retard_max_arrivee']} minutes\n")

        fichier.write("\n===== ANALYSE PAR LIGNE =====\n")
        fichier.write(par_ligne.to_string(index=False))

        fichier.write("\n\n===== ANALYSE PAR GARE DE DÉPART =====\n")
        fichier.write(par_gare.to_string(index=False))

        fichier.write("\n\n===== ANALYSE PAR CAUSE =====\n")
        fichier.write(par_cause.to_string(index=False))

        fichier.write("\n\n===== ANALYSE PAR JOUR =====\n")
        fichier.write(par_jour.to_string(index=False))

        fichier.write("\n\n===== ANALYSE PAR HEURE =====\n")
        fichier.write(par_heure.to_string(index=False))


def sauvegarder_tables(par_ligne, par_gare, par_cause, par_jour, par_heure):
    """
    Sauvegarde les analyses sous forme de fichiers CSV.
    """
    par_ligne.to_csv("data/processed/analyse_par_ligne.csv", index=False, encoding="utf-8-sig")
    par_gare.to_csv("data/processed/analyse_par_gare_depart.csv", index=False, encoding="utf-8-sig")
    par_cause.to_csv("data/processed/analyse_par_cause.csv", index=False, encoding="utf-8-sig")
    par_jour.to_csv("data/processed/analyse_par_jour.csv", index=False, encoding="utf-8-sig")
    par_heure.to_csv("data/processed/analyse_par_heure.csv", index=False, encoding="utf-8-sig")


def main():
    print("===== CHARGEMENT DES DONNÉES NETTOYÉES =====")
    df = charger_donnees(DATA_PATH)

    print("Dimensions :", df.shape)

    print("\n===== CALCUL DES KPI =====")
    kpi = calculer_kpi_globaux(df)

    for cle, valeur in kpi.items():
        print(f"{cle} : {valeur}")

    print("\n===== ANALYSE PAR LIGNE =====")
    par_ligne = analyser_par_ligne(df)
    print(par_ligne)

    print("\n===== ANALYSE PAR GARE DE DÉPART =====")
    par_gare = analyser_par_gare_depart(df)
    print(par_gare)

    print("\n===== ANALYSE PAR CAUSE =====")
    par_cause = analyser_causes(df)
    print(par_cause)

    print("\n===== ANALYSE PAR JOUR =====")
    par_jour = analyser_par_jour(df)
    print(par_jour)

    print("\n===== ANALYSE PAR HEURE =====")
    par_heure = analyser_par_heure(df)
    print(par_heure)

    sauvegarder_rapport(kpi, par_ligne, par_gare, par_cause, par_jour, par_heure)
    sauvegarder_tables(par_ligne, par_gare, par_cause, par_jour, par_heure)

    print("\n===== RAPPORT GÉNÉRÉ =====")
    print(f"Rapport sauvegardé dans : {REPORT_PATH}")


if __name__ == "__main__":
    main()