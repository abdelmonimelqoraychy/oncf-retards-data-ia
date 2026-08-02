import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DATA_PATH = Path("data/processed/circulations_clean.csv")
FIGURES_PATH = Path("reports/figures")

FIGURES_PATH.mkdir(parents=True, exist_ok=True)


def charger_donnees(path):
    df = pd.read_csv(path)
    df["date_trajet"] = pd.to_datetime(df["date_trajet"], errors="coerce")
    return df


def graphique_retard_moyen_par_ligne(df):
    analyse = (
        df.groupby("ligne")["retard_arrivee_minutes"]
        .mean()
        .reset_index()
        .sort_values(by="retard_arrivee_minutes", ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=analyse,
        x="retard_arrivee_minutes",
        y="ligne"
    )

    plt.title("Retard moyen à l'arrivée par ligne")
    plt.xlabel("Retard moyen à l'arrivée en minutes")
    plt.ylabel("Ligne")
    plt.tight_layout()

    plt.savefig(FIGURES_PATH / "retard_moyen_par_ligne.png", dpi=300)
    plt.close()


def graphique_retards_par_cause(df):
    analyse = (
        df["cause_retard"]
        .value_counts()
        .reset_index()
    )

    analyse.columns = ["cause_retard", "nombre_trains"]

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=analyse,
        x="nombre_trains",
        y="cause_retard"
    )

    plt.title("Nombre de trains par cause de retard")
    plt.xlabel("Nombre de trains")
    plt.ylabel("Cause")
    plt.tight_layout()

    plt.savefig(FIGURES_PATH / "retards_par_cause.png", dpi=300)
    plt.close()


def graphique_retard_moyen_par_heure(df):
    analyse = (
        df.groupby("heure_depart")["retard_arrivee_minutes"]
        .mean()
        .reset_index()
        .sort_values(by="heure_depart")
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=analyse,
        x="heure_depart",
        y="retard_arrivee_minutes",
        marker="o"
    )

    plt.title("Retard moyen à l'arrivée par heure de départ")
    plt.xlabel("Heure de départ prévue")
    plt.ylabel("Retard moyen à l'arrivée en minutes")
    plt.tight_layout()

    plt.savefig(FIGURES_PATH / "retard_moyen_par_heure.png", dpi=300)
    plt.close()


def graphique_taux_retard_par_ligne(df):
    analyse = (
        df.groupby("ligne")
        .agg(
            nombre_trains=("train_id", "count"),
            nombre_retards=("train_en_retard", "sum")
        )
        .reset_index()
    )

    analyse["taux_retard"] = (
        analyse["nombre_retards"] / analyse["nombre_trains"] * 100
    ).round(2)

    analyse = analyse.sort_values(by="taux_retard", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=analyse,
        x="taux_retard",
        y="ligne"
    )

    plt.title("Taux de retard par ligne")
    plt.xlabel("Taux de retard (%)")
    plt.ylabel("Ligne")
    plt.tight_layout()

    plt.savefig(FIGURES_PATH / "taux_retard_par_ligne.png", dpi=300)
    plt.close()


def graphique_evolution_retard_par_date(df):
    analyse = (
        df.groupby("date_trajet")["retard_arrivee_minutes"]
        .mean()
        .reset_index()
        .sort_values(by="date_trajet")
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=analyse,
        x="date_trajet",
        y="retard_arrivee_minutes",
        marker="o"
    )

    plt.title("Évolution du retard moyen à l'arrivée par date")
    plt.xlabel("Date du trajet")
    plt.ylabel("Retard moyen à l'arrivée en minutes")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(FIGURES_PATH / "evolution_retard_par_date.png", dpi=300)
    plt.close()


def main():
    print("===== CHARGEMENT DES DONNÉES =====")
    df = charger_donnees(DATA_PATH)

    print("===== GÉNÉRATION DES GRAPHIQUES =====")

    graphique_retard_moyen_par_ligne(df)
    print("Graphique créé : retard_moyen_par_ligne.png")

    graphique_retards_par_cause(df)
    print("Graphique créé : retards_par_cause.png")

    graphique_retard_moyen_par_heure(df)
    print("Graphique créé : retard_moyen_par_heure.png")

    graphique_taux_retard_par_ligne(df)
    print("Graphique créé : taux_retard_par_ligne.png")

    graphique_evolution_retard_par_date(df)
    print("Graphique créé : evolution_retard_par_date.png")

    print("\n===== TERMINÉ =====")
    print(f"Les figures sont sauvegardées dans : {FIGURES_PATH}")


if __name__ == "__main__":
    main()