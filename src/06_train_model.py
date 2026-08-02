import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================
# CHEMINS DU PROJET
# =========================

DATA_PATH = Path("data/processed/circulations_clean.csv")

MODEL_PATH = Path("models/model_retard_train.pkl")

REPORT_PATH = Path("reports/ml_report.txt")
COMPARISON_PATH = Path("reports/model_comparison.csv")
FEATURE_IMPORTANCE_PATH = Path("reports/feature_importance.csv")

FIGURES_PATH = Path("reports/figures")
CONFUSION_MATRIX_PATH = FIGURES_PATH / "confusion_matrix_best_model.png"
FEATURE_IMPORTANCE_FIGURE_PATH = FIGURES_PATH / "feature_importance_best_model.png"


# Création automatique des dossiers si nécessaire
Path("models").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)


# =========================
# CHARGEMENT DES DONNÉES
# =========================

def charger_donnees(path):
    """
    Charge le fichier CSV nettoyé.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}. "
            "Lance d'abord : py src/02_clean_data.py"
        )

    df = pd.read_csv(path)
    return df


# =========================
# PRÉPARATION DES DONNÉES
# =========================

def preparer_donnees(df):
    """
    Prépare les variables explicatives X et la variable cible y.

    Important :
    On n'utilise pas cause_retard comme variable explicative,
    car dans un vrai système prédictif, la cause du retard peut ne pas être connue
    avant ou au moment du départ.
    """

    features = [
        "type_train",
        "ligne",
        "gare_depart",
        "gare_arrivee",
        "jour_semaine",
        "heure_depart",
        "retard_depart_minutes"
    ]

    target = "train_en_retard"

    colonnes_manquantes = []

    for colonne in features + [target]:
        if colonne not in df.columns:
            colonnes_manquantes.append(colonne)

    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes : {colonnes_manquantes}")

    X = df[features]
    y = df[target]

    return X, y, features


# =========================
# PRÉPROCESSEMENT
# =========================

def creer_preprocesseur():
    """
    Crée le préprocesseur :
    - OneHotEncoder pour les variables catégorielles
    - passthrough pour les variables numériques
    """

    variables_categorielles = [
        "type_train",
        "ligne",
        "gare_depart",
        "gare_arrivee",
        "jour_semaine"
    ]

    variables_numeriques = [
        "heure_depart",
        "retard_depart_minutes"
    ]

    preprocesseur = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                variables_categorielles
            ),
            (
                "num",
                "passthrough",
                variables_numeriques
            )
        ]
    )

    return preprocesseur


# =========================
# MODÈLES À COMPARER
# =========================

def creer_modeles():
    """
    Crée plusieurs modèles de classification à comparer.
    """

    modeles = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            class_weight="balanced"
        )
    }

    return modeles


# =========================
# ENTRAÎNEMENT ET COMPARAISON
# =========================

def entrainer_et_comparer_modeles(X_train, X_test, y_train, y_test):
    """
    Entraîne plusieurs modèles, calcule les métriques,
    puis sélectionne le meilleur selon le F1-score.
    """

    modeles = creer_modeles()

    resultats = []
    pipelines = {}

    for nom_modele, modele in modeles.items():
        print(f"\n===== ENTRAÎNEMENT : {nom_modele} =====")

        pipeline = Pipeline(
            steps=[
                ("preprocessing", creer_preprocesseur()),
                ("model", modele)
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        resultats.append({
            "modele": nom_modele,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        })

        pipelines[nom_modele] = pipeline

        print("Accuracy :", round(accuracy, 4))
        print("Precision :", round(precision, 4))
        print("Recall :", round(recall, 4))
        print("F1-score :", round(f1, 4))

    df_resultats = pd.DataFrame(resultats)
    df_resultats = df_resultats.sort_values(
        by="f1_score",
        ascending=False
    )

    meilleur_nom = df_resultats.iloc[0]["modele"]
    meilleur_modele = pipelines[meilleur_nom]

    return df_resultats, meilleur_nom, meilleur_modele


# =========================
# MATRICE DE CONFUSION
# =========================

def sauvegarder_matrice_confusion(modele, X_test, y_test, nom_modele):
    """
    Sauvegarde la matrice de confusion du meilleur modèle sous forme d'image.
    """

    y_pred = modele.predict(X_test)

    matrice = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrice,
        display_labels=["À l'heure", "En retard"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax)
    ax.set_title(f"Matrice de confusion - {nom_modele}")

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    return matrice


# =========================
# IMPORTANCE DES VARIABLES
# =========================

def sauvegarder_importance_variables(modele):
    """
    Sauvegarde l'importance des variables.

    Cas 1 : Random Forest ou Decision Tree
    → utilisation de feature_importances_

    Cas 2 : Logistic Regression
    → utilisation de la valeur absolue des coefficients
    """

    preprocesseur = modele.named_steps["preprocessing"]
    modele_interne = modele.named_steps["model"]

    noms_variables = preprocesseur.get_feature_names_out()

    if hasattr(modele_interne, "feature_importances_"):
        importances = modele_interne.feature_importances_

    elif hasattr(modele_interne, "coef_"):
        importances = abs(modele_interne.coef_[0])

    else:
        print("Ce modèle ne permet pas d'extraire l'importance des variables.")
        return

    df_importances = pd.DataFrame({
        "variable": noms_variables,
        "importance": importances
    })

    df_importances = df_importances.sort_values(
        by="importance",
        ascending=False
    )

    df_importances.to_csv(
        FEATURE_IMPORTANCE_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    top_importances = df_importances.head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_importances["variable"],
        top_importances["importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Variable")
    plt.title("Top 15 des variables les plus importantes")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(FEATURE_IMPORTANCE_FIGURE_PATH, dpi=300)
    plt.close()


# =========================
# RAPPORT MACHINE LEARNING
# =========================

def sauvegarder_rapport(
    df_resultats,
    meilleur_nom,
    meilleur_modele,
    X_test,
    y_test,
    features,
    nb_lignes,
    matrice
):
    """
    Sauvegarde un rapport texte complet sur la phase Machine Learning.
    """

    y_pred = meilleur_modele.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    rapport_classification = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as fichier:
        fichier.write("===== RAPPORT MACHINE LEARNING =====\n\n")

        fichier.write("1. Objectif du modèle\n")
        fichier.write(
            "L'objectif est de prédire si un train sera en retard ou à l'heure "
            "à partir des caractéristiques disponibles avant ou au moment du départ.\n\n"
        )

        fichier.write("2. Variable cible\n")
        fichier.write("Variable cible : train_en_retard\n")
        fichier.write("0 = train à l'heure\n")
        fichier.write("1 = train en retard\n\n")

        fichier.write("3. Variables explicatives utilisées\n")
        for feature in features:
            fichier.write(f"- {feature}\n")

        fichier.write(f"\nNombre total de lignes utilisées : {nb_lignes}\n\n")

        fichier.write("4. Comparaison des modèles\n")
        fichier.write(df_resultats.to_string(index=False))

        fichier.write("\n\n5. Meilleur modèle sélectionné\n")
        fichier.write(f"Modèle sélectionné : {meilleur_nom}\n")
        fichier.write(f"Accuracy : {round(accuracy, 4)}\n")
        fichier.write(f"Precision : {round(precision, 4)}\n")
        fichier.write(f"Recall : {round(recall, 4)}\n")
        fichier.write(f"F1-score : {round(f1, 4)}\n\n")

        fichier.write("6. Rapport de classification\n")
        fichier.write(rapport_classification)

        fichier.write("\n7. Matrice de confusion\n")
        fichier.write(str(matrice))

        fichier.write("\n\n8. Remarque méthodologique\n")
        fichier.write(
            "Les résultats dépendent fortement de la qualité, du volume et de la représentativité "
            "des données utilisées. Lorsque les données sont simulées, l'objectif principal est de "
            "valider la chaîne complète de modélisation et non de produire une performance opérationnelle définitive.\n"
        )


# =========================
# PROGRAMME PRINCIPAL
# =========================

def main():
    print("===== CHARGEMENT DES DONNÉES =====")

    df = charger_donnees(DATA_PATH)

    print("Dimensions du dataset :", df.shape)

    if len(df) < 100:
        print(
            "Attention : le dataset contient peu de lignes. "
            "Les résultats du modèle peuvent être instables."
        )

    print("\n===== PRÉPARATION DES DONNÉES =====")

    X, y, features = preparer_donnees(df)

    print("Variables explicatives :", features)
    print("Variable cible : train_en_retard")

    print("\nRépartition de la variable cible :")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError(
            "La variable cible contient une seule classe. "
            "Il faut avoir des trains à l'heure et des trains en retard."
        )

    stratify_option = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify_option
    )

    print("\nTaille X_train :", X_train.shape)
    print("Taille X_test :", X_test.shape)

    print("\n===== COMPARAISON DES MODÈLES =====")

    df_resultats, meilleur_nom, meilleur_modele = entrainer_et_comparer_modeles(
        X_train,
        X_test,
        y_train,
        y_test
    )

    print("\n===== TABLEAU DE COMPARAISON =====")
    print(df_resultats)

    print("\nMeilleur modèle sélectionné :", meilleur_nom)

    print("\n===== SAUVEGARDE DES RÉSULTATS =====")

    # Sauvegarde du tableau de comparaison
    df_resultats.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # Sauvegarde du meilleur modèle
    joblib.dump(meilleur_modele, MODEL_PATH)

    # Sauvegarde matrice de confusion
    matrice = sauvegarder_matrice_confusion(
        meilleur_modele,
        X_test,
        y_test,
        meilleur_nom
    )

    # Sauvegarde importance des variables
    sauvegarder_importance_variables(meilleur_modele)

    # Sauvegarde rapport ML
    sauvegarder_rapport(
        df_resultats=df_resultats,
        meilleur_nom=meilleur_nom,
        meilleur_modele=meilleur_modele,
        X_test=X_test,
        y_test=y_test,
        features=features,
        nb_lignes=len(df),
        matrice=matrice
    )

    print(f"Modèle sauvegardé dans : {MODEL_PATH}")
    print(f"Rapport ML sauvegardé dans : {REPORT_PATH}")
    print(f"Comparaison des modèles sauvegardée dans : {COMPARISON_PATH}")
    print(f"Importance des variables sauvegardée dans : {FEATURE_IMPORTANCE_PATH}")
    print(f"Matrice de confusion sauvegardée dans : {CONFUSION_MATRIX_PATH}")
    print(f"Graphique importance variables sauvegardé dans : {FEATURE_IMPORTANCE_FIGURE_PATH}")

    print("\n===== TERMINÉ =====")


if __name__ == "__main__":
    main()