# Diagrammes du projet ONCF — Analyse et prédiction des retards ferroviaires

## 1. Diagramme d’architecture générale

```mermaid
flowchart LR
    A[Données brutes CSV/Excel] --> B[Nettoyage Python pandas]
    B --> C[Données nettoyées]
    C --> D[Analyse KPI]
    C --> E[Base PostgreSQL]
    C --> F[Modèle Machine Learning]
    D --> G[Dashboard Streamlit]
    E --> G
    F --> G
    G --> H[Utilisateur / Encadrant]
```

## Description

Ce diagramme représente l’architecture globale de la plateforme. Les données brutes sont importées, nettoyées avec Python et pandas, puis utilisées pour l’analyse statistique, le stockage dans PostgreSQL, l’entraînement du modèle Machine Learning et l’alimentation du dashboard Streamlit.

---

## 2. Diagramme du pipeline Data

```mermaid
flowchart TD
    A[Début] --> B[Génération ou importation des données]
    B --> C[Lecture du fichier CSV]
    C --> D[Vérification des colonnes]
    D --> E[Nettoyage des valeurs textuelles]
    E --> F[Conversion des dates et heures]
    F --> G[Calcul des retards]
    G --> H[Création de la variable cible]
    H --> I[Sauvegarde des données nettoyées]
    I --> J[Calcul des KPI]
    J --> K[Génération des graphiques]
    K --> L[Entraînement du modèle IA]
    L --> M[Sauvegarde du modèle]
    M --> N[Dashboard Streamlit]
    N --> O[Fin]
```

## Description

Ce pipeline décrit les différentes étapes de traitement des données. Il commence par l’importation ou la génération des données, puis passe par le nettoyage, le calcul des retards, la création des indicateurs, la génération des visualisations et l’entraînement du modèle prédictif.

---

## 3. Diagramme de cas d’utilisation

```mermaid
flowchart LR
    U[Utilisateur / Encadrant]

    U --> A[Consulter les KPI]
    U --> B[Filtrer les données]
    U --> C[Visualiser les retards]
    U --> D[Exporter les données filtrées]
    U --> E[Faire une prédiction IA]
    U --> F[Consulter la performance du modèle]
    U --> G[Analyser les causes de retard]

    A --> S[Dashboard Streamlit]
    B --> S
    C --> S
    D --> S
    E --> S
    F --> S
    G --> S
```

## Description

Ce diagramme présente les principales fonctionnalités offertes à l’utilisateur. Le dashboard permet de consulter les indicateurs, appliquer des filtres, analyser les retards, exporter les données et utiliser le modèle de prédiction IA.

---

## 4. Diagramme du modèle de données simplifié

```mermaid
erDiagram
    CIRCULATIONS {
        int id
        date date_trajet
        string train_id
        string type_train
        string ligne
        string gare_depart
        string gare_arrivee
        time heure_depart_prevue
        time heure_depart_reelle
        time heure_arrivee_prevue
        time heure_arrivee_reelle
        string cause_retard
        int retard_depart_minutes
        int retard_arrivee_minutes
        int train_en_retard
        string statut
        string jour_semaine
        int heure_depart
    }
```

## Description

Ce modèle de données simplifié représente la table principale utilisée dans le projet. La table `circulations` contient les informations relatives aux trajets ferroviaires, aux horaires prévus et réels, aux retards calculés, au statut du train et aux variables utilisées pour l’analyse et le Machine Learning.

---

# Utilisation dans le rapport

Ces diagrammes peuvent être intégrés dans le chapitre :

```text
Chapitre 2 — Analyse et conception
```

ou dans :

```text
Chapitre 3 — Réalisation technique
```