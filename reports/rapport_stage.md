# Rapport de stage  
# Conception d’une application Data & IA pour l’analyse et la prédiction des retards ferroviaires

## Remarque importante

Cette version du projet utilise un dataset simulé afin de valider la chaîne complète de traitement. Les résultats ne représentent pas des statistiques officielles de l’ONCF. Le projet est conçu pour pouvoir intégrer ensuite des données réelles.

---

# Introduction générale

Le secteur ferroviaire joue un rôle important dans la mobilité des personnes et des marchandises. La ponctualité des trains constitue un indicateur essentiel de performance, car elle influence directement la qualité du service, la satisfaction des voyageurs et l’organisation de l’exploitation.

Dans ce contexte, l’analyse des retards ferroviaires représente un enjeu important pour comprendre les causes possibles des perturbations, suivre les indicateurs de performance et anticiper les risques de retard.

Ce projet consiste à concevoir une application Data & IA permettant d’analyser les données de circulation ferroviaire, de calculer des indicateurs de ponctualité, de visualiser les résultats à travers un dashboard interactif et de prédire le risque de retard d’un train à l’aide d’un modèle de Machine Learning.

---

# Chapitre 1 — Contexte général du projet

## 1.1 Contexte du stage

Ce projet a été réalisé dans le cadre d’un stage d’initiation/professionnalisation autour de la Data et de l’Intelligence Artificielle appliquées au domaine ferroviaire.

L’objectif principal est de mettre en place une première version fonctionnelle d’une plateforme permettant d’exploiter les données de circulation afin de produire des analyses décisionnelles et des prédictions.

## 1.2 Problématique

Les retards ferroviaires peuvent dépendre de plusieurs facteurs : ligne concernée, type de train, gare de départ, horaire, jour de circulation, conditions d’exploitation ou incidents.

La problématique centrale du projet est donc la suivante :

**Comment concevoir une application Data & IA permettant d’analyser les retards ferroviaires, de suivre les indicateurs de ponctualité et de prédire le risque de retard d’un train ?**

## 1.3 Objectifs du projet

Les objectifs du projet sont :

- importer les données de circulation ferroviaire ;
- nettoyer et structurer les données ;
- calculer automatiquement les retards au départ et à l’arrivée ;
- créer des indicateurs de performance liés à la ponctualité ;
- analyser les retards par ligne, gare, cause, jour et heure ;
- stocker les données dans PostgreSQL ;
- développer un dashboard interactif avec Streamlit ;
- entraîner un modèle de Machine Learning ;
- intégrer une fonctionnalité de prédiction IA dans le dashboard.

---

# Chapitre 2 — Analyse et conception

## 2.1 Description des données

Le dataset utilisé contient les colonnes suivantes :

| Colonne | Description |
|---|---|
| date_trajet | Date de circulation du train |
| train_id | Identifiant du train |
| type_train | Type de train |
| ligne | Ligne ferroviaire |
| gare_depart | Gare de départ |
| gare_arrivee | Gare d’arrivée |
| heure_depart_prevue | Heure de départ prévue |
| heure_depart_reelle | Heure de départ réelle |
| heure_arrivee_prevue | Heure d’arrivée prévue |
| heure_arrivee_reelle | Heure d’arrivée réelle |
| cause_retard | Cause observée du retard |

Après nettoyage, plusieurs colonnes sont ajoutées automatiquement :

| Colonne | Description |
|---|---|
| retard_depart_minutes | Retard au départ en minutes |
| retard_arrivee_minutes | Retard à l’arrivée en minutes |
| train_en_retard | Variable cible binaire |
| statut | Statut du train : en retard ou à l’heure |
| jour_semaine | Jour de la semaine |
| heure_depart | Heure prévue de départ |

## 2.2 Définition de la variable cible

La variable cible utilisée pour le Machine Learning est :

```text
train_en_retard