# 🎓 Rapport de Projet : Prédiction du Churn Clients (BNP)
**Auteur : Salah-eddine BEKKARI**  
**Niveau : Licence MIASHS / Master MIAGE**

---

## 1. Introduction et Problématique
Ce projet a pour but d'aider les équipes de la BNP à mieux comprendre pourquoi certains clients quittent la banque (le "Churn"). L'idée est de créer un outil capable de prédire si un client va partir, afin de pouvoir lui proposer une offre adaptée avant qu'il ne s'en aille.

## 2. Méthodologie (Ma démarche)
Pour répondre à ce problème, j'ai suivi les étapes suivantes :
1.  **Préparation des données (`src/data_prep.py`)** : J'ai nettoyé les données et transformé les variables (comme le pays ou le genre) en chiffres pour que l'ordinateur puisse les lire. J'ai aussi cherché les "Gros Comptes" qui ont des soldes très élevés.
2.  **Modélisation (`src/modeling.py`)** : J'ai testé deux modèles. Le modèle **Random Forest** est celui qui a le mieux fonctionné avec un score AUC de 0.85. J'ai aussi réglé le modèle pour qu'il soit plus sensible aux départs.
3.  **Segmentation (`src/segmentation.py`)** : J'ai regroupé les clients en 3 groupes (clusters) selon leur âge et leur solde bancaire pour mieux les cibler.
4.  **Export Power BI (`src/export_powerbi.py`)** : J'ai créé un fichier final qui regroupe toutes les infos pour faire un beau tableau de bord.

## 3. Analyse des résultats
- **Performance** : Le modèle arrive à bien séparer les clients fidèles des clients qui partent (AUC 0.85).
- **Groupes identifiés** :
    - Le groupe des **"Clients Premium à risque"** est celui qu'il faut surveiller en priorité car ils ont beaucoup d'argent et une forte probabilité de départ.

## 4. Conclusion
Ce projet montre comment la Data Science peut aider une banque à garder ses clients. En utilisant ces scripts, la banque peut maintenant cibler précisément les clients à risque et agir rapidement.

---

## 📂 Organisation des fichiers
- `data/raw/` : Les données brutes (le fichier CSV d'origine).
- `data/processed/` : Les fichiers que j'ai créés pour Power BI.
- `src/` : Mes scripts Python (avec mes explications).
- `exports/` : Mes graphiques (Courbe ROC, Matrice de Confusion...).
