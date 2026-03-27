import pandas as pd
from data_prep import X_train, X_train_transformed
from modeling import rf_model
from segmentation import kmeans, indices

# --- Étape 1 : Préparation du fichier pour Power BI ---
print("--- Début de l'exportation pour Power BI ---")

# On prend les colonnes utiles pour faire un beau tableau de bord
colonnes_utiles = ['CustomerId', 'Geography', 'Gender', 'Age', 'Balance', 'High_Value_Outlier']
df_pbi = X_train[colonnes_utiles].copy()

# On ajoute le risque de départ calculé par notre modèle
print("Ajout des probabilités de départ...")
df_pbi['Risque_Churn'] = rf_model.predict_proba(X_train_transformed)[:, 1]

# On ajoute le groupe de clients calculé par la segmentation
print("Ajout des numéros de groupes...")
X_financial = X_train_transformed[:, indices]
df_pbi['Numero_Groupe'] = kmeans.predict(X_financial)

# --- Étape 2 : Donner des noms sympas aux groupes ---
print("Traduction des numéros de groupes en noms compréhensibles...")
noms_groupes = {
    0: "Clients fragiles (faible solde)",
    1: "Clients fidèles",
    2: "Clients Premium à risque"
}
df_pbi['Nom_Groupe'] = df_pbi['Numero_Groupe'].map(noms_groupes)

# --- Étape 3 : Sauvegarde finale ---
df_pbi.to_csv('data/processed/Client_Segmentation_PowerBI.csv', index=False)
print("\n--- Exportation réussie : Fichier data/processed/Client_Segmentation_PowerBI.csv créé ---")
