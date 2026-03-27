import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from data_prep import X_train_transformed, X_train, y_train, feature_names
from modeling import rf_model

# --- Étape 1 : Choix des variables pour la segmentation ---
print("--- Début de la Segmentation Clients ---")
# On choisit des colonnes importantes comme l'Age, le Solde (Balance) et le Nombre de Produits
selected_cols = ['num__Age', 'num__Balance', 'num__NumOfProducts', 'num__IsActiveMember']
indices = [list(feature_names).index(col) for col in selected_cols]
X_financial = X_train_transformed[:, indices]

# --- Étape 2 : Méthode du Coude (Elbow) ---
# On cherche le nombre idéal de groupes (clusters) à créer
print("Calcul de la méthode du Coude pour trouver le bon nombre de groupes...")
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_financial)
    inertia.append(km.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, 'bx-')
plt.title('Trouver le nombre de groupes idéal (Méthode du Coude)')
plt.savefig('exports/elbow_method.png')

# --- Étape 3 : Création des groupes avec K-Means ---
# On a décidé de faire 3 groupes (clusters)
print("Création de 3 groupes de clients (K-Means)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_financial)

# --- Étape 4 : Visualisation avec t-SNE ---
# Le t-SNE permet de mettre des données complexes en 2D pour les voir sur un graphique
print("Préparation du graphique t-SNE pour voir les groupes en 2D...")
tsne = TSNE(n_components=2, perplexity=40, random_state=42)
X_tsne = tsne.fit_transform(X_financial)

plt.figure(figsize=(10, 7))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=clusters, palette='viridis')
plt.title('Visualisation des 3 groupes de clients')
plt.savefig('exports/segmentation_tsne.png')

# --- Étape 5 : Analyse des résultats ---
print("\nAnalyse des groupes créés :")
X_train_analysis = X_train.copy()
X_train_analysis['Groupe'] = clusters
X_train_analysis['Proba_Depart'] = rf_model.predict_proba(X_train_transformed)[:, 1]

# On regarde la moyenne du solde et du risque de départ par groupe
stats = X_train_analysis.groupby('Groupe').agg({
    'Balance': 'mean',
    'Proba_Depart': 'mean'
})
print(stats)

# Export du résultat pour Power BI
X_train_analysis.to_csv('data/processed/final_segments.csv', index=False)
print("\n--- Fin de la segmentation : Fichier final_segments.csv créé ---")
