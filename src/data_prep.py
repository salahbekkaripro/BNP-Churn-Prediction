import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Étape 1 : Chargement des données ---
print("--- Début du projet : Chargement des données ---")
# On charge le fichier CSV qui contient les informations des clients
df = pd.read_csv('data/raw/Churn_Modelling.csv')
print(f"Le fichier contient {df.shape[0]} clients et {df.shape[1]} colonnes.")

# On regarde s'il y a des données manquantes pour éviter les erreurs plus tard
print("\nVérification des valeurs manquantes :")
print(df.isnull().sum().sum(), "valeur(s) manquante(s) trouvée(s).")

# On vérifie si on a beaucoup de clients qui partent (Exited = 1)
print("\nRépartition des clients (0 = reste, 1 = part) :")
print(df['Exited'].value_counts(normalize=True) * 100)

# --- Étape 2 : Sélection des variables ---
# On enlève les colonnes qui ne servent pas à la prédiction (ID, Nom...)
target = 'Exited'
# Variables numériques (chiffres)
numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
# Variables catégorielles (mots)
categorical_features = ['Geography', 'Gender']

# --- Étape 3 : Préparation des données (Preprocessing) ---
print("\nPréparation des transformations (Standardisation et Encodage)...")

# Pour les chiffres : on les centre et on les réduit pour qu'ils soient à la même échelle
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Pour les mots : on les transforme en chiffres (0 ou 1) car le modèle ne lit que les chiffres
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# On regroupe toutes les transformations dans un seul objet
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- Étape 4 : Séparation des données ---
print("Séparation des données en deux groupes : Entraînement (80%) et Test (20%).")
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Étape 5 : Analyse des valeurs atypiques (Outliers) ---
# On calcule le Z-Score pour voir si certains clients ont un solde très bizarre (très élevé)
print("Analyse des clients 'Gros Comptes' via le Z-Score...")
mean_balance = X_train['Balance'].mean()
std_balance = X_train['Balance'].std()
X_train['Z_Score_Balance'] = (X_train['Balance'] - mean_balance) / std_balance

# On marque les clients qui ont un solde à plus de 3 écart-types de la moyenne
X_train['High_Value_Outlier'] = (X_train['Z_Score_Balance'] > 3).astype(int)
print(f"Nombre de clients atypiques identifiés : {X_train['High_Value_Outlier'].sum()}")

# --- Étape 6 : Application finale des transformations ---
X_train_transformed = preprocessor.fit_transform(X_train[numeric_features + categorical_features])
X_test_transformed = preprocessor.transform(X_test[numeric_features + categorical_features])
feature_names = preprocessor.get_feature_names_out()

print("--- Fin de la préparation des données ---")
