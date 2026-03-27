import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_prep import X_train_transformed, X_test_transformed, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# --- Étape 1 : Création des modèles ---
print("--- Début de la Modélisation ---")

# On teste d'abord un modèle simple : la Régression Logistique
print("Entraînement du modèle simple (Régression Logistique)...")
lr_model = LogisticRegression(class_weight='balanced', random_state=42)
lr_model.fit(X_train_transformed, y_train)

# Ensuite, on teste un modèle plus puissant : le Random Forest (Forêt Aléatoire)
print("Entraînement du modèle complexe (Random Forest)...")
rf_model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf_model.fit(X_train_transformed, y_train)

# --- Étape 2 : Comparaison des performances ---
print("\nComparaison des scores AUC-ROC (plus c'est proche de 1, mieux c'est) :")
y_probs_lr = lr_model.predict_proba(X_test_transformed)[:, 1]
y_probs_rf = rf_model.predict_proba(X_test_transformed)[:, 1]

auc_lr = roc_auc_score(y_test, y_probs_lr)
auc_rf = roc_auc_score(y_test, y_probs_rf)
print(f"Score AUC Régression Logistique : {auc_lr:.3f}")
print(f"Score AUC Random Forest : {auc_rf:.3f}")

# --- Étape 3 : Optimisation pour le métier ---
# Dans notre cas, on préfère détecter trop de clients à risque plutôt que d'en rater.
# On change le seuil de décision à 0.35 au lieu de 0.50.
print("\nOptimisation du seuil de détection (0.35) pour augmenter le Recall.")
threshold = 0.35
y_pred_rf_custom = (y_probs_rf >= threshold).astype(int)

# --- Étape 4 : Sauvegarde des graphiques ---
print("Génération et sauvegarde des graphiques de performance...")

# Courbe ROC pour comparer les deux modèles visuellement
plt.figure(figsize=(10, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_probs_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
plt.plot(fpr_lr, tpr_lr, label=f'Régression Logistique (AUC = {auc_lr:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (seuil de chance)')
plt.xlabel('Faux Positifs')
plt.ylabel('Vrais Positifs (Recall)')
plt.title('Comparaison des modèles')
plt.legend()
plt.savefig('exports/roc_curves.png')

# Matrice de confusion pour voir où le modèle se trompe
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf_custom)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Où le modèle se trompe (Matrice de Confusion)')
plt.savefig('exports/confusion_matrix_rf.png')

print("--- Fin de la modélisation : Les graphiques sont dans le dossier exports/ ---")
