import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


# 8. Sélection des variables — Importance Random Forest
# Le Random Forest calcule l'importance de chaque variable en mesurant la réduction moyenne de l'impureté de Gini.
#
# Seules les variables dont l'importance est supérieure à la moyenne sont conservées.
def selectionner_variables_random_forest(X_train, y_train, X_test):
    """
    Sélectionne les variables importantes via le Random Forest.
    Entraîné sur X_train uniquement — appliqué ensuite à X_test.

    Paramètres :
        X_train (DataFrame): Variables d'entraînement (après SMOTE).
        y_train (array): Étiquettes d'entraînement.
        X_test (DataFrame): Variables de test.

    Retourne :
        Liste des variables sélectionnées, X_train filtré, X_test filtré.
    """
    # Entraînement du Random Forest pour calculer les importances
    rf_selecteur = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_selecteur.fit(X_train, y_train)

    # Récupération et seuillage des importances
    importances = rf_selecteur.feature_importances_
    seuil = np.mean(importances)  # Seuil = importance moyenne
    variables_importantes = X_train.columns[importances > seuil].tolist()

    print(f"Seuil d'importance (moyenne) : {seuil:.4f}")
    print(f"Nombre de variables sélectionnées : {len(variables_importantes)} / {X_train.shape[1]}")

    # Visualisation des 20 variables les plus importantes
    df_importances = pd.DataFrame({
        'Variable': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(data=df_importances, x='Importance', y='Variable', palette='viridis')
    plt.title("Top 20 variables les plus importantes — Random Forest",
              fontsize=13, fontweight='bold')
    plt.xlabel("Score d'importance")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()

    print(f"\nVariables sélectionnées : {variables_importantes}")

    return variables_importantes, X_train[variables_importantes], X_test[variables_importantes]
