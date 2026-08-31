"""
Script d'entraînement — Télémarketing Bancaire.

Ce script est le pendant "production" du notebook Bank_customer_Deposit.ipynb.
Il exécute le même pipeline (imputation → encodage → split → SMOTE →
sélection de variables → comparaison de modèles), dans un ORDRE légèrement
différent du notebook original — le split train/test est fait AVANT
l'imputation et l'encodage, et non après, afin d'éviter que les statistiques
d'imputation (mode, moyenne) et les colonnes d'encodage soient calculées en
tenant compte des données de test (fuite de données).

Méthode de sélection de variables retenue pour le déploiement : Importance
Random Forest (décision prise le 23/08/2026, pour son interprétabilité).

Ce script sauvegarde à la fin tous les artefacts nécessaires à l'inférence
dans le dossier models/ :
    - model.pkl                    : le modèle gagnant (meilleur ROC-AUC)
    - model_name.pkl               : son nom (pour affichage dans l'app)
    - stats_imputation.pkl         : modes/moyennes calculés sur le train
    - colonnes_reference.pkl       : colonnes exactes après encodage du train
    - variables_selectionnees.pkl  : variables retenues par la sélection RF
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from data import load_data
from preprocessing import imputer_valeurs_manquantes, encoder_variables_categorielles
from feature_selection import selectionner_variables_random_forest
from model import developper_et_evaluer_modeles

CHEMIN_DONNEES = os.environ.get("BANK_DATA_PATH", "data/bank_marketing.csv")
DOSSIER_MODELS = "models"


def main():
    os.makedirs(DOSSIER_MODELS, exist_ok=True)

    # 1. Chargement (fonction originale, inchangée)
    donnees = load_data(CHEMIN_DONNEES)
    if donnees is None:
        raise SystemExit("Chargement des données impossible, arrêt du script.")

    # 2. Split AVANT tout traitement dépendant des données, pour que les stats
    #    d'imputation et les colonnes d'encodage ne voient jamais le test set
    df_train, df_test = train_test_split(
        donnees, test_size=0.2, random_state=42, stratify=donnees['y']
    )
    print(f"Train : {df_train.shape[0]:,} lignes | Test : {df_test.shape[0]:,} lignes")

    # 3. Imputation — stats calculées sur train, réappliquées telles quelles sur test
    df_train, stats_imputation = imputer_valeurs_manquantes(df_train)
    df_test, _ = imputer_valeurs_manquantes(df_test, stats=stats_imputation)

    # 4. Encodage — colonnes de référence figées sur train, réappliquées sur test
    df_train_encode = encoder_variables_categorielles(df_train, target='y')
    colonnes_reference = df_train_encode.columns.tolist()
    df_test_encode = encoder_variables_categorielles(
        df_test, target='y', colonnes_reference=colonnes_reference
    )

    X_train = df_train_encode.drop(columns=['y'])
    y_train = df_train_encode['y']
    X_test = df_test_encode
    y_test = df_test['y'].map({'no': 0, 'yes': 1})

    # 5. SMOTE — rééquilibrage du train uniquement (fonction originale du notebook,
    #    reprise ici telle quelle, sans passer par un module séparé car elle ne
    #    sert qu'à l'entraînement, jamais à l'inférence)
    print("\nDistribution avant SMOTE (entraînement) :")
    print(y_train.value_counts())
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("\nDistribution après SMOTE (entraînement) :")
    print(pd.Series(y_train_resampled).value_counts())

    # 6. Sélection de variables — Importance Random Forest (méthode retenue)
    variables_selectionnees, X_train_final, X_test_final = selectionner_variables_random_forest(
        X_train_resampled, y_train_resampled, X_test
    )

    # 7. Comparaison des 3 modèles (fonction originale, modifiée pour retourner
    #    aussi les modèles entraînés — voir commentaire "AJOUT" dans model.py)
    resultats, modeles_entraines = developper_et_evaluer_modeles(
        X_train_final, X_test_final, y_train_resampled, y_test,
        label_methode="Importance RF"
    )

    # 8. Sélection du modèle gagnant selon le ROC-AUC (même logique que la
    #    conclusion du notebook original)
    meilleur_nom = max(resultats, key=lambda k: resultats[k]['ROC-AUC'])
    meilleur_modele = modeles_entraines[meilleur_nom]
    print(f"\nModèle retenu pour le déploiement : {meilleur_nom} "
          f"(ROC-AUC = {resultats[meilleur_nom]['ROC-AUC']:.4f})")

    # 9. Sauvegarde des artefacts nécessaires à l'inférence
    joblib.dump(meilleur_modele, os.path.join(DOSSIER_MODELS, "model.pkl"))
    joblib.dump(meilleur_nom, os.path.join(DOSSIER_MODELS, "model_name.pkl"))
    joblib.dump(stats_imputation, os.path.join(DOSSIER_MODELS, "stats_imputation.pkl"))
    joblib.dump(colonnes_reference, os.path.join(DOSSIER_MODELS, "colonnes_reference.pkl"))
    joblib.dump(variables_selectionnees, os.path.join(DOSSIER_MODELS, "variables_selectionnees.pkl"))

    print(f"\nArtefacts sauvegardés dans '{DOSSIER_MODELS}/' :")
    for f in ["model.pkl", "model_name.pkl", "stats_imputation.pkl",
              "colonnes_reference.pkl", "variables_selectionnees.pkl"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
