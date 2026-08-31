import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    roc_curve
)


# 10. Développement et évaluation des modèles

# Trois algorithmes sont comparés :
#   1. Régression Logistique: modèle linéaire de référence (baseline)
#   2. Random Forest: ensemble d'arbres de décision robuste aux outliers
#   3. XGBoost: gradient boosting performant sur données déséquilibrées
#
# Métriques d'évaluation :
#   - Accuracy: proportion de prédictions correctes
#   - ROC-AUC: capacité du modèle à distinguer les deux classes (0 à 1)
#   - Matrice de confusion: détail des vrais/faux positifs et négatifs
#   - Rapport de classification: précision, rappel et F1-score par classe
def developper_et_evaluer_modeles(X_train, X_test, y_train, y_test, label_methode=""):
    """
    Entraîne et évalue trois modèles de classification.

    Paramètres :
        X_train (DataFrame)    : Variables d'entraînement (après sélection de variables)
        X_test (DataFrame)     : Variables de test
        y_train (array)        : Étiquettes d'entraînement (rééchantillonnées par SMOTE)
        y_test (array)         : Étiquettes de test (originales, jamais modifiées)
        label_methode (str)    : Libellé de la méthode de sélection pour les graphiques

    Retourne :
        Dictionnaire des résultats {nom_modele: {Accuracy, ROC-AUC}},
        Dictionnaire des modèles entraînés {nom_modele: modele}  [AJOUT pour le déploiement]
    """
    # Normalisation de la cible: 'no'/'yes' --> 0/1
    def _preparer_y(y):
        y_series = pd.Series(y).copy()
        valeurs_uniques = set(y_series.unique())
        # Cas labels texte
        if valeurs_uniques <= {"no", "yes"}:
            mapping = {"no": 0, "yes": 1}
            y_series = y_series.map(mapping)
        return y_series

    y_train_prepared = _preparer_y(y_train)
    y_test_prepared = _preparer_y(y_test)

    # Définition des modèles à comparer
    modeles = {
        "Régression Logistique": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":         RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost":               XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    }

    resultats = {}

    for nom_modele, modele in modeles.items():
        # Entraînement du modèle sur les données rééchantillonnées
        modele.fit(X_train, y_train_prepared)

        # Prédictions sur le jeu de test (jamais vu pendant l'entraînement)
        y_pred = modele.predict(X_test)              # Classe prédite (0 ou 1)
        y_prob = modele.predict_proba(X_test)[:, 1]  # Probabilité de souscription

        # Calcul des métriques
        accuracy = accuracy_score(y_test_prepared, y_pred)
        roc_auc  = roc_auc_score(y_test_prepared, y_prob)
        resultats[nom_modele] = {"Accuracy": round(accuracy, 4), "ROC-AUC": round(roc_auc, 4)}

        print(f"\n{'='*55}")
        print(f"  {nom_modele} — Sélection : {label_methode}")
        print(f"{'='*55}")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"\n  Rapport de classification détaillé :")
        print(classification_report(y_test_prepared, y_pred, target_names=['Non souscripteur (0)', 'Souscripteur (1)']))

        # --- Matrice de confusion ---
        # Lecture :
        #   Vrai Négatif (VN)  : prédit Non, réel Non  → appels évités correctement
        #   Faux Positif (FP)  : prédit Oui, réel Non  → appels inutiles
        #   Faux Négatif (FN)  : prédit Non, réel Oui  → clients souscripteurs manqués
        #   Vrai Positif (VP)  : prédit Oui, réel Oui  → conversions correctement identifiées
        cm = confusion_matrix(y_test_prepared, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Prédit : Non', 'Prédit : Oui'],
                    yticklabels=['Réel : Non', 'Réel : Oui'])
        plt.title(f"Matrice de confusion — {nom_modele} ({label_methode})",
                  fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # --- Courbe ROC combinée pour comparaison des trois modèles ---
    # La courbe ROC représente le compromis entre le taux de vrais positifs
    # et le taux de faux positifs selon le seuil de décision.
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5,
             label='Classifieur aléatoire (AUC = 0,50)')

    couleurs = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for (nom_modele, modele), couleur in zip(modeles.items(), couleurs):
        y_prob = modele.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_prepared, y_prob)
        auc = resultats[nom_modele]['ROC-AUC']
        plt.plot(fpr, tpr, linewidth=2, color=couleur,
                 label=f"{nom_modele} (AUC = {auc:.2f})")

    plt.xlabel("Taux de faux positifs (1 - Spécificité)")
    plt.ylabel("Taux de vrais positifs (Sensibilité / Rappel)")
    plt.title(f"Comparaison des courbes ROC — Sélection : {label_methode}",
              fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # AJOUT pour le déploiement : retour des modèles entraînés en plus des métriques.
    # Sans cet ajout, il faudrait ré-entraîner le modèle gagnant une seconde fois
    # dans train.py pour pouvoir le sauvegarder — ce qui duplique le calcul et
    # peut produire un modèle légèrement différent si une source d'aléa n'est
    # pas parfaitement figée. Retourner directement les objets déjà entraînés
    # est la solution la plus sûre et la plus économe.
    return resultats, modeles
