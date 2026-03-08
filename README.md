# Télémarketing Bancaire — Scoring de Souscription Client à un compte de dépôt à long terme:

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![SMOTE](https://img.shields.io/badge/SMOTE-Rééquilibrage-purple)
![Statut](https://img.shields.io/badge/Statut-Complet-brightgreen)
![ROC-AUC](https://img.shields.io/badge/Meilleur%20ROC--AUC-0.88-success)

---

## Contexte Métier du projet:

Dans le secteur bancaire, les campagnes de téléprospection représentent un coût opérationnel significatif. **Cibler des clients non receptifs, entraine la perte des ressources commerciales et dégrade l'expérience client.**

L'enjeu de ce projet est de réponde à une problématique concrète : **comment identifier, avant l'appel, les clients les plus susceptibles de souscrire à un compte de dépôt à long terme ?**

L'approche retenue repose sur un pipeline Machine Learning complet, depuis l'analyse exploratoire jusqu'au scoring final, avec comparaison de plusieurs algorithmes de classification.

> **Jeu de données :** Données réelles d'une banque portugaise — 41 187 contacts téléphoniques collectés entre mai 2008 et juin 2013.
> **Déséquilibre de classes :** Seulement 11,26 % de souscriptions réussies (4 640 sur 41 187).

## Approche Technique:

### Pipeline complet

```
Données brutes (41 187 contacts)
        │
        ▼
┌──────────────────────────┐
│  EDA & Statistiques      │  → Analyse de 20 variables (démographiques,
│  Descriptives            │    comportementales, macro-économiques)
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Preprocessing           │  → Encodage One-Hot (détection automatique
│                          │    des colonnes catégorielles via select_dtypes)
│                          │    Imputation des 'unknown' par le mode,
│                          │    des NaN par la moyenne
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Découpage Train/Test    │  → 80% / 20%, stratifié (stratify=y)
│  (avant SMOTE)           │    pour maintenir le déséquilibre de classes
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Rééquilibrage SMOTE     │  → Appliqué UNIQUEMENT sur l'ensemble
│  (train uniquement)      │    d'entraînement — le test n'est jamais modifié
└──────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  Sélection des Variables (2 méthodes)        │
│  ├── ACP — Analyse en Composantes            │
│  │         Principales (95% de variance)     │
│  └── Random Forest — Importance des          │
│                       variables (> moyenne)  │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Modélisation            │  → Régression Logistique, Random Forest,
│  & Benchmarking          │    XGBoost
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Évaluation              │  → ROC-AUC, Accuracy, Matrices de confusion,
└──────────────────────────┘    Courbes ROC comparatives
```

## Résultats:

### Comparaison des modèles — Sélection par ACP

| Modèle | Accuracy | ROC-AUC |
|---|---|---|
| Régression Logistique | 0,85 | 0,87 |
| Random Forest | 0,86 | 0,88 |
| XGBoost | 0,87 | **0,89** |

### Comparaison des modèles — Sélection par Importance Random Forest

| Modèle | Accuracy | ROC-AUC |
|---|---|---|
| Régression Logistique | 0,84 | 0,86 |
| Random Forest | **0,88** | **0,90** |
| XGBoost | 0,87 | 0,89 |

### Meilleur modèle : Random Forest + Importance des variables

**ROC-AUC = 0,90** — le modèle distingue efficacement les souscripteurs potentiels des non-souscripteurs dans 90 % des cas, contre 50 % pour une prédiction aléatoire.

### Matrice de Confusion — Random Forest (Importance des variables)

|  | Prédit : Non | Prédit : Oui |
|---|---|---|
| **Réel : Non** | 3 900 | 100 |
| **Réel : Oui** | 250 | 750  |

**Lecture métier :**
- **750 vrais positifs** → clients souscripteurs correctement identifiés
- **100 faux positifs** → appels inutiles évités par le modèle
- **250 faux négatifs** → clients potentiels manqués (acceptable sur ce volume)

**Impact opérationnel :** En ne ciblant que les clients avec un score élevé, la banque réduit le volume d'appels inutiles tout en maximisant son taux de conversion.


## Résultats d'analyse:

### Profil du client le plus susceptible de souscrire

| Variable | Segment à fort potentiel | Interprétation métier |
|---|---|---|
| Âge | 30–50 ans | Revenus stables, projets d'épargne (immobilier, retraite) |
| Profession | Management, Technicien | Revenus réguliers favorisant les placements |
| Statut matrimonial | Marié | Projets communs (maison, retraite) |
| Éducation | Diplôme universitaire | Meilleure compréhension des produits financiers |
| Défaut de crédit | Aucun | Profil de risque favorable |
| Résultat campagne précédente | Succès | Fidélité produit démontrée |
| Taux Euribor 3M | Bas | Contexte macro favorable aux placements bancaires |

### Facteurs opérationnels identifiés

- **Durée d'appel > 4 minutes** → fortement corrélée à la souscription (variable exclue du modèle — non disponible avant l'appel)
- **Contact en mai–juin** → meilleur taux de conversion observé
- **Prêt personnel actif** → facteur défavorable (contrainte financière)
- **Contact via téléphone portable** → légèrement plus efficace que le fixe

---

## Points Techniques:

**Détection automatique des colonnes catégorielles**
L'encodage utilise `select_dtypes(include=["object", "category"])` pour identifier automatiquement les colonnes à encoder — le pipeline est robuste aux évolutions du dataset.

**Gestion du déséquilibre de classes**
Avec seulement 11,26 % de souscriptions, un modèle naïf prédirait "non" en permanence à 88,74 % d'accuracy sans rien apprendre. SMOTE synthétise des exemples de la classe minoritaire **uniquement sur l'ensemble d'entraînement** — le test set est toujours évalué sur données réelles.

**Prévention rigoureuse du data leakage**
Trois précautions appliquées : (1) découpage train/test avant SMOTE, (2) StandardScaler et ACP fittés sur le train uniquement puis appliqués en `transform` sur le test, (3) variable `duration` supprimée car non disponible avant l'appel.

**Gestion défensive de la variable cible**
La fonction d'évaluation inclut une normalisation automatique de `y` (`no/yes` → `0/1`) pour garantir la compatibilité quel que soit l'état du pipeline en entrée.

**Deux méthodes de sélection comparées**
L'ACP réduit la dimensionnalité mais perd l'interprétabilité des variables originales. La sélection par importance Random Forest conserve les variables originales et permet une lecture métier directe — résultat confirmé par les métriques : ROC-AUC 0,90 vs 0,89 pour l'ACP.

---

## Structure du Projet

```
Bank_telemarketing/
│
├── Bank_customer_Deposit_FR.ipynb  # Notebook principal — pipeline complet commenté en français
├── bank_marketing.csv              # Jeu de données (41 187 observations, 21 variables)
├── bank-additional-names.txt       # Description détaillée des attributs
├── requirements.txt                # Dépendances Python
└── README.md                       # Ce fichier
```

## Reproduire le Projet

### Installation des dépendances

```bash
pip install -r requirements.txt
```

### Lancer le notebook

```bash
git clone https://github.com/Abdelali-Merhom/Bank_telemarketing.git
cd Bank_telemarketing
jupyter notebook Bank_customer_Deposit.ipynb
```

> **Note :** Le fichier `bank_marketing.csv` doit se trouver dans le même répertoire que le notebook. Le chemin est relatif et fonctionne pour tous les utilisateurs qui clonent le dépôt.

---

## 📈 Pistes d'Amélioration

- [ ] Optimisation du seuil de décision selon le coût métier (appel inutile vs client manqué)
- [ ] Test de la méthode Boruta pour la sélection de variables
- [ ] Ajout de valeurs SHAP pour l'explicabilité des prédictions individuelles
- [ ] Déploiement d'une interface Streamlit pour le scoring en temps réel
- [ ] Validation sur fenêtre temporelle glissante (backtesting)

---

# Auteur

**Abdelali MERHOM** — Data Scientist spécialisé en scoring, détection d'anomalies et données réglementées
[LinkedIn](https://www.linkedin.com/in/Abdelali-merhom/) • [GitHub](https://github.com/Abdelali-Merhom)

---

*Données issues du UCI Machine Learning Repository — Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict the success of bank telemarketing. Decision Support Systems.*
