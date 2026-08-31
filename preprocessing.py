import pandas as pd


# 4. Imputation des valeurs manquantes
# Stratégie d'imputation :
#   - Variables catégorielles: remplacement des 'unknown' par le mode (la modalité la plus fréquente)
#   - Variables numériques: remplacement des NaN par la moyenne de la colonne
def imputer_valeurs_manquantes(df, stats=None):
    """
    Impute les valeurs manquantes du Dataset.
    - 'unknown' dans les variables catégorielles → remplacé par le mode.
    - NaN dans les variables numériques → remplacé par la moyenne.

    Paramètres :
        df (DataFrame) : Dataset brut.
        stats (dict, optionnel) : AJOUT pour le déploiement — si fourni, réutilise
            des modes/moyennes déjà calculés (typiquement à l'entraînement) au lieu
            de les recalculer sur df. Indispensable en inférence, où un DataFrame
            d'un seul client ne permet pas de calculer un mode ou une moyenne fiable.

    Retourne :
        DataFrame avec valeurs imputées, dict des modes/moyennes utilisés (stats_calculees).
    """
    # CORRECTION : copie explicite pour ne pas muter le DataFrame de l'appelant
    # (la version originale modifiait df en place, ce qui est risqué si df est
    # mis en cache ou réutilisé ailleurs, par exemple avec @st.cache_data)
    df = df.copy()

    # Variables catégorielles contenant des valeurs 'unknown'
    variables_categorielles = [
        'job', 'marital', 'education', 'default',
        'housing', 'loan', 'contact', 'month', 'poutcome']
    # CORRECTION : ne traiter que les colonnes réellement présentes dans df,
    # pour éviter un KeyError si une colonne manque (cas d'un DataFrame partiel
    # en inférence)
    variables_categorielles = [c for c in variables_categorielles if c in df.columns]

    stats_calculees = {'modes': {}, 'moyennes': {}}

    for col in variables_categorielles:
        mode_valeur = stats['modes'][col] if stats else df[col].mode()[0]
        stats_calculees['modes'][col] = mode_valeur
        df[col] = df[col].replace('unknown', mode_valeur)

    # Variables numériques pouvant contenir des NaN
    variables_numeriques = [
        'age', 'duration', 'campaign', 'pdays', 'previous',
        'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
        'euribor3m', 'nr.employed']
    variables_numeriques = [c for c in variables_numeriques if c in df.columns]

    for col in variables_numeriques:
        moyenne_valeur = stats['moyennes'][col] if stats else df[col].mean()
        stats_calculees['moyennes'][col] = moyenne_valeur
        # CORRECTION : réassignation explicite au lieu de fillna(..., inplace=True)
        # sur une colonne extraite (df[col]), pattern de "chained assignment"
        # déprécié par pandas et qui ne garantit pas la répercussion sur df
        df[col] = df[col].fillna(moyenne_valeur)

    print("Imputation terminée avec succès!")
    return df, stats_calculees


# 5. Encodage des variables catégorielles (One-Hot Encoding)
# Pour traiter les variables textuelles, on utilise pd.get_dummies() pour créer des variables binaires (0/1) pour chaque modalité.
# La variable 'duration' est supprimée ici pour éviter le data leakage.
# La variable cible 'y' est convertie en binaire : 0 (non) / 1 (oui).
def encoder_variables_categorielles(df, target='y', colonnes_reference=None):
    """
    Applique l'encodage One-Hot sur les variables catégorielles.
    Supprime 'duration' (fuite de données potentielle).
    Encode la variable cible en binaire (0/1).

    Paramètres :
        df (DataFrame): Dataset après imputation.
        target: Nom de la variable cible.
        colonnes_reference (list, optionnel) : AJOUT pour le déploiement — si fourni,
            réaligne le DataFrame encodé sur cette liste de colonnes exacte (celle
            obtenue à l'entraînement), pour garantir que l'encodage d'un nouveau
            client corresponde à ce qu'attend le modèle, même si certaines
            modalités sont absentes du client à scorer.

    Retourne :
        DataFrame encodé.
    """
    # On travaille sur une copie pour ne pas modifier le DataFrame d'origine
    df_copy = df.copy()

    # Identification automatique des colonnes catégorielles (object / category),
    # en excluant la variable cible
    colonnes_categorielles = (
        df_copy.select_dtypes(include=["object", "category"])  # types catégoriels
        .columns
        .tolist()
    )
    if target in colonnes_categorielles:
        colonnes_categorielles.remove(target)

    # Application du One-Hot Encoding
    # drop_first=False : on conserve toutes les modalités pour l'interprétabilité
    df_encode = pd.get_dummies(df_copy, columns=colonnes_categorielles, drop_first=False, dtype=int)

    # Suppression de 'duration' : variable non disponible avant l'appel
    # Son inclusion constituerait une fuite de données vers le modèle prédictif
    if 'duration' in df_encode.columns:
        df_encode = df_encode.drop(columns=['duration'])
        print("Variable 'duration' supprimée — non disponible avant l'appel (prévention du data leakage).")

    # Encodage de la variable cible : 'no' → 0, 'yes' → 1 (si elle est encore catégorielle)
    if target in df_encode.columns and df_encode[target].dtype == 'object':
        df_encode[target] = df_encode[target].map({'no': 0, 'yes': 1})

    # AJOUT pour le déploiement : réalignement sur les colonnes de l'entraînement.
    # La variable cible n'existe pas pour un client à scorer, elle est donc exclue
    # de la liste de référence si présente.
    if colonnes_reference is not None:
        colonnes_a_garder = [c for c in colonnes_reference if c != target]
        df_encode = df_encode.reindex(columns=colonnes_a_garder, fill_value=0)

    print("Encodage terminé !")
    print(f"Dimensions du Dataset encodé : {df_encode.shape}")
    return df_encode
