import pandas as pd


# 2. Chargement du Dataset
# Si vous utilisez Google Colab, importez d'abord le fichier.
def load_data('/Users/abdel/Projects/dev/Bank_telemarketing/bank-additional-full.csv'):
    """
    Charger le fichier CSV contenant les données de télémarketing bancaire.

    Paramètres : filepath: Chemin vers le fichier CSV.

    Retourne : DataFrame Pandas ou None en cas d'erreur.
    """
    try:
        # séparateur : le point-virgule
        df = pd.read_csv(filepath, sep=';')
        print(f"Dataset chargé avec succès")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement du Data : {e}")
        return None
