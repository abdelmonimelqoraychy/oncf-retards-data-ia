import pandas as pd


DATA_PATH = "data/raw/circulations_oncf_sample.csv"


def charger_donnees(path):
    df = pd.read_csv(path)
    return df


df = charger_donnees(DATA_PATH)

print("===== APERÇU DES DONNÉES =====")
print(df.head())

print("\n===== INFORMATIONS =====")
print(df.info())

print("\n===== DIMENSION DU DATASET =====")
print(df.shape)

print("\n===== COLONNES =====")
print(df.columns)