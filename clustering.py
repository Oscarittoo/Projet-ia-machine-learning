import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer


def load_clean_data(data_dir: str, filename: str = "hr_merged_clean.csv") -> pd.DataFrame:
    """Charge le fichier fusionné/nettoyé."""
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    print("Données propres chargées pour le clustering :", df.shape)
    return df


def build_cluster_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sélectionne les variables à utiliser pour le clustering.
    On garde ici des variables continues / ordinales liées :
    - à l'âge
    - au salaire
    - à l'ancienneté
    - aux satisfactions
    """

    vars_cluster = [
        "Age",
        "MonthlyIncome",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsSinceLastPromotion",
        "JobSatisfaction",
        "EnvironmentSatisfaction",
        "WorkLifeBalance",
        "JobInvolvement",
        "PerformanceRating",
    ]

    # On garde seulement celles qui existent vraiment
    vars_cluster = [v for v in vars_cluster if v in df.columns]

    print("Variables utilisées pour le clustering :", vars_cluster)

    # DataFrame des variables choisies
    sub_df = df[vars_cluster].copy()

    return sub_df, vars_cluster


def preprocess_for_clustering(sub_df: pd.DataFrame) -> np.ndarray:
    """
    Imputation des NaN + standardisation pour le clustering.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Imputation
    imputed = imputer.fit_transform(sub_df)

    # Standardisation
    scaled = scaler.fit_transform(imputed)

    return scaled


def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> np.ndarray:
    """Applique KMeans et renvoie les labels (numéro de cluster par individu)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels


def main():
    # Dossier du projet et de Data
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "Data")

    # 1) Chargement
    df = load_clean_data(data_dir)

    # 2) Sélection des variables pour le clustering
    sub_df, vars_cluster = build_cluster_dataset(df)

    # 3) Prétraitement (NaN + scaling)
    X_scaled = preprocess_for_clustering(sub_df)

    # 4) KMeans
    n_clusters = 3  # tu peux tester 4, 5, etc. si tu veux
    labels = run_kmeans(X_scaled, n_clusters=n_clusters)

    # 5) Ajout des labels de cluster au DataFrame original
    df["cluster"] = labels

    print("\nRépartition des employés par cluster :")
    print(df["cluster"].value_counts().sort_index())

    # 6) Description des clusters (moyennes des variables numériques utilisées)
    print("\nMoyenne des variables par cluster :")
    cluster_means = df.groupby("cluster")[vars_cluster].mean()
    print(cluster_means)

    # 7) Taux d'attrition par cluster (très utile pour l'interprétation)
    if "Attrition" in df.columns:
        attrition_rates = (
            df.groupby("cluster")["Attrition"]
              .value_counts(normalize=True)
              .rename("proportion")
              .reset_index()
              .pivot(index="cluster", columns="Attrition", values="proportion")
        )
        print("\nTaux d'attrition par cluster :")
        print(attrition_rates)

    # 8) Sauvegarde des données avec le numéro de cluster
    output_path = os.path.join(data_dir, "hr_with_clusters.csv")
    df.to_csv(output_path, index=False)
    print(f"\nFichier avec clusters sauvegardé : {output_path}")


if __name__ == "__main__":
    print("Script clustering.py lancé ")
    main()
