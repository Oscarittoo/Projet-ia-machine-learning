import os
import numpy as np
import pandas as pd


def load_data(data_dir: str):
    """Charge les trois fichiers CSV depuis le dossier Data/."""
    general_path = os.path.join(data_dir, "general_data.csv")
    manager_path = os.path.join(data_dir, "manager_survey_data.csv")
    survey_path = os.path.join(data_dir, "employee_survey_data.csv")

    general = pd.read_csv(general_path)
    manager = pd.read_csv(manager_path)
    survey = pd.read_csv(survey_path)

    print("general_data :", general.shape)
    print("manager_survey_data :", manager.shape)
    print("employee_survey_data :", survey.shape)

    return general, manager, survey


def merge_data(general: pd.DataFrame,
               manager: pd.DataFrame,
               survey: pd.DataFrame) -> pd.DataFrame:
    """Fusionne les trois tables sur EmployeeID."""
    df = general.merge(manager, on="EmployeeID", how="left") \
                .merge(survey, on="EmployeeID", how="left")
    print("Shape après fusion :", df.shape)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage minimum :
    - remplace les 'NA' texte par NaN
    - supprime les colonnes constantes
    - affiche les taux de valeurs manquantes
    """
    df = df.replace("NA", np.nan)

    missing_ratio = df.isna().mean().sort_values(ascending=False)
    print("\nTaux de valeurs manquantes (top 10) :")
    print(missing_ratio.head(10))

    const_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if const_cols:
        print("\nColonnes constantes supprimées :", const_cols)
        df = df.drop(columns=const_cols)

    return df


def save_clean_data(df: pd.DataFrame, data_dir: str,
                    filename: str = "hr_merged_clean.csv") -> str:
    """Sauvegarde le DataFrame nettoyé dans le dossier Data/."""
    output_path = os.path.join(data_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"\nFichier nettoyé sauvegardé : {output_path}")
    return output_path


def main():
    # base_dir = Projet IA MachineLearning
    base_dir = os.path.dirname(os.path.dirname(__file__))

    # dossier Data (attention à la majuscule, comme dans ton projet)
    data_dir = os.path.join(base_dir, "Data")

    print("Dossier Data utilisé :", data_dir)

    general, manager, survey = load_data(data_dir)
    df = merge_data(general, manager, survey)
    df = basic_cleaning(df)

    print("\nAperçu des données nettoyées :")
    print(df.head())

    save_clean_data(df, data_dir)


if __name__ == "__main__":
    print("Script Load_and_prepare.py lancé ")
    main()
