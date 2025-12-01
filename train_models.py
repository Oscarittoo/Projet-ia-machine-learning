import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_clean_data(data_dir: str, filename: str = "hr_merged_clean.csv") -> pd.DataFrame:
    """Charge le fichier fusionné et nettoyé."""
    path = os.path.join(data_dir, filename)
    df = pd.read_csv(path)
    print("Données propres chargées :", df.shape)
    return df


def prepare_X_y(df: pd.DataFrame):
    """Sépare X et y, encode Attrition, retire EmployeeID."""
    # Cible
    y = df["Attrition"].map({"Yes": 1, "No": 0})

    # On enlève Attrition et EmployeeID des features
    X = df.drop(columns=["Attrition", "EmployeeID"])

    # Colonnes
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()

    print("Variables catégorielles :", cat_cols)
    print("Variables numériques :", num_cols)

    return X, y, cat_cols, num_cols


def build_preprocessor(cat_cols, num_cols):
    """Préprocesseur : imputation + encodage + scaling."""

    # Catégorielles : imputer NaN par la modalité la plus fréquente, puis OneHot
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    # Numériques : imputer NaN par la médiane, puis standardisation
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, cat_cols),
            ("num", num_pipeline, num_cols),
        ]
    )

    return preprocessor


def evaluate_model(name: str, model, X_test, y_test):
    """Affiche les métriques principales pour un modèle."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC :", roc_auc_score(y_test, y_proba))
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, y_pred))


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "Data")

    df = load_clean_data(data_dir)
    X, y, cat_cols, num_cols = prepare_X_y(df)
    preprocessor = build_preprocessor(cat_cols, num_cols)

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # Modèles baselines
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
    }

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("prep", preprocessor),
            ("model", clf),
        ])
        pipe.fit(X_train, y_train)
        evaluate_model(name, pipe, X_test, y_test)

    # GridSearch sur RandomForest
    print("\n=== GridSearch RandomForest ===")

    rf_pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", RandomForestClassifier(random_state=42)),
    ])

    param_grid = {
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
    }

    grid = GridSearchCV(
        rf_pipe,
        param_grid,
        cv=3,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    print("Meilleurs paramètres :", grid.best_params_)

    best_model = grid.best_estimator_
    evaluate_model("RandomForest (optimisé)", best_model, X_test, y_test)


if __name__ == "__main__":
    print("Script train_models.py lancé ")
    main()
