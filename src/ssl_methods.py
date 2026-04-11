from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evaluation import compute_metrics, metrics_to_dataframe
from models import (
    build_label_spreading_model,
    build_self_training_model,
    build_supervised_baseline,
)

def parse_args():
    parser = argparse.ArgumentParser(description="SSL pipeline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--labeled-ratio", type=float, default=0.2, help="Labeled ratio")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")
    return parser.parse_args()

def save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    processed_path = data_dir / "air_quality_processed.csv"

    if not processed_path.exists():
        raise FileNotFoundError(
            "No se encontró data/air_quality_processed.csv. Ejecuta primero src/preprocessing.py"
        )
    df = pd.read_csv(processed_path)
    if "target" not in df.columns:
        raise ValueError("El dataset procesado no contiene la columna target")
    drop_cols = [c for c in ["Name", "Start_Date"] if c in df.columns]
    X = df.drop(columns=drop_cols + ["target"])
    # Convierte categóricas a variables binarias para evitar fallos de casteo.
    X = pd.get_dummies(X, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    y = df["target"].astype(int)
    
    numeric_cols = [
        "Geo Join ID",
        "Data Value",
        "year",
        "month",
        "day_of_week",
    ]
    cols_to_scale = [c for c in numeric_cols if c in X.columns]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    scaler = StandardScaler()
    X_train.loc[:, cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test.loc[:, cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X_train, y_train, test_size=(1.0 - args.labeled_ratio),
        random_state=args.seed, stratify=y_train,
    )
    # Serialización para trazabilidad/reusabilidad
    save_pickle(X_labeled, data_dir / "X_labeled.pkl")
    save_pickle(y_labeled, data_dir / "y_labeled.pkl")
    save_pickle(X_unlabeled, data_dir / "X_unlabeled.pkl")
    save_pickle(y_unlabeled, data_dir / "y_unlabeled.pkl")
    save_pickle(X_test, data_dir / "X_test.pkl")
    save_pickle(y_test, data_dir / "y_test.pkl")

    rows = []
    # RANDOM FOREST SUPERVISED
    baseline = build_supervised_baseline(random_state=args.seed)
    baseline.fit(X_labeled, y_labeled)
    pred_baseline = baseline.predict(X_test)
    rows.append({"model": "SupervisedBaseline(RandomForest)", **compute_metrics(y_test, pred_baseline)})

    # SELF-TRAINING CLASSIFIER (SSL)
    ssl_self = build_self_training_model(random_state=args.seed)
    X_ssl = pd.concat([X_labeled, X_unlabeled], axis=0)
    y_ssl = np.concatenate([y_labeled.to_numpy(), np.full(shape=len(X_unlabeled), fill_value=-1, dtype=int)])
    ssl_self.fit(X_ssl, y_ssl)
    pred_self = ssl_self.predict(X_test)
    rows.append({"model": "SemiSupervised(SelfTraining)", **compute_metrics(y_test, pred_self)})

    # LABEL SPREADING (SSL)
    ssl_graph = build_label_spreading_model()
    ssl_graph.fit(X_ssl, y_ssl)
    pred_graph = ssl_graph.predict(X_test)
    rows.append({"model": "SemiSupervised(LabelSpreading)", **compute_metrics(y_test, pred_graph)})

    # REPORTE
    report = metrics_to_dataframe(rows)
    report_path = data_dir / "ssl_results.csv"
    report.to_csv(report_path, index=False)
    print(f"Pipeline SSL completado")
    print(f"Labeled train size: {len(X_labeled)}")
    print(f"Unlabeled train size: {len(X_unlabeled)}")
    print(f"Test size: {len(X_test)}")
    print("\nResultados (ordenados por F1):")
    print(report.to_string(index=False))
    print(f"\nArchivo de resultados: {report_path}")

if __name__ == "__main__":
    main()
