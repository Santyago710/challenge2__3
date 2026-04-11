from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
	"""Compute standard multiclass metrics."""
	return {
		"accuracy": accuracy_score(y_true, y_pred),
		"precision_weighted": precision_score(
			y_true, y_pred, average="weighted", zero_division=0
		),
		"recall_weighted": recall_score(
			y_true, y_pred, average="weighted", zero_division=0
		),
		"f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
	}


def metrics_to_dataframe(rows: Iterable[Dict[str, float]]) -> pd.DataFrame:
	"""Convert metric rows into a sorted DataFrame for display and saving."""
	df = pd.DataFrame(rows)
	order = [
		"model",
		"accuracy",
		"precision_weighted",
		"recall_weighted",
		"f1_weighted",
	]
	cols = [c for c in order if c in df.columns] + [c for c in df.columns if c not in order]
	return df[cols].sort_values(by="f1_weighted", ascending=False).reset_index(drop=True)
