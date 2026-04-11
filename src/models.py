from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier


def build_supervised_baseline(random_state: int = 42) -> RandomForestClassifier:
	"""Strong baseline using only labeled data."""
	return RandomForestClassifier(
		n_estimators=300,
		random_state=random_state,
		n_jobs=-1,
		class_weight="balanced_subsample",
	)


def build_self_training_model(random_state: int = 42) -> SelfTrainingClassifier:
	"""Self-training over a probabilistic classifier."""
	base = LogisticRegression(
		max_iter=1000,
		random_state=random_state,
		class_weight="balanced",
		solver="lbfgs",
	)
	return SelfTrainingClassifier(
		estimator=base,
		threshold=0.85,
		criterion="threshold",
		max_iter=10,
		verbose=False,
	)


def build_label_spreading_model() -> LabelSpreading:
	"""Graph-based SSL model that propagates labels to unlabeled samples."""
	return LabelSpreading(kernel="rbf", gamma=0.5, alpha=0.2, max_iter=50)
