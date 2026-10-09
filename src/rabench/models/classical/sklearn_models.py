"""
sklearn_models.py

Thin BaseRegressor wrappers around the four classical regressors named in
the project's objectives: Random Forest, XGBoost, CatBoost, Decision Tree.

All four operate on the FLAT dataset view (see rabench.data.datasets):
one row = one (frequency, geometry) point, in -> out is plain tabular
regression. This is the same task formulation used by the classical-ML
baselines in your literature review (Jain et al., Rai et al.) that the
project aims to compare against the paper's deep models.

Multi-output handling
----------------------
Antenna I has 6 target channels. RandomForestRegressor and
DecisionTreeRegressor support multi-output natively. XGBoost and CatBoost
do not (in their sklearn-API single-estimator form), so those two are
automatically wrapped in `sklearn.multioutput.MultiOutputRegressor`,
which trains one independent model per output channel under the hood.
This wrapping is transparent to the rest of the pipeline -- `fit`/
`predict` behave identically regardless of output dimensionality.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeRegressor

from rabench.models.base import BaseRegressor

#: Cap on parallel workers for tree-ensemble training. `n_jobs=-1` builds
#: as many trees concurrently as there are CPU cores, each holding its own
#: copy of the training data + growing tree in memory at once -- on a
#: many-core / memory-constrained machine that blows past available RAM
#: well before any single tree is large. Capping this only changes how
#: many trees are built *at the same time*; the resulting trees (same
#: n_estimators/max_depth/random_state) and thus model quality are
#: identical, just slower to train.
_MAX_PARALLEL_JOBS = 4

#: MultiOutputRegressor(n_jobs>1) fits each output channel in its own OS
#: process (joblib's loky backend) and pickles the fitted booster back
#: across the process boundary. For XGBoost/CatBoost that cross-process
#: deserialization of the C++ booster is flaky on Windows and can crash
#: with `bad allocation` / BrokenProcessPool under memory pressure. Keep
#: the per-channel fit in-process (sequential) and let XGBoost/CatBoost's
#: own internal multithreading (`_MAX_PARALLEL_JOBS`) provide the speedup
#: instead -- same effective parallelism, no cross-process pickling.
_MULTI_OUTPUT_JOBS = 1


def _is_multi_output(y: np.ndarray) -> bool:
    return y.ndim > 1 and y.shape[1] > 1


class RandomForestModel(BaseRegressor):
    name = "RandomForest"
    family = "classical"

    def __init__(self, n_estimators: int = 200, max_depth: Optional[int] = None, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=_MAX_PARALLEL_JOBS
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class DecisionTreeModel(BaseRegressor):
    name = "DecisionTree"
    family = "classical"

    def __init__(self, max_depth: Optional[int] = None, random_state: int = 42):
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel(BaseRegressor):
    name = "XGBoost"
    family = "classical"

    def __init__(self, n_estimators: int = 300, max_depth: int = 6, learning_rate: float = 0.05, random_state: int = 42):
        self._base_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            n_jobs=_MAX_PARALLEL_JOBS,
            objective="reg:squarederror",
        )
        self.model = None  # built at fit() time once we know output dim

    def fit(self, X_train, y_train):
        base = xgb.XGBRegressor(**self._base_kwargs)
        if _is_multi_output(y_train):
            self.model = MultiOutputRegressor(base, n_jobs=_MULTI_OUTPUT_JOBS)
        else:
            self.model = base
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class CatBoostModel(BaseRegressor):
    name = "CatBoost"
    family = "classical"

    def __init__(self, iterations: int = 300, depth: int = 6, learning_rate: float = 0.05, random_state: int = 42):
        self._base_kwargs = dict(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=_MAX_PARALLEL_JOBS,
        )
        self.model = None

    def fit(self, X_train, y_train):
        base = CatBoostRegressor(**self._base_kwargs)
        if _is_multi_output(y_train):
            self.model = MultiOutputRegressor(base, n_jobs=_MULTI_OUTPUT_JOBS)
        else:
            self.model = base
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


#: Registry used by the benchmarking runner / configs to instantiate
#: classical models by name.
CLASSICAL_MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "decision_tree": DecisionTreeModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
}
