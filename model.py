import os
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Optional import — if available, these often give best performance
try:
    import xgboost as xgb
    has_xgb = True
except:
    has_xgb = False
try:
    import lightgbm as lgb
    has_lgb = True
except:
    has_lgb = False


print("Loaded:", df.shape)

# Target
TARGET = 'koi_disposition'
if TARGET not in df.columns:
    raise ValueError("target column not found")

# Drop id-like columns we don't want
drop_cols = ['kepid','kepoi_name','kepler_name','koi_tce_delivname']
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=[c])

# Quick target encoding
y = df[TARGET].fillna('UNKNOWN').astype(str)
le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)
print("Classes:", le_target.classes_)
print(y.value_counts())

# Features
X = df.drop(columns=[TARGET])

# Separate numeric and categorical
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
print("Numeric cols:", len(num_cols), "Categorical cols:", len(cat_cols))

# Drop extremely sparse columns (>60% missing)
miss_frac = X.isnull().mean()
drop_high_miss = miss_frac[miss_frac > 0.6].index.tolist()
print("Dropping columns with >60% missing:", len(drop_high_miss))
X = X.drop(columns=drop_high_miss)
num_cols = [c for c in num_cols if c not in drop_high_miss]
cat_cols = [c for c in cat_cols if c not in drop_high_miss]

# Simple numeric pipeline: median impute + scaling
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# For categories: simple fill and target-encoding (mean target per category) implemented below
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    # we will convert to target-encoded numeric features manually
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    # We'll treat cat columns separately because we use target encoding (so skip here)
], remainder='drop', sparse_threshold=0)

# Split dataset (hold out 15% test)
X_full = X.copy()
X_train_full, X_test, y_train_full, y_test = train_test_split(X_full, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
print("Train/test split:", X_train_full.shape, X_test.shape)

# --- FEATURE ENGINEERING ---
# 1) Numeric: impute & scale
X_train_num = pd.DataFrame(num_pipeline.fit_transform(X_train_full[num_cols]), columns=num_cols, index=X_train_full.index)
X_test_num = pd.DataFrame(num_pipeline.transform(X_test[num_cols]), columns=num_cols, index=X_test.index)

# 2) Categorical: target-encoding (mean target) computed on training data (map to test)
X_train_cat_te = pd.DataFrame(index=X_train_full.index)
X_test_cat_te = pd.DataFrame(index=X_test.index)
for c in cat_cols:
    mapping = pd.Series(y_train_full).groupby(X_train_full[c].fillna('missing')).mean()
    X_train_cat_te[c + '_te'] = X_train_full[c].fillna('missing').map(mapping).fillna(np.mean(y_train_full))
    X_test_cat_te[c + '_te'] = X_test[c].fillna('missing').map(mapping).fillna(np.mean(y_train_full))

# 3) Simple interaction features among top numeric features
from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X_train_num.fillna(0), y_train_full, discrete_features=False, random_state=42)
mi_ser = pd.Series(mi, index=num_cols).sort_values(ascending=False)
top_num = mi_ser.index[:8].tolist()  # choose top 8 for interactions
X_train_inter = pd.DataFrame(index=X_train_full.index)
X_test_inter = pd.DataFrame(index=X_test.index)
for i in range(len(top_num)):
    for j in range(i+1, len(top_num)):
        a, b = top_num[i], top_num[j]
        X_train_inter[f'{a}_x_{b}'] = X_train_num[a] * X_train_num[b]
        X_test_inter[f'{a}_x_{b}'] = X_test_num[a] * X_test_num[b]

# 4) Row-level aggregates
X_train_num['row_mean'] = X_train_num.mean(axis=1)
X_train_num['row_std'] = X_train_num.std(axis=1)
X_test_num['row_mean'] = X_test_num.mean(axis=1)
X_test_num['row_std'] = X_test_num.std(axis=1)

# Compose final feature sets
X_train_proc = pd.concat([X_train_num.reset_index(drop=True), X_train_inter.reset_index(drop=True), X_train_cat_te.reset_index(drop=True)], axis=1)
X_test_proc = pd.concat([X_test_num.reset_index(drop=True), X_test_inter.reset_index(drop=True), X_test_cat_te.reset_index(drop=True)], axis=1)
print("Processed shapes:", X_train_proc.shape, X_test_proc.shape)

# --- MODELING: try several strong learners, then stack ---
estimators = []
# 1) HistGradientBoosting (fast, strong)
hgb = HistGradientBoostingClassifier(random_state=42)
estimators.append(('hgb', hgb))

# 2) RandomForest
rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs= -1)
estimators.append(('rf', rf))

# 3) XGBoost or LightGBM if present (use them as extra strong learners)
if has_xgb:
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=1)
    estimators.append(('xgb', xgb_clf))
if has_lgb:
    lgb_clf = lgb.LGBMClassifier(random_state=42, n_jobs=1)
    estimators.append(('lgb', lgb_clf))

# Stacking meta-learner
stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=2000), n_jobs=-1, passthrough=True)

# Quick randomized search on the stack (light) — adjust n_iter for more thorough search
param_dist = {
    # tune just a little to speed things up; expand for better performance
    'hgb__max_iter': [100, 200],
    'rf__n_estimators': [80, 150],
}
from scipy.stats import randint
rs = RandomizedSearchCV(stack, param_distributions=param_dist, n_iter=6, cv=3, scoring='accuracy', random_state=42, n_jobs=-1, verbose=1)
rs.fit(X_train_proc, y_train_full)
print("Best CV score (stack):", rs.best_score_)
best_model = rs.best_estimator_

# Evaluate on test set
y_pred = best_model.predict(X_test_proc)
test_acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_acc)
print("Classification report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and preprocessing artifacts
out = {'model': best_model, 'label_encoder': le_target, 'num_cols': num_cols, 'cat_cols': cat_cols, 'top_num': top_num}

with open("final_model.pkl", "wb") as f:
    pickle.dump(out, f)
print("Saved final model to final_model.pkl")
