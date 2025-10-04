# save_as: plot_koi_top10.py
# Requirements: pandas, scikit-learn, matplotlib, numpy
# Run in Colab or local Python. If in Colab, upload your CSV to the notebook or mount Drive.

import os, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Path to your uploaded file
FILE = "/mnt/data/cumulative_2025.10.03_23.29.15.csv"  # adjust if different

# 1) Load while skipping commented header lines
df = pd.read_csv(FILE, comment='#', low_memory=False)
print("Loaded:", df.shape)

# 2) Option B: exclude koi_pdisposition and ID-like columns
TARGET = 'koi_disposition'
drop_cols = ['kepid','kepoi_name','kepler_name','koi_tce_delivname','koi_pdisposition']
for c in drop_cols:
    if c in df.columns:
        df = df.drop(columns=[c])

# 3) Keep numeric columns only (coerce to numeric to be robust)
X_all = df.drop(columns=[TARGET])
X_numeric = X_all.apply(pd.to_numeric, errors='coerce')
valid_numeric_cols = [c for c in X_numeric.columns if not X_numeric[c].isna().all()]
X = X_numeric[valid_numeric_cols].copy()
y = df[TARGET].fillna('UNKNOWN').astype(str)

# 4) Clean rows where target is missing (if any)
mask = y.notnull()
X = X[mask]
y = y[mask]

# 5) Impute numeric missing values
imp = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)

# 6) Encode the target
le = LabelEncoder()
y_enc = le.fit_transform(y)
print("Classes:", list(le.classes_))

# 7) Quick train/test split
X_train, X_test, y_train, y_test = train_test_split(X_imp, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
print("Train/test shapes:", X_train.shape, X_test.shape)

# 8) Train a fast gradient-boosting tree (small max_iter to be quick)
model = HistGradientBoostingClassifier(max_iter=100, random_state=42)
model.fit(X_train, y_train)

# 9) Evaluate on holdout
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 10) Feature importances and plot top 10
feat_imp = pd.Series(model.feature_importances_, index=X_imp.columns).sort_values(ascending=False)
top10 = feat_imp.head(10)
print("Top 10 features:\n", top10)

plt.figure(figsize=(8,5))
top10[::-1].plot(kind='barh')   # highest at top
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances — Numeric features only (no koi_pdisposition)")
plt.tight_layout()

OUTPATH = "/mnt/data/top10_feature_importance.png"
plt.savefig(OUTPATH)
plt.show()

# 11) Save model artifacts (optional)
with open("/mnt/data/hgb_numeric_model.pkl","wb") as f:
    pickle.dump({'model':model,'imputer':imp,'le':le,'numeric_cols':valid_numeric_cols}, f)

print("Saved plot to:", OUTPATH)
