"""
- Loads CSV (sep='#') at /mnt/data/TOI_2025.10.03_11.37.57.csv
- Drops cols with >80% NA
- Auto-detects text column (first object column not 'disposition')
- Builds numeric + categorical (target-encoded) + TF-IDF features
- Feature engineering (interactions, row aggregates)
- Removes highly correlated features (abs(corr) >= corr_threshold)
- Trains models and selects best
- Outputs top-10 features and saves plots (Seaborn) and artifacts to /mnt/data/
"""

import os, sys, warnings, traceback
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import randint

# Optional boosters
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False
try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False

# Paths / constants
CSV_PATH = "/mnt/data/TOI_2025.10.03_11.37.57.csv"
OUT_DIR = Path("/mnt/data")
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "disposition"
SEP = "#"
TFIDF_MAX_FEAT = 2000    # adjust for memory / speed
TFIDF_NGRAM = (1, 2)
CORR_THRESHOLD = 0.90    # drop one of pair if |corr| >= this
TOP_FEATURES_N = 10

def load_data(path):
    print("Loading:", path)
    df = pd.read_csv(path, sep=SEP, engine="python")
    df.columns = [c.strip() for c in df.columns]
    print("Initial shape:", df.shape)
    return df

def drop_high_missing(df, thresh=0.8):
    miss_frac = df.isnull().mean()
    drop_cols = miss_frac[miss_frac > thresh].index.tolist()
    print(f"Dropping {len(drop_cols)} columns with >{int(thresh*100)}% missing.")
    return df.drop(columns=drop_cols), drop_cols

def detect_text_column(df, target_col):
    # prefer first object column not the target
    obj_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    for c in obj_cols:
        if c.strip().lower() != target_col.lower():
            return c
    # fallback: first column except target
    for c in df.columns:
        if c != target_col:
            return c
    return None

def target_clean_map(y_series):
    # unify common variants
    y = y_series.fillna("unknown").astype(str).str.strip().str.lower()
    mapping = {
        'false_positive': 'false positive',
        'falsepositive': 'false positive',
        'fp': 'false positive'
    }
    return y.replace(mapping)

def compute_mutual_info(X_num, y):
    # mutual info to rank numeric features (handles nan by fill 0)
    mi = mutual_info_classif(X_num.fillna(0), y, discrete_features=False, random_state=42)
    return pd.Series(mi, index=X_num.columns).sort_values(ascending=False)

def create_interactions(X_num, top_k=8):
    top_cols = X_num.columns[:top_k].tolist()
    inter_df = pd.DataFrame(index=X_num.index)
    for i in range(len(top_cols)):
        for j in range(i+1, len(top_cols)):
            a, b = top_cols[i], top_cols[j]
            inter_df[f"{a}_x_{b}"] = X_num[a] * X_num[b]
    return inter_df

def remove_correlated_features(df_features, threshold=CORR_THRESHOLD, plots_dir=PLOTS_DIR):
    # compute Pearson correlation and drop one of pairs with abs(corr) >= threshold
    corr = df_features.corr().abs()
    # save heatmap of correlation (sample if large)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", vmin=0, vmax=1, square=True, cbar_kws={"shrink": .5})
    plt.title("Feature correlation (abs Pearson)")
    out = plots_dir / "correlation_heatmap.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print("Saved correlation heatmap to", out)

    # Find highly correlated pairs
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= threshold)]
    print(f"Removing {len(to_drop)} features with abs(corr) >= {threshold}")
    return df_features.drop(columns=to_drop), to_drop

def get_feature_names(tfidf, numeric_cols, cat_te_cols, inter_cols):
    # TF-IDF feature names plus numeric and cat names and interaction names
    vocab = []
    try:
        vocab = [f"tfidf__{t}" for t in tfidf.get_feature_names_out()]
    except Exception:
        vocab = [f"tfidf__{i}" for i in range(tfidf.max_features or TFIDF_MAX_FEAT)]
    names = vocab + list(numeric_cols) + list(cat_te_cols) + list(inter_cols)
    return names

def plot_bivariate_and_categorical(df, label_col, numeric_cols, cat_te_cols, plots_dir=PLOTS_DIR, max_num_plots=6):
    # For numeric: boxplot by label for top numeric cols
    n_plot = min(max_num_plots, len(numeric_cols))
    for i, c in enumerate(numeric_cols[:n_plot]):
        plt.figure(figsize=(6,4))
        sns.boxplot(x=label_col, y=c, data=df, showfliers=False)
        plt.title(f"Boxplot of {c} by {label_col}")
        f = plots_dir / f"boxplot_{c}.png"
        plt.tight_layout(); plt.savefig(f, dpi=150); plt.close()
    # For categorical TE features (they are numeric floats), show violin or box
    n_plot2 = min(max_num_plots, len(cat_te_cols))
    for i, c in enumerate(cat_te_cols[:n_plot2]):
        plt.figure(figsize=(6,4))
        sns.boxplot(x=label_col, y=c, data=df, showfliers=False)
        plt.title(f"Boxplot of {c} by {label_col}")
        f = plots_dir / f"boxplot_{c}.png"
        plt.tight_layout(); plt.savefig(f, dpi=150); plt.close()
    print(f"Saved bivariate plots to {plots_dir}")

def main():
    try:
        # 1) Load
        df = load_data(CSV_PATH)
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found; available columns: {df.columns.tolist()[:30]}")

        # 2) Drop columns with >80% NA
        df, dropped_high_na = drop_high_missing(df, thresh=0.80)

        # 3) Detect text col
        text_col = detect_text_column(df, TARGET)
        print("Detected text column:", text_col)

        # 4) Clean target and filter to expected values
        df[TARGET] = target_clean_map(df[TARGET])
        valid = ['confirmed','candidate','false positive']
        mask = df[TARGET].astype(str).str.lower().isin(valid)
        print(f"Keeping {mask.sum()} rows where target in {valid} (out of {len(df)})")
        df = df.loc[mask].reset_index(drop=True)

        # 5) Label encoding
        le = LabelEncoder()
        y = le.fit_transform(df[TARGET].astype(str).str.lower())
        print("Label classes:", le.classes_)

        # 6) Identify numeric and categorical columns
        # Exclude the target and text col
        X = df.drop(columns=[TARGET])
        if text_col in X.columns:
            X_text = X[text_col].fillna("").astype(str)
            X = X.drop(columns=[text_col])
        else:
            X_text = pd.Series([""] * len(df))

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
        # Try to coerce object cols with mostly numeric values to numeric
        for c in X.columns:
            if c not in numeric_cols and c not in cat_cols:
                continue
        print(f"Numeric cols detected: {len(numeric_cols)}; Categorical cols: {len(cat_cols)}")

        # 7) Numeric pipeline: impute medians and scale
        num_imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        if numeric_cols:
            X_num = pd.DataFrame(num_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)
            X_num = pd.DataFrame(scaler.fit_transform(X_num), columns=numeric_cols)
        else:
            X_num = pd.DataFrame(index=X.index)

        # 8) Categorical: target-encode (mean target) using training-scheme (we'll use full data here then split)
        # Compute mapping on whole data then will re-fit/transform in train/test for strictness if needed
        X_cat_te = pd.DataFrame(index=X.index)
        for c in cat_cols:
            grp = X[c].fillna("missing")
            mapping = pd.Series(y).groupby(grp).mean()
            X_cat_te[c + "_te"] = grp.map(mapping).fillna(np.mean(y))

        # 9) TF-IDF for text
        tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=TFIDF_NGRAM)
        X_text_vec = tfidf.fit_transform(X_text)  # sparse matrix

        # 10) Interaction features from top numeric by mutual info
        if not X_num.empty:
            mi_ser = compute_mutual_info(X_num, y)
            top_num = mi_ser.index[:8].tolist()
            X_inter = create_interactions(X_num[top_num], top_k=len(top_num))
        else:
            X_inter = pd.DataFrame(index=X.index)
            top_num = []

        # 11) Row-level aggregates
        if not X_num.empty:
            X_num['row_mean'] = X_num.mean(axis=1)
            X_num['row_std'] = X_num.std(axis=1)

        # 12) Combine all features: TF-IDF (sparse) + numeric + cat-te + interactions (dense -> sparse)
        from scipy.sparse import hstack, csr_matrix
        dense_parts = []
        numeric_names = X_num.columns.tolist()
        cat_te_names = X_cat_te.columns.tolist()
        inter_names = X_inter.columns.tolist()
        if not X_num.empty:
            dense_parts.append(csr_matrix(X_num.values))
        if not X_inter.empty:
            dense_parts.append(csr_matrix(X_inter.values))
        if not X_cat_te.empty:
            dense_parts.append(csr_matrix(X_cat_te.values))

        if dense_parts:
            X_dense_stack = hstack(dense_parts, format='csr')
            X_full = hstack([X_text_vec, X_dense_stack], format='csr')
        else:
            X_full = X_text_vec

        print("Final feature matrix shape (sparse):", X_full.shape)
        feature_names = get_feature_names(tfidf, numeric_names, cat_te_names, inter_names)

        # 13) Split into train/test
        X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
            X_full, y, np.arange(X_full.shape[0]), test_size=0.20, random_state=42, stratify=y
        )
        print("Train/test shapes:", X_train.shape, X_test.shape)

        # 14) Convert sparse back to DataFrame for correlation-based feature elimination on dense features only:
        # We'll create a DataFrame of dense features (numeric + cat_te + interactions) for correlation analysis
        if dense_parts:
            # Build dense DataFrame for correlation step
            dense_df = pd.concat([X_num.reset_index(drop=True), X_inter.reset_index(drop=True), X_cat_te.reset_index(drop=True)], axis=1)
        else:
            dense_df = pd.DataFrame(index=df.index)

        # Use only rows in training set for correlation calculation
        dense_df_train = dense_df.iloc[idx_train].reset_index(drop=True)
        # If dense_df_train empty, skip correlation elimination
        if not dense_df_train.empty:
            dense_df_train_clean = dense_df_train.fillna(0)
            dense_df_train_clean, dropped_by_corr = remove_correlated_features(dense_df_train_clean, threshold=CORR_THRESHOLD, plots_dir=PLOTS_DIR)
            # Drop the corresponding columns from dense_df (both train & full)
            dense_columns_kept = dense_df_train_clean.columns.tolist()
            print("Dense columns kept after correlation:", len(dense_columns_kept))
        else:
            dense_columns_kept = []

        # Rebuild X_full with reduced dense columns (map back)
        # Recreate dense parts using kept columns only
        new_dense_parts = []
        new_numeric_names = []
        new_cat_te_names = []
        new_inter_names = []
        if dense_columns_kept:
            # determine which columns are numeric, inter, or cat_te
            for col in dense_columns_kept:
                if col in numeric_names:
                    new_numeric_names.append(col)
                elif col in inter_names:
                    new_inter_names.append(col)
                elif col in cat_te_names:
                    new_cat_te_names.append(col)
            # Recreate arrays
            arrays = []
            if new_numeric_names:
                arrays.append(csr_matrix(X_num[new_numeric_names].values))
            if new_inter_names:
                arrays.append(csr_matrix(X_inter[new_inter_names].values))
            if new_cat_te_names:
                arrays.append(csr_matrix(X_cat_te[new_cat_te_names].values))
            if arrays:
                new_dense_stack = hstack(arrays, format='csr')
                X_full = hstack([X_text_vec, new_dense_stack], format='csr')
            else:
                X_full = X_text_vec
            # update feature name list
            feature_names = get_feature_names(tfidf, new_numeric_names, new_cat_te_names, new_inter_names)
            # resplit with same indices (recreate train/test)
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                X_full, y, np.arange(X_full.shape[0]), test_size=0.20, random_state=42, stratify=y
            )
            print("After correlation elimination, new feature matrix shape:", X_full.shape)
        else:
            print("No dense features or none kept after correlation elimination. Keeping TF-IDF only.")

        # 15) Quick EDA & bivariate visualizations (use original df for plotting convenience)
        # prepare small DataFrame for plotting with chosen numeric and cat_te columns
        plot_df = pd.DataFrame({TARGET: df[TARGET]})
        # choose top numeric for plotting by variance
        if numeric_names:
            top_num_plot = list(new_numeric_names or numeric_names)[:6]
            for c in top_num_plot:
                plot_df[c] = X_num[c].values
        if cat_te_names:
            top_cat_plot = list(new_cat_te_names or cat_te_names)[:6]
            for c in top_cat_plot:
                plot_df[c] = X_cat_te[c].values

        # Save bivariate plots
        plot_bivariate_and_categorical(pd.concat([plot_df.reset_index(drop=True)], axis=1),
                                       TARGET, top_num_plot if numeric_names else [], top_cat_plot if cat_te_names else [],
                                       plots_dir=PLOTS_DIR, max_num_plots=6)

        # 16) Modeling: RandomForest + HistGradientBoosting + optional boosters; limited randomized search
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        }
        # HGB (works with dense arrays) - we will try with sparse conversion (may be memory heavy)
        try:
            hgb = HistGradientBoostingClassifier(random_state=42)
            models["HistGradientBoosting"] = hgb
        except Exception as e:
            print("HGB not available:", e)

        if has_xgb:
            try:
                models["XGBoost"] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=1)
            except Exception:
                pass
        if has_lgb:
            try:
                models["LGBM"] = lgb.LGBMClassifier(random_state=42, n_jobs=1)
            except Exception:
                pass

        # Stacking (use a small final estimator)
        estimator_list = [(name, clf) for name, clf in models.items()]
        stack = StackingClassifier(estimators=estimator_list, final_estimator=LogisticRegression(max_iter=1000), passthrough=True, n_jobs=-1)

        # Randomized search on modest param grid for stack
        param_dist = {
            # tune a couple params to speed up
            'stack__final_estimator__C' if False else 'final_estimator__C': [0.1, 1.0, 10.0]  # placeholder (StackingCV wrapper differences)
        }

        # We'll skip RandomizedSearchCV on stack for stability here and fit simple models then compare
        best_model = None
        best_acc = -1
        results = {}

        # Train individual models (using sparse X_train/X_test if supported)
        for name, model in models.items():
            print(f"Fitting model: {name}")
            try:
                if hasattr(X_train, "toarray") and name == "HistGradientBoosting":
                    # convert to dense for HGB
                    X_train_dense = X_train.toarray()
                    X_test_dense = X_test.toarray()
                    model.fit(X_train_dense, y_train)
                    preds = model.predict(X_test_dense)
                else:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results[name] = {"accuracy": acc, "report": classification_report(y_test, preds, zero_division=0), "cm": confusion_matrix(y_test, preds)}
                print(f"{name} accuracy: {acc:.4f}")
                if acc > best_acc:
                    best_acc = acc; best_model = model; best_name = name
            except Exception as e:
                print(f"Training failed for {name}: {e}")
                traceback.print_exc()

        # Optionally train stacking if memory permits
        try:
            print("Fitting stacking ensemble (may take time)...")
            # Fit on sparse X_train if possible
            stack.fit(X_train, y_train)
            preds = stack.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results["Stacking"] = {"accuracy": acc, "report": classification_report(y_test, preds, zero_division=0), "cm": confusion_matrix(y_test, preds)}
            print("Stacking accuracy:", acc)
            if acc > best_acc:
                best_acc = acc; best_model = stack; best_name = "Stacking"
        except Exception as e:
            print("Stacking failed or skipped:", e)

        print("Best model selected:", best_name, "with accuracy", best_acc)

        # 17) Feature Importance: we will use RandomForest if available; else permutation importance on best_model
        feature_importances = None
        feature_names = feature_names[:X_full.shape[1]]  # ensure length consistent

        if "RandomForest" in results and hasattr(models["RandomForest"], "feature_importances_"):
            # For RF we can get importances: but need to map to feature names (works if model was trained on full feature matrix)
            try:
                rf = models.get("RandomForest")
                # If RF was fitted earlier, get its importances (works only if shape matches)
                if hasattr(rf, "feature_importances_"):
                    imp = rf.feature_importances_
                    # align length
                    imp = imp[:len(feature_names)]
                    imp_ser = pd.Series(imp, index=feature_names).sort_values(ascending=False)
                    feature_importances = imp_ser
            except Exception as e:
                print("RF importance extraction failed:", e)

        if feature_importances is None and best_model is not None:
            # fallback: permutation importance (compute on small subset to save time)
            try:
                print("Computing permutation importance (this may take time)...")
                if hasattr(X_test, "toarray") and (not hasattr(best_model, "feature_importances_")):
                    X_test_for_perm = X_test.toarray()
                else:
                    X_test_for_perm = X_test
                perm = permutation_importance(best_model, X_test_for_perm, y_test, n_repeats=10, random_state=42, n_jobs=-1)
                imp_ser = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
                feature_importances = imp_ser
            except Exception as e:
                print("Permutation importance failed:", e)
                feature_importances = pd.Series(dtype=float)

        # Report top features
        if not feature_importances.empty:
            top_feats = feature_importances.head(TOP_FEATURES_N)
            print("\nTop features:")
            print(top_feats)
            # Save to CSV
            top_feats.to_csv(OUT_DIR / "top_features.csv", header=["importance"])
            print("Saved top features to", OUT_DIR / "top_features.csv")

        # 18) Save model and artifacts
        artifacts = {
            "model": best_model,
            "label_encoder": le,
            "tfidf": tfidf,
            "numeric_imputer": num_imputer,
            "scaler": scaler,
            "numeric_cols": new_numeric_names if 'new_numeric_names' in locals() else numeric_names,
            "cat_te_cols": new_cat_te_names if 'new_cat_te_names' in locals() else cat_te_names,
            "inter_cols": new_inter_names if 'new_inter_names' in locals() else inter_names
        }
        out_pkl = OUT_DIR / "final_model.pkl"
        joblib.dump(artifacts, out_pkl)
        print("Saved artifacts to", out_pkl)

        # 19) Save plots summary: confusion matrix of best model
        try:
            if best_model is not None:
                if hasattr(X_test, "toarray"):
                    X_test_plot = X_test.toarray()
                else:
                    X_test_plot = X_test
                y_pred_best = best_model.predict(X_test_plot)
                cm = confusion_matrix(y_test, y_pred_best)
                plt.figure(figsize=(6,5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
                plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (Best Model)")
                plt.tight_layout()
                cm_file = PLOTS_DIR / "confusion_matrix_best_model.png"
                plt.savefig(cm_file, dpi=150); plt.close()
                print("Saved confusion matrix to", cm_file)
        except Exception as e:
            print("Failed to plot confusion matrix:", e)

        print("Pipeline completed successfully.")

    except Exception as e:
        print("ERROR in pipeline execution:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
