import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from io import StringIO, BytesIO
import base64  # For download

# Load model and artifacts
@st.cache_resource
def load_model():
    try:
        with open('final_model_confirmed.pkl', 'rb') as f:
            out = pickle.load(f)
        st.success("Model loaded successfully!")
        return out
    except FileNotFoundError:
        st.error("Model file 'final_model_confirmed.pkl' not found. Train and save it first.")
        return None

out = load_model()
if out is None:
    st.stop()

model = out['model']
le_target = out['label_encoder']
num_cols = out['num_cols']
cat_cols = out['cat_cols']
top_num = out['top_num']
num_pipeline = out['num_pipeline']

labels = le_target.classes_

st.title("🚀 Exoplanet Hunter AI")
st.markdown("Upload exoplanet data (CSV from NASA Archive) for AI classification: CONFIRMED, CANDIDATE, or FALSE POSITIVE.")

# Universal column mapping (handles Confirmed Planets/Kepler/K2 variations)
COLUMN_MAP = {
    'period': ['pl_orbper', 'koi_period', 'k2_period', 'period', 'per'],
    'prad': ['pl_rade', 'koi_prad', 'k2_prad', 'radius', 'prad'],
    'insol': ['pl_insol', 'koi_insol', 'k2_insol', 'insol'],
    'teq': ['pl_eqt', 'koi_teq', 'k2_teq', 'teq'],
    'steff': ['st_teff', 'koi_steff', 'k2_steff', 'teff'],
    'srad': ['st_rad', 'koi_srad', 'k2_srad', 'srad'],
    'slogg': ['st_logg', 'koi_slogg', 'k2_slogg', 'slogg'],
    'disposition': ['disposition', 'koi_disposition', 'k2_disp', 'disp']  # For eval mode
}

@st.cache_data
def preprocess_uploaded_csv(uploaded_file):
    """Universal CSV cleaner: Remap, normalize, impute, return X (features) and y (if labels)."""
    df = pd.read_csv(uploaded_file)
    st.write(f"Uploaded {len(df)} rows from: {uploaded_file.name}")
    
    # Auto-remap columns
    standard_cols = {}
    for std_col, aliases in COLUMN_MAP.items():
        for alias in aliases:
            if alias in df.columns:
                standard_cols[std_col] = alias
                break
        if std_col not in standard_cols:
            st.warning(f"Missing data for {std_col}. Expected one of: {aliases[:3]}...")  # Soft warning
    
    df_renamed = df.rename(columns=standard_cols)
    
    # Normalize (e.g., insol as proxy for depth if no direct)
    if 'insol' in df_renamed:
        df_renamed['insol'] = df_renamed['insol'].fillna(1.0)  # Default Earth-like
    
    # Select available core features (fallback to what's there)
    feature_cols = ['period', 'prad', 'insol', 'teq', 'steff', 'srad', 'slogg']
    available_features = [col for col in feature_cols if col in df_renamed]
    X = df_renamed[available_features].copy()
    
    # Impute missings (median for numerics)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    
    # Extract labels if present (for eval)
    y = None
    if 'disposition' in df_renamed:
        label_map = {label: i for i, label in enumerate(labels)}
        y = df_renamed['disposition'].map(label_map).fillna(-1)  # -1 for unknown
        y = y[y != -1] if y is not None else None  # Filter known for eval
        X = X.loc[y.index] if y is not None else X
    
    # Scale with pipeline (subset to available)
    available_num = [col for col in num_cols if col in X.columns]
    if available_num:
        X_scaled = num_pipeline.transform(X[available_num])
        X_proc = pd.DataFrame(X_scaled, columns=available_num, index=X.index)
    else:
        st.warning("No numeric features found—using raw data.")
        X_proc = X
    
    # Add interactions/aggregates (simplified for speed)
    if len(available_num) >= 2:
        for i in range(min(3, len(available_num))):  # Limit to top 3 for demo
            for j in range(i+1, min(3, len(available_num))):
                a, b = available_num[i], available_num[j]
                X_proc[f'{a}_x_{b}'] = X_proc[a] * X_proc[b]
    X_proc['row_mean'] = X_proc.mean(axis=1)
    
    return X_proc, y, df_renamed[available_features + (['disposition'] if 'disposition' in df_renamed else [])]

# UI: Sidebar for manual input
st.sidebar.header("Manual Entry")
period = st.sidebar.number_input("Orbital Period (days)", value=10.0, min_value=0.1)
prad = st.sidebar.number_input("Planet Radius (Earth radii)", value=2.0, min_value=0.1)
insol = st.sidebar.number_input("Insolation Flux (Earth flux)", value=1.0, min_value=0.0)
teq = st.sidebar.number_input("Equilibrium Temp (K)", value=300.0, min_value=0.0)

manual_data = pd.DataFrame({
    'period': [period], 'prad': [prad], 'insol': [insol], 'teq': [teq]
})

# Main UI: Upload
st.header("📁 Upload Dataset")
uploaded_file = st.file_uploader("Choose CSV file (NASA Exoplanet Archive)", type="csv")

if uploaded_file is not None:
    X_proc, y_true, df_display = preprocess_uploaded_csv(uploaded_file)
    
    if st.button("🔮 Predict Exoplanets"):
        # Predict
        preds = model.predict(X_proc)
        probs = model.predict_proba(X_proc)
        confidences = np.max(probs, axis=1)
        
        # Map back to labels
        pred_labels = [labels[p] for p in preds]
        
        # Results DF
        results = df_display.copy()
        results['prediction'] = pred_labels
        results['confidence'] = confidences
        
        st.subheader("📊 Predictions")
        st.dataframe(results.head(10))  # Preview
        
        # Download
        csv_buffer = StringIO()
        results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name="exoplanet_predictions.csv",
            mime="text/csv"
        )
        
        # Metrics (if labels present)
        if y_true is not None:
            st.subheader("📈 Model Performance")
            st.text(classification_report(y_true, preds, target_names=labels))
            
            # Confusion Matrix Plot
            cm = confusion_matrix(y_true, preds)
            fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale='Blues',
                               title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # Feature Importance (from model)
        if hasattr(model, 'named_estimators_') and 'rf' in model.named_estimators_:
            importances = model.named_estimators_['rf'].feature_importances_
            feat_names = list(X_proc.columns)
            top_feats = pd.Series(importances, index=feat_names).sort_values(ascending=False)[:10]
            fig_imp = px.bar(x=top_feats.values, y=top_feats.index, orientation='h',
                             title="Top 10 Feature Importances", color=top_feats.values)
            st.plotly_chart(fig_imp, use_container_width=True)

# Manual Prediction
if st.button("🔮 Predict Single Entry"):
    manual_proc = num_pipeline.transform(manual_data[num_pipeline.feature_names_in_ if hasattr(num_pipeline, 'feature_names_in_') else manual_data.columns])
    manual_pred = model.predict(manual_proc)[0]
    manual_prob = model.predict_proba(manual_proc).max()
    st.success(f"Prediction: **{labels[manual_pred]}** (Confidence: {manual_prob:.2%})")

st.markdown("---")
st.caption("Built for NASA Space Apps 2025 | Model: Confirmed Planets (94% Acc) | [GitHub](https://github.com/your-repo)")
