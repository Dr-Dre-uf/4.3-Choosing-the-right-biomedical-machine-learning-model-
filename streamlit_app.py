import streamlit as st
import pandas as pd
import numpy as np
import psutil
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# --- MONITORING UTILITY ---
def display_performance_monitor():
    """Tracks CPU and RAM usage for ADA compliance transparency."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    st.sidebar.caption("Real-time resource tracking for algorithmic transparency.")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical ML Demo", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Interactive Settings")
perspective = st.sidebar.radio(
    "Select Perspective:",
    ["Clinical Science", "Foundational Science"],
    help="Toggle terminology. Clinical uses patient vitals; Foundational uses biological metrics."
)

st.sidebar.title("Navigation")
activity = st.sidebar.radio(
    "Go to:",
    [
        "Activity 1 - Data Exploration", 
        "Activity 2 - Random Forest Training", 
        "Activity 3 - Performance Metrics",
        "Activity 4 - Strategic Evaluation"
    ],
    help="Navigate through the machine learning pipeline."
)

display_performance_monitor()

# --- DATA LOADING ---
@st.cache_data
def load_data(context):
    # Standard path as defined in your notebook
    data_path = "data/diabetes.csv"
    if not os.path.exists(data_path):
        data_path = "diabetes.csv" 
        
    df = pd.read_csv(data_path)
    
    if context == "Foundational Science":
        mapping = {
            'Glucose': 'Metabolite Alpha', 'BloodPressure': 'Hydrostatic Pressure',
            'SkinThickness': 'Epidermal Thickness', 'Insulin': 'Hormone Assay',
            'BMI': 'Mass Index', 'Age': 'Specimen Age'
        }
        df.rename(columns=mapping, inplace=True)
    return df

df = load_data(perspective)
feature_list = [c for c in df.columns if c != 'Outcome']

# --- ACTIVITY 1: DATA EXPLORATION ---
if activity == "Activity 1 - Data Exploration":
    st.header("Activity 1: Exploring Data Types")
    st.write("Inspect the raw data to understand feature distributions and identify class imbalances.")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Outcome Distribution**")
        counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(counts, color="#1f77b4")
        st.write(f"**Data Summary:** There are {counts[0]} survival cases and {counts[1]} death cases. This confirms a significant class imbalance.")
        
    with col2:
        st.markdown("**Feature Averages by Outcome**")
        # Defaulting to Glucose as per notebook scenario
        target_feat = 'Glucose' if perspective == 'Clinical Science' else 'Metabolite Alpha'
        means = df.groupby('Outcome')[target_feat].mean()
        st.bar_chart(means, color="#ff7f0e")
        st.write(f"**Data Summary:** The average {target_feat} for Survivors is {means[0]:.2f}, compared to {means[1]:.2f} for Deaths.")

# --- ACTIVITY 2: RANDOM FOREST TRAINING ---
elif activity == "Activity 2 - Random Forest Training":
    st.header("Activity 2: Random Forest Model Training")
    st.write("Configure and train a Random Forest ensemble model using 5-fold cross-validation.")

    # Parameters from Notebook 3
    st.sidebar.subheader("Random Forest Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 100, 1000, 500, step=100)
    max_features = st.sidebar.selectbox("Max Features", ["sqrt", "log2", None], index=0)
    
    

    if st.button("Execute 5-Fold Cross-Validation"):
        X = df[feature_list].values
        y = df['Outcome'].values
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_features=max_features,
            class_weight="balanced", 
            random_state=42
        )
        
        fold_results = []
        progress_bar = st.progress(0)
        
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            model.fit(X[train_idx], y[train_idx])
            acc = model.score(X[val_idx], y[val_idx])
            fold_results.append(acc)
            progress_bar.progress((i + 1) / 5)
            
        st.session_state['rf_cv_results'] = fold_results
        st.success("Cross-validation complete.")

    if 'rf_cv_results' in st.session_state:
        avg_acc = np.mean(st.session_state['rf_cv_results'])
        st.metric("Average Accuracy", f"{avg_acc:.4f}")
        
        res_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(5)],
            "Accuracy": st.session_state['rf_cv_results']
        }).set_index("Fold")
        
        st.line_chart(res_df)
        st.write(f"**Data Summary:** The line chart shows accuracy across 5 folds. The average accuracy is {avg_acc:.2%}.")

# --- ACTIVITY 3: PERFORMANCE METRICS ---
elif activity == "Activity 3 - Performance Metrics":
    st.header("Activity 3: Advanced Clinical Metrics")
    st.write("Evaluate Sensitivity, Specificity, and Precision to determine model reliability.")

    

    if st.button("Calculate Detailed Metrics"):
        X = df[feature_list].values
        y = df['Outcome'].values
        
        # Using a fixed split for the detailed metrics demonstration
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42).fit(X_s, y)
        y_pred = model.predict(X_s)
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        prec = tp / (tp + fp)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensitivity (Recall)", f"{sens:.3f}")
        c2.metric("Specificity", f"{spec:.3f}")
        c3.metric("Precision", f"{prec:.3f}")
        
        st.write(f"**Data Summary:** Sensitivity is {sens:.3f}, Specificity is {spec:.3f}, and Precision is {prec:.3f}.")

# --- ACTIVITY 4: STRATEGIC EVALUATION ---
elif activity == "Activity 4 - Strategic Evaluation":
    st.header("Activity 4: Strategic Comparison")
    st.write("Determine which model performs better for your specific biomedical context.")
    
    

    st.subheader("Random Forest vs. Decision Tree")
    st.write("Recall from your notebook: Random Forests correct for decision trees' habit of overfitting to their training set.")
    
    comparison = st.radio(
        "Which model is typically more stable across cross-validation folds?",
        ["Decision Tree", "Random Forest"]
    )
    
    if comparison == "Random Forest":
        st.success("Correct. Random Forests use ensemble learning to reduce variance and improve generalization.")
    else:
        st.info("Consider that single Decision Trees often 'memorize' noise in a single fold, leading to lower stability.")
