import streamlit as st
import pandas as pd
import numpy as np
import psutil
import os
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
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%", help="Real-time CPU usage.")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB", help="Memory allocated to this app.")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical ML Analytics", layout="wide")

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Module Settings")
perspective = st.sidebar.radio(
    "Select Perspective:",
    ["Clinical Science", "Foundational Science"],
    help="Toggle terminology. Clinical uses patient vitals; Foundational uses biological metrics."
)

st.sidebar.title("Navigation")
activity = st.sidebar.radio(
    "Go to:",
    ["Activity 1 - Data Exploration", "Activity 2 - Cross-Validation Analysis", "Activity 3 - Model Comparison"],
    help="Navigate through the module activities."
)

display_performance_monitor()

# --- DATA LOADING ---
@st.cache_data
def load_data(context):
    # Path aligned with notebook: data/diabetes.csv
    data_path = "data/diabetes.csv"
    if not os.path.exists(data_path):
        data_path = "diabetes.csv" # Fallback
        
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
X = df.drop(columns=['Outcome']).values
y = df['Outcome'].values

# --- ACTIVITY 1: DATA EXPLORATION ---
if activity == "Activity 1 - Data Exploration":
    st.header("Activity 1: Exploring Data Types")
    st.write("Before training, inspect the dataset to understand feature distributions and class imbalance.")
    
    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Outcome Distribution**")
        counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(counts, color="#1f77b4")
        st.write(f"**Data Summary:** The dataset contains {counts[0]} survival cases and {counts[1]} death cases, indicating an imbalanced distribution.")
        
    with col2:
        st.markdown("**Average Metric per Outcome**")
        target_col = df.columns[1] # Use Glucose/Metabolite as example
        means = df.groupby('Outcome')[target_col].mean()
        st.bar_chart(means, color="#ff7f0e")
        st.write(f"**Data Summary:** Average {target_col} for Survivors is {means[0]:.2f}, compared to {means[1]:.2f} for Deaths.")

# --- ACTIVITY 2: CROSS-VALIDATION ---
elif activity == "Activity 2 - Cross-Validation Analysis":
    st.header("Activity 2: 5-Fold Cross-Validation")
    st.write("This mirrors Notebook 3 by evaluating a Random Forest model across 5 data splits.")

    if st.button("Run 5-Fold Evaluation"):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42)
        
        metrics = []
        for train_idx, val_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_va = scaler.transform(X[val_idx])
            
            model.fit(X_tr, y[train_idx])
            y_pred = model.predict(X_va)
            
            tn, fp, fn, tp = confusion_matrix(y[val_idx], y_pred).ravel()
            metrics.append([
                (tp+tn)/(tp+tn+fp+fn), # Acc
                tp/(tp+fn),           # Sens
                tn/(tn+fp),           # Spec
                tp/(tp+fp)            # Prec
            ])
            
        res_df = pd.DataFrame(metrics, columns=['Accuracy', 'Sensitivity', 'Specificity', 'Precision'])
        st.session_state['cv_results'] = res_df

    if 'cv_results' in st.session_state:
        st.table(st.session_state['cv_results'])
        st.line_chart(st.session_state['cv_results'])
        st.write("**Data Summary:** The chart shows performance stability across folds. Average Accuracy: {:.3f}".format(st.session_state['cv_results']['Accuracy'].mean()))

# --- ACTIVITY 3: MODEL COMPARISON ---
elif activity == "Activity 3 - Model Comparison":
    st.header("Activity 3: Selecting the Right Model")
    st.write("Based on your evaluation, compare the strategic trade-offs between models.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Tree")
        st.write("- Logic: Interpretable 'If-Then' splits.")
        st.write("- Reliability: Prone to high variance across folds.")
    with col2:
        st.subheader("Random Forest")
        st.write("- Logic: Ensemble of multiple trees.")
        st.write("- Reliability: Corrects for overfitting; usually more stable.")

    st.info("Question: Which performs better, Random Forest or Decision Tree? Why?")
    st.write("Answer: As seen in the cross-validation metrics, Random Forests typically provide higher sensitivity and more stable accuracy because they aggregate results from many trees, reducing the impact of outliers in the data.")
