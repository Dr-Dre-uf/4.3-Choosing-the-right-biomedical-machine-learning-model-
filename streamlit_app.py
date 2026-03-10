import streamlit as st
import pandas as pd
import numpy as np
import psutil
import os
import matplotlib.pyplot as plt
# Added the missing train_test_split import here
from sklearn.model_selection import KFold, train_test_split 
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
        "Activity 3 - Tree vs. Forest Comparison"
    ],
    help="Navigate through the machine learning pipeline."
)

display_performance_monitor()

# --- DATA LOADING ---
@st.cache_data
def load_data(context):
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
        st.write(f"**Data Summary:** The dataset contains {counts[0]} survival cases and {counts[1]} death cases.")
        
    with col2:
        st.markdown("**Feature Averages by Outcome**")
        target_feat = 'Glucose' if perspective == 'Clinical Science' else 'Metabolite Alpha'
        means = df.groupby('Outcome')[target_feat].mean()
        st.bar_chart(means, color="#ff7f0e")
        st.write(f"**Data Summary:** Average {target_feat} for Survivors: {means[0]:.2f}, Deaths: {means[1]:.2f}.")

# --- ACTIVITY 2: RANDOM FOREST TRAINING ---
elif activity == "Activity 2 - Random Forest Training":
    st.header("Activity 2: Random Forest Model Training")
    st.write("Train a Random Forest ensemble model using 5-fold cross-validation and observe performance stability.")

    st.sidebar.subheader("RF Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Trees", 100, 1000, 500, step=100)
    
    if st.button("Execute 5-Fold Evaluation"):
        X = df[feature_list].values
        y = df['Outcome'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", random_state=42)
        
        metrics = []
        for train_idx, val_idx in kf.split(X):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            tn, fp, fn, tp = confusion_matrix(y[val_idx], y_pred).ravel()
            metrics.append([(tp+tn)/(tp+tn+fp+fn), tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
            
        st.session_state['cv_res'] = pd.DataFrame(metrics, columns=['Accuracy', 'Sensitivity', 'Specificity', 'Precision'])

    if 'cv_res' in st.session_state:
        st.table(st.session_state['cv_res'])
        st.line_chart(st.session_state['cv_res']['Accuracy'])
        st.write(f"**Data Summary:** Average Cross-Validation Accuracy: {st.session_state['cv_res']['Accuracy'].mean():.3f}.")

# --- ACTIVITY 3: TREE VS FOREST COMPARISON ---
elif activity == "Activity 3 - Tree vs. Forest Comparison":
    st.header("Activity 3: Model Comparison Sandbox")
    st.write("Adjust parameters for both models to answer the final question: Which one performs better and why?")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Tree Settings")
        dt_depth = st.slider("DT Max Depth", 1, 30, 5, help="Single trees with high depth often overfit.")
        
    with col2:
        st.subheader("Random Forest Settings")
        rf_trees = st.slider("RF Estimators", 10, 500, 100, help="Ensembles reduce variance by averaging multiple trees.")

    if st.button("Compare Models"):
        X = df[feature_list].values
        y = df['Outcome'].values
        
        # Train both models on the same split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=42).fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=5, random_state=42).fit(X_train, y_train)
        
        dt_acc = dt.score(X_test, y_test)
        rf_acc = rf.score(X_test, y_test)
        
        c1, c2 = st.columns(2)
        c1.metric("Decision Tree Accuracy", f"{dt_acc:.3f}")
        c2.metric("Random Forest Accuracy", f"{rf_acc:.3f}", delta=f"{rf_acc - dt_acc:.3f}")
        
        st.write(f"**Data Summary:** At the selected parameters, the Decision Tree achieved {dt_acc:.1%} accuracy while the Random Forest achieved {rf_acc:.1%}.")
        st.info("Strategic Insight: Random Forests correct for the 'memorization' (overfitting) tendency of Decision Trees by aggregating multiple trees.")

    st.markdown("---")
    st.markdown("### Canvas Reflection Question")
    st.write("Which one performs better, random forest or decision tree? Use the metrics above to justify your answer on Canvas.")
