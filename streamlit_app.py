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
        "Activity 2 - Interactive Training", 
        "Activity 3 - Performance Metrics",
        "Activity 4 - Strategic Evaluation"
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
    st.header("Activity 1: Exploratory Data Analysis")
    st.write("Before training, inspect the dataset to understand feature distributions and class imbalance.")
    
    st.subheader("Interactive Feature Viewer")
    selected_feature = st.selectbox("Select a feature to analyze:", feature_list)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Outcome Distribution**")
        counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(counts, color="#1f77b4")
        st.write(f"**Data Summary:** The dataset contains {counts[0]} survival cases and {counts[1]} death cases. The target variable is binary.")
        
    with col2:
        st.markdown(f"**Average {selected_feature} by Outcome**")
        means = df.groupby('Outcome')[selected_feature].mean()
        st.bar_chart(means, color="#ff7f0e")
        st.write(f"**Data Summary:** Average {selected_feature} for Survivors is {means[0]:.2f}, compared to {means[1]:.2f} for Deaths.")

# --- ACTIVITY 2: INTERACTIVE TRAINING ---
elif activity == "Activity 2 - Interactive Training":
    st.header("Activity 2: Model Configuration")
    st.write("Adjust the parameters below to see how model complexity affects the learning process.")

    # Hyperparameter Sliders
    st.sidebar.subheader("Model Hyperparameters")
    n_trees = st.sidebar.slider("Number of Trees", 10, 1000, 500, step=50)
    tree_depth = st.sidebar.slider("Maximum Depth", 1, 20, 5)
    
    selected_inputs = st.multiselect(
        "Select Features to Include in Model:",
        options=feature_list,
        default=feature_list
    )

    if st.button("Execute Training Run"):
        if not selected_inputs:
            st.error("Please select at least one feature.")
        else:
            X = df[selected_inputs].values
            y = df['Outcome'].values
            
            # 5-Fold Evaluation logic from Notebook
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            model = RandomForestClassifier(
                n_estimators=n_trees, 
                max_depth=tree_depth,
                class_weight="balanced", 
                random_state=42
            )
            
            accuracies = []
            for train_idx, val_idx in kf.split(X):
                model.fit(X[train_idx], y[train_idx])
                accuracies.append(model.score(X[val_idx], y[val_idx]))
            
            st.session_state['demo_acc'] = np.mean(accuracies)
            st.success("Training run complete.")

    if 'demo_acc' in st.session_state:
        st.metric("Mean Cross-Validation Accuracy", f"{st.session_state['demo_acc']:.4f}")
        st.write(f"**Data Summary:** Using {n_trees} trees with a depth of {tree_depth}, the model achieved a mean accuracy of {st.session_state['demo_acc']:.2%}.")

# --- ACTIVITY 3: PERFORMANCE METRICS ---
elif activity == "Activity 3 - Performance Metrics":
    st.header("Activity 3: Advanced Metric Analysis")
    st.write("In biomedical contexts, accuracy is often insufficient. Examine Sensitivity and Specificity.")

    

    threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 
                         help="Lowering the threshold increases Sensitivity (catching more deaths) but reduces Specificity.")
    
    X = df[feature_list].values
    y = df['Outcome'].values
    
    # Simple split for the threshold demo
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_s, y)
    probs = model.predict_proba(X_s)[:, 1]
    preds = (probs > threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    c1, c2 = st.columns(2)
    c1.metric("Sensitivity (Recall)", f"{sens:.3f}")
    c2.metric("Specificity", f"{spec:.3f}")
    
    st.write(f"**Data Summary:** At a threshold of {threshold}, the model correctly identified {sens:.1%} of actual mortality cases (Sensitivity) and {spec:.1%} of actual survival cases (Specificity).")

# --- ACTIVITY 4: STRATEGIC EVALUATION ---
elif activity == "Activity 4 - Strategic Evaluation":
    st.header("Activity 4: Model Selection Strategy")
    st.write("Determine the best approach based on your specific requirements.")
    
    requirement = st.selectbox(
        "What is your primary organizational requirement?",
        ["Maximum Predictive Power", "Legal/Regulatory Transparency", "Speed and Efficiency"]
    )
    
    if requirement == "Maximum Predictive Power":
        st.success("Recommendation: Random Forest. This ensemble approach reduces overfitting and handles complex interactions between lab results.")
    elif requirement == "Legal/Regulatory Transparency":
        st.info("Recommendation: Single Decision Tree. This 'White Box' model allows clinicians to follow the exact 'If-Then' logic of the prediction.")
    else:
        st.warning("Recommendation: Logistic Regression. This baseline model is computationally light and provides direct coefficients for each variable.")
