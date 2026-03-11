import streamlit as st
import pandas as pd
import numpy as np
import psutil
import os
import matplotlib.pyplot as plt
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
    st.sidebar.caption("Tracks the resource usage of this app in real-time.")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%", help="Real-time CPU processing load.")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB", help="Real-time memory allocation.")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Choosing the Right Biomedical Machine Learning Model", layout="wide")

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
    [
        "Activity 1 - Data Exploration", 
        "Activity 2 - Random Forest Training", 
        "Activity 3 - Tree vs. Forest Comparison"
    ],
    help="Navigate through the machine learning activities as outlined in the notebook."
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

st.title("Choosing the Right Biomedical Machine Learning Model")

# --- ACTIVITY 1: DATA EXPLORATION ---
if activity == "Activity 1 - Data Exploration":
    st.header("Activity 1: Exploring Feature Relationships")
    st.write("Inspect the raw data to understand feature distributions and how variables interact with one another to influence the outcome.")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis_feature = st.selectbox("Select X-Axis Feature:", feature_list, index=1, help="Choose the variable for the horizontal axis.")
    with col2:
        y_axis_feature = st.selectbox("Select Y-Axis Feature:", feature_list, index=5, help="Choose the variable for the vertical axis.")
    
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # ADA Compliant Color Palette (viridis)
    scatter = ax.scatter(df[x_axis_feature], df[y_axis_feature], c=df['Outcome'], cmap='viridis', alpha=0.7, edgecolors='k')
    ax.set_xlabel(x_axis_feature)
    ax.set_ylabel(y_axis_feature)
    
    handles, labels = scatter.legend_elements()
    labels = ['Survival / Negative', 'Death / Positive']
    ax.legend(handles, labels, title="Outcome")
    
    st.pyplot(fig)
    st.write(f"**Data Summary:** A scatter plot comparing {x_axis_feature} (X-axis) against {y_axis_feature} (Y-axis). The darker points represent negative/survival outcomes, and the lighter points represent positive/death outcomes. Look for distinct clustering to see if these two variables easily separate the classes.")

    st.markdown("---")
    st.subheader("Outcome Distribution Overview")
    counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
    st.bar_chart(counts, color="#1f77b4")
    st.write(f"**Data Summary:** The dataset contains {counts[0]} survival cases and {counts[1]} death cases. This demonstrates a class imbalance.")

# --- ACTIVITY 2: RANDOM FOREST TRAINING ---
elif activity == "Activity 2 - Random Forest Training":
    st.header("Activity 2: Random Forest Model Training")
    st.write("Train a Random Forest ensemble model using 5-fold cross-validation and observe performance stability and feature importance.")

    st.sidebar.subheader("RF Hyperparameters")
    n_estimators = st.sidebar.slider("Number of Trees", 50, 1000, 500, step=50, help="Matches the default in the notebook.")
    max_depth = st.sidebar.slider("Maximum Tree Depth", 2, 20, 10, help="Limits how deep each individual tree can grow.")
    
    if st.button("Execute 5-Fold Evaluation", help="Initiate the cross-validation training process."):
        X = df[feature_list].values
        y = df['Outcome'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced", random_state=42)
        
        metrics = []
        progress_bar = st.progress(0, text="Training across 5 folds...")
        
        for i, (train_idx, val_idx) in enumerate(kf.split(X)):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            tn, fp, fn, tp = confusion_matrix(y[val_idx], y_pred).ravel()
            metrics.append([(tp+tn)/(tp+tn+fp+fn), tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
            progress_bar.progress((i + 1) / 5, text=f"Completed fold {i+1}/5")
            
        st.session_state['cv_res'] = pd.DataFrame(metrics, columns=['Accuracy', 'Sensitivity', 'Specificity', 'Precision'])
        
        # Save feature importances from the final fold
        st.session_state['feature_importances'] = model.feature_importances_

    if 'cv_res' in st.session_state:
        st.table(st.session_state['cv_res'])
        st.line_chart(st.session_state['cv_res']['Accuracy'])
        st.write(f"**Data Summary:** The table and chart display performance across 5 folds. The average Cross-Validation Accuracy is {st.session_state['cv_res']['Accuracy'].mean():.3f}.")
        
        st.markdown("---")
        st.subheader("Feature Importance Analysis")
        imp_df = pd.DataFrame({
            'Feature': feature_list,
            'Importance': st.session_state['feature_importances']
        }).sort_values(by='Importance', ascending=True)
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(imp_df['Feature'], imp_df['Importance'], color='#1f77b4')
        ax2.set_xlabel('Relative Importance')
        st.pyplot(fig2)
        st.write("**Data Summary:** A horizontal bar chart ranking the variables by how heavily the Random Forest relied on them to make predictions. Features with longer bars had the greatest impact on the model's accuracy.")

# --- ACTIVITY 3: TREE VS FOREST COMPARISON ---
elif activity == "Activity 3 - Tree vs. Forest Comparison":
    st.header("Activity 3: Model Comparison Sandbox")
    st.write("Adjust parameters for both models and the data split size to investigate why an ensemble approach is often preferred over a single tree.")

    st.subheader("Data Split Configuration")
    test_size_pct = st.slider("Test Set Size (%)", 10, 50, 20, help="Adjust the percentage of data held back for testing. Smaller training sets usually cause single decision trees to overfit more severely.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Tree Settings")
        dt_depth = st.slider("DT Max Depth", 1, 30, 5, help="Controls the depth of the single decision tree.")
        
    with col2:
        st.subheader("Random Forest Settings")
        rf_trees = st.slider("RF Estimators", 10, 500, 100, help="Determines how many trees are averaged in the forest.")

    if st.button("Compare Models", help="Train both models simultaneously on the configured data split."):
        X = df[feature_list].values
        y = df['Outcome'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(test_size_pct / 100.0), random_state=42)
        
        dt = DecisionTreeClassifier(max_depth=dt_depth, random_state=42).fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=rf_trees, max_depth=5, random_state=42).fit(X_train, y_train)
        
        dt_acc = dt.score(X_test, y_test)
        rf_acc = rf.score(X_test, y_test)
        
        c1, c2 = st.columns(2)
        c1.metric("Decision Tree Accuracy", f"{dt_acc:.3f}")
        c2.metric("Random Forest Accuracy", f"{rf_acc:.3f}", delta=f"{rf_acc - dt_acc:.3f}")
        
        st.write(f"**Data Summary:** Using a test size of {test_size_pct}%, the Decision Tree achieved {dt_acc:.1%} accuracy and the Random Forest achieved {rf_acc:.1%}.")
        st.info("Strategic Insight: Random Forests generally provide more stable predictions by averaging multiple views of the data, which prevents the model from memorizing noise.")

    st.markdown("---")
    st.markdown("### Reflection Question")
    st.write("Which one performs better, random forest or decision tree? Use the metrics generated in these activities to justify your answer.")
