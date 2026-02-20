import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical ML Activities", layout="wide")

# --- DATA LOADING ---
@st.cache_data
def load_clinical_data():
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    # Create a synthetic binary outcome (1 = High Risk, 0 = Low Risk)
    df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
    df.drop(columns='target', inplace=True)
    return df

df = load_clinical_data()
X = df.drop(columns='Outcome')
y = df['Outcome']

# Split and Scale data for the activities
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Module Navigation")
activity = st.sidebar.radio(
    "Select an Activity:",
    ["Introduction", "Activity 1: Data & Logistic Regression", "Activity 2: Trees & Forests", "Activity 3: SVM & Model Comparison"],
    help="Work through these activities in order to understand how different models handle the same clinical data."
)

# --- INTRODUCTION ---
if activity == "Introduction":
    st.title("Choosing the Right Biomedical ML Model")
    st.markdown("""
    Welcome! In this interactive module, we will explore the basic algorithmic concepts of several classic machine learning models and compare their similarities and differences.
    
    **Learning Objectives:**
    * Understand the inputs, outputs, and parameters of classic ML models.
    * Compare feature interpretability vs. prediction accuracy.
    * Learn guidelines for choosing the right model in biomedical contexts.
    
    Navigate to **Activity 1** in the sidebar to begin.
    """)

# --- ACTIVITY 1: LOGISTIC REGRESSION ---
elif activity == "Activity 1: Data & Logistic Regression":
    st.title("Activity 1: The Baseline & Logistic Regression")
    
    with st.expander("📝 Activity Instructions", expanded=True):
        st.write("""
        1. Review the clinical dataset below. Notice the input features (covariates) and the binary output.
        2. Click 'Train Logistic Regression' to see the model's beta coefficients.
        3. Observe which features have the strongest effect on the outcome.
        """)

    st.subheader("1. The Biomedical Dataset")
    st.dataframe(df.head(), use_container_width=True)
    st.caption("Input Features: Patient characteristics (age, sex, bmi, bp, etc.). Output: Disease status (Outcome).")

    st.markdown("---")
    st.subheader("2. Logistic Regression Interpretability")
    st.info("Logistic regression stands out for its simplicity and interpretability. The model parameters are beta coefficients, which represent the effect of each predictor variable on the log-odds of the outcome.")
    
    if st.button("Train Logistic Regression & Extract Coefficients", help="Fits the model to the data and extracts the calculated Beta coefficients."):
        log_reg = LogisticRegression()
        log_reg.fit(X_train_scaled, y_train)
        acc = log_reg.score(X_test_scaled, y_test)
        
        st.success(f"Model trained! Test Accuracy: **{acc:.2%}**")
        
        # Display Coefficients
        coef_df = pd.DataFrame({
            'Feature': X.columns,
            'Beta Coefficient': log_reg.coef_[0]
        }).sort_values(by='Beta Coefficient', ascending=False)
        
        st.bar_chart(coef_df.set_index('Feature'))
        st.caption("Higher positive/negative values indicate a stronger effect on the probability of a High Risk outcome.")

# --- ACTIVITY 2: TREES & FORESTS ---
elif activity == "Activity 2: Trees & Forests":
    st.title("Activity 2: Decision Trees vs. Random Forests")
    
    with st.expander("📝 Activity Instructions", expanded=True):
        st.write("""
        1. Adjust the 'Max Depth' of the Decision Tree to see how it affects accuracy.
        2. Adjust the 'Number of Trees' in the Random Forest.
        3. Compare the performance. Does the ensemble method (Random Forest) reduce overfitting and improve generalization as the script suggests?
        """)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Decision Tree")
        st.write("Recursively splits the dataset based on feature values.")
        tree_depth = st.slider("Max Tree Depth", min_value=1, max_value=20, value=3, 
                               help="Limits how many consecutive splits the tree can make. Deeper trees memorize data (overfit) but may fail on new data.")
        
        dt_model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)
        dt_model.fit(X_train, y_train)
        dt_acc = dt_model.score(X_test, y_test)
        st.metric("Decision Tree Accuracy", f"{dt_acc:.2%}")

    with col2:
        st.subheader("Random Forest")
        st.write("An ensemble method combining multiple decision trees.")
        n_trees = st.slider("Number of Trees in Forest", min_value=10, max_value=200, value=50, step=10,
                            help="More trees usually improve accuracy and prevent overfitting, but take longer to compute.")
        
        rf_model = RandomForestClassifier(n_estimators=n_trees, max_depth=tree_depth, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_acc = rf_model.score(X_test, y_test)
        st.metric("Random Forest Accuracy", f"{rf_acc:.2%}", delta=f"{(rf_acc - dt_acc):.2%} vs DT")

    
    st.info("Notice how the Random Forest (usually) outperforms the single Decision Tree. However, you lose the ability to visualize a single, clean decision path.")

# --- ACTIVITY 3: SVM & COMPARISON ---
elif activity == "Activity 3: SVM & Model Comparison":
    st.title("Activity 3: Support Vector Machines & Final Comparison")
    
    with st.expander("📝 Activity Instructions", expanded=True):
        st.write("""
        1. Train the Support Vector Machine (SVM).
        2. Review the final comparison matrix.
        3. Reflect on the trade-offs between interpretability and computational requirements when selecting a model for a clinical setting.
        """)

    st.subheader("Support Vector Machine (SVM)")
    st.write("Finds the optimal decision boundary that maximizes the margin between the two classes.")
    
    c_param = st.select_slider("SVM Regularization (C)", options=[0.1, 1.0, 10.0, 100.0], value=1.0, 
                               help="Controls the trade-off between achieving a low training error and a low testing error. High C can lead to overfitting.")
    
    if st.button("Train SVM"):
        with st.spinner("Training SVM... (Note: SVMs can be computationally heavy on large datasets)"):
            time.sleep(1) # Simulated delay to emphasize the script's point about compute time
            svm_model = SVC(C=c_param, random_state=42)
            svm_model.fit(X_train_scaled, y_train)
            svm_acc = svm_model.score(X_test_scaled, y_test)
            st.success(f"SVM trained! Accuracy: **{svm_acc:.2%}**")

    st.markdown("---")
    st.subheader("Final Review: Selecting the Right Model")
    st.write("As summarized in the video, choosing a model involves balancing multiple priorities.")
    
    comparison_data = {
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
        "Best For": ["Feature Interpretability", "Simple Rule Extraction", "High Accuracy & Generalization", "Complex Binary Boundaries"],
        "Drawbacks": ["Limited with complex, non-linear data", "Prone to overfitting", "Lacks feature interpretability ('Black Box')", "Computationally slow on large datasets"]
    }
    st.table(pd.DataFrame(comparison_data))
