import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical ML Sandbox", layout="wide")

# --- SIDEBAR NAVIGATION & SETTINGS ---
st.sidebar.title("Module Settings")
scientific_context = st.sidebar.radio(
    "Select Learning Context:",
    ["Clinical (Patient Care)", "Foundational (Basic Science)"],
    help="Toggle terminology to match your study field. The underlying math remains identical."
)
st.sidebar.markdown("---")

st.sidebar.title("ML Learning Module")
mode = st.sidebar.radio(
    "Select an Activity:",
    [
        "Activity 1: The Feature Inspector (LogReg)", 
        "Activity 2: Decision Boundary Visualizer", 
        "Activity 3: The What-If Simulator",
        "Activity 4: Cross-Validation and Metrics"
    ],
    help="Navigate through interactive activities to explore different machine learning models."
)

# --- DATA LOADING ---
@st.cache_data
def load_and_prep_data(context):
    # Load from the path specified in the notebook
    data_path = "data/diabetes.csv"
    if not os.path.exists(data_path):
        # Local fallback for development environment
        data_path = "diabetes.csv"
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Rename columns based on the selected context
    if context == "Clinical (Patient Care)":
        rename_map = {
            'Pregnancies': 'Pregnancies', 'Glucose': 'Glucose', 
            'BloodPressure': 'Blood Pressure', 'SkinThickness': 'Skin Thickness', 
            'Insulin': 'Insulin', 'BMI': 'BMI', 
            'DiabetesPedigreeFunction': 'Diabetes Pedigree', 'Age': 'Age'
        }
    else:
        rename_map = {
            'Pregnancies': 'Reproductive Cycles', 'Glucose': 'Metabolite Alpha', 
            'BloodPressure': 'Hydrostatic Pressure', 'SkinThickness': 'Tissue Thickness', 
            'Insulin': 'Hormone Level', 'BMI': 'Mass Index', 
            'DiabetesPedigreeFunction': 'Lineage Factor', 'Age': 'Specimen Age'
        }
        
    X = X.rename(columns=rename_map)
    return X, y, list(X.columns)

X, y, feature_names = load_and_prep_data(scientific_context)

# Scale data globally for consistency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

target_positive = "High Risk" if scientific_context == "Clinical (Patient Care)" else "Positive Outcome"
target_negative = "Low Risk" if scientific_context == "Clinical (Patient Care)" else "Negative Outcome"

# ==========================================
# ACTIVITY 1: FEATURE INSPECTOR
# ==========================================
if mode == "Activity 1: The Feature Inspector (LogReg)":
    st.title("Activity 1: Feature Engineering and Log-Odds")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Select specific features from the dropdown to include them in the model.
        2. Observe the Accuracy metric and the Beta Coefficient chart.
        3. Identify which clinical or foundational features have the strongest impact on the outcome.
        """)
    
    selected_features = st.multiselect(
        "Select Features for the Logistic Regression model:",
        options=feature_names,
        default=feature_names[1:6],
        help="Select predictors. Logistic regression uses these to calculate the log-odds of the outcome."
    )
    
    if selected_features:
        feature_indices = [feature_names.index(f) for f in selected_features]
        log_reg = LogisticRegression().fit(X_train[:, feature_indices], y_train)
        acc = log_reg.score(X_test[:, feature_indices], y_test)
        
        c1, c2 = st.columns([1, 2])
        c1.metric("Model Accuracy", f"{acc:.2%}")
        
        coef_df = pd.DataFrame({'Feature': selected_features, 'Beta': log_reg.coef_[0]}).sort_values('Beta')
        c2.subheader("Beta Coefficients")
        st.bar_chart(coef_df.set_index('Feature'))
        st.caption("A chart showing feature weights. Positive values indicate a direct relationship with the outcome.")

# ==========================================
# ACTIVITY 2: DECISION BOUNDARY VISUALIZER
# ==========================================
elif mode == "Activity 2: Decision Boundary Visualizer":
    st.title("Activity 2: Visualizing Model Logic")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("Compare how different models draw boundaries to separate data classes.")
    
    model_type = st.radio("Select Model:", ["Decision Tree", "Random Forest", "SVM"], help="Choose the algorithm structure.")
    
    if model_type == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=st.slider("Tree Depth", 1, 10, 3))
    elif model_type == "Random Forest":
        clf = RandomForestClassifier(n_estimators=st.slider("Number of Trees", 10, 100, 20), max_depth=3)
    else:
        clf = SVC(C=st.select_slider("Regularization (C)", [0.1, 1, 10, 100]), kernel='rbf')

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    clf.fit(X_pca, y_train)
    
    # Boundary visualization using viridis (colorblind safe)
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolors='k', cmap='viridis')
    st.pyplot(fig)
    st.caption(f"Decision boundaries separating {target_positive} from {target_negative}.")

# ==========================================
# ACTIVITY 3: WHAT-IF SIMULATOR
# ==========================================
elif mode == "Activity 3: The What-If Simulator":
    st.title("Activity 3: Multi-Model Patient Simulation")
    
    st.sidebar.subheader("Adjust Vitals")
    v1 = st.sidebar.slider(f"{feature_names[1]}", -3.0, 3.0, 0.0)
    v2 = st.sidebar.slider(f"{feature_names[5]}", -3.0, 3.0, 0.0)
    
    # Synthetic patient
    patient = np.zeros((1, 8))
    patient[0, 1], patient[0, 5] = v1, v2
    
    cols = st.columns(4)
    models = {
        "LogReg": LogisticRegression().fit(X_train, y_train),
        "Tree": DecisionTreeClassifier(max_depth=3).fit(X_train, y_train),
        "Forest": RandomForestClassifier(n_estimators=50).fit(X_train, y_train),
        "SVM": SVC().fit(X_train, y_train)
    }
    
    for i, (name, m) in enumerate(models.items()):
        pred = m.predict(patient)[0]
        status = target_positive if pred == 1 else target_negative
        cols[i].markdown(f"**{name} Prediction:**\n### {status}")

# ==========================================
# ACTIVITY 4: CROSS-VALIDATION
# ==========================================
elif mode == "Activity 4: Cross-Validation and Metrics":
    st.title("Activity 4: Evaluating Model Stability")
    
    if st.button("Run 5-Fold Cross Validation", help="Trains the model 5 times on different data slices."):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
        
        metrics = []
        for train_idx, val_idx in kf.split(X_scaled):
            model.fit(X_scaled[train_idx], y.values[train_idx])
            y_pred = model.predict(X_scaled[val_idx])
            tn, fp, fn, tp = confusion_matrix(y.values[val_idx], y_pred).ravel()
            metrics.append([
                (tp+tn)/(tp+tn+fp+fn), # Acc
                tp/(tp+fn),           # Sens
                tn/(tn+fp)            # Spec
            ])
        
        
        res_df = pd.DataFrame(metrics, columns=['Accuracy', 'Sensitivity', 'Specificity'])
        st.write("Average Performance Across 5 Folds:")
        st.table(res_df.mean())
        st.line_chart(res_df)
