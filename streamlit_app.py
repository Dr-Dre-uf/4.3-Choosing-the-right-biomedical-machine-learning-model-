import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# --- PAGE CONFIG ---
st.set_page_config(page_title="Biomedical ML Sandbox", layout="wide")

# --- SIDEBAR NAVIGATION & SETTINGS ---
st.sidebar.title("Module Settings")
scientific_context = st.sidebar.radio(
    "Select Learning Context:",
    ["Clinical (Patient Care)", "Foundational (Basic Science)"],
    help="Toggle the terminology to match your specific field of study. The underlying math remains identical."
)
st.sidebar.markdown("---")

st.sidebar.title("ML Learning Module")
mode = st.sidebar.radio(
    "Select an Activity:",
    [
        "Activity 1: The Feature Inspector (LogReg)", 
        "Activity 2: Decision Boundary Visualizer", 
        "Activity 3: The 'What-If' Simulator"
    ],
    help="Navigate through the interactive activities to explore different machine learning models."
)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_prep_data(context):
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    
    # Binary target: 1 = Positive/High Risk, 0 = Negative/Low Risk
    df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
    X = df.drop(columns=['target', 'Outcome'])
    y = df['Outcome']
    
    # Contextual Column Mapping
    if context == "Clinical (Patient Care)":
        rename_map = {
            'age': 'Age', 'sex': 'Sex', 'bmi': 'BMI', 'bp': 'Blood Pressure',
            's1': 'Total Cholesterol', 's2': 'LDL', 's3': 'HDL', 
            's4': 'TC/HDL Ratio', 's5': 'Triglycerides', 's6': 'Blood Sugar'
        }
    else:
        # Foundational Science Abstractions
        rename_map = {
            'age': 'Specimen Age (Days)', 'sex': 'Biological Sex', 
            'bmi': 'Organism Mass Index', 'bp': 'Hydrostatic Pressure',
            's1': 'Biomarker Alpha (s1)', 's2': 'Biomarker Beta (s2)', 
            's3': 'Biomarker Gamma (s3)', 's4': 'Assay Ratio (s4)', 
            's5': 'Target Metabolite (s5)', 's6': 'Glucose Concentration'
        }
        
    X = X.rename(columns=rename_map)
    return X, y, list(X.columns)

# Load data dynamically based on the radio button selection
X, y, feature_names = load_and_prep_data(scientific_context)

# Scale data globally for the session
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Set dynamic target labels based on context
target_positive = "High Risk" if scientific_context == "Clinical (Patient Care)" else "Positive Assay"
target_negative = "Low Risk" if scientific_context == "Clinical (Patient Care)" else "Negative Assay"

# ==========================================
# ACTIVITY 1: FEATURE INSPECTOR
# ==========================================
if mode == "Activity 1: The Feature Inspector (LogReg)":
    st.title("Activity 1: Feature Engineering & Log-Odds")
    
    with st.expander("📝 Activity Instructions", expanded=True):
        st.write("""
        1. Review the available features in the dropdown below.
        2. Add or remove features to see how the Logistic Regression model adapts.
        3. Observe the bar chart to see how the **Beta Coefficients** change dynamically based on your feature selection.
        """)
    
    selected_features = st.multiselect(
        "Select Features to include in the model:",
        options=feature_names,
        default=feature_names[2:7], # Selects BMI/Mass Index through s3
        help="Logistic regression relies on these features to calculate the log-odds of the outcome. Removing highly correlated features can shift the weight of the remaining ones."
    )
    
    if len(selected_features) > 0:
        feature_indices = [feature_names.index(f) for f in selected_features]
        X_train_sub = X_train[:, feature_indices]
        X_test_sub = X_test[:, feature_indices]
        
        log_reg = LogisticRegression()
        log_reg.fit(X_train_sub, y_train)
        acc = log_reg.score(X_test_sub, y_test)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Custom Model Accuracy", f"{acc:.2%}", help="This accuracy score is calculated using ONLY the features you selected above.")
            st.info("Logistic Regression is highly interpretable because we can extract the exact mathematical weight (Beta Coefficient) it assigns to every single feature.")
            
        with col2:
            st.subheader("Beta Coefficients (Log-Odds Impact)")
            coef_df = pd.DataFrame({
                'Feature': selected_features,
                'Weight (Beta)': log_reg.coef_[0]
            }).sort_values(by='Weight (Beta)')
            
            st.bar_chart(coef_df.set_index('Feature'))
    else:
        st.warning("Please select at least one feature to train the model.")

# ==========================================
# ACTIVITY 2: DECISION BOUNDARY VISUALIZER
# ==========================================
elif mode == "Activity 2: Decision Boundary Visualizer":
    st.title("Activity 2: Visualizing How Models 'Think'")
    
    with st.expander("📝 Activity Instructions", expanded=True):
        st.write(f"""
        1. Select a machine learning model from the radio buttons.
        2. Adjust the model's specific hyperparameters using the sliders.
        3. Watch the 2D plot update to see exactly how the model draws its 'decision boundary' between {target_positive} (Red) and {target_negative} (Blue).
        """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Model Tuning")
        model_type = st.radio(
            "Choose Model to Visualize:", 
            ["Decision Tree", "Random Forest", "SVM"],
            help="Select the algorithm. Notice how Trees create blocky, recursive splits, while SVMs create smoother geometric boundaries."
        )
        
        if model_type == "Decision Tree":
            param = st.slider("Max Tree Depth", 1, 10, 3, help="Limits how many times the tree can recursively split the data. Higher depth increases accuracy on training data but risks overfitting.")
            clf = DecisionTreeClassifier(max_depth=param)
        elif model_type == "Random Forest":
            param = st.slider("Number of Trees", 1, 50, 10, help="An ensemble method. More trees generally improve accuracy and reduce the overfitting seen in single Decision Trees.")
            clf = RandomForestClassifier(n_estimators=param, max_depth=3)
        else:
            param = st.select_slider("SVM Margin Regularization (C)", [0.01, 0.1, 1, 10, 100], value=1, help="Controls the trade-off between maximizing the margin and minimizing misclassifications. A higher C forces the model to classify training points strictly.")
            clf = SVC(C=param, kernel='rbf')

    with col2:
        # Reduce data to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)
        clf.fit(X_pca, y_train)
        
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolor='k', cmap='coolwarm', s=20)
        ax.set_title(f"{model_type} Decision Boundary (2D PCA)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        st.pyplot(fig)
        st.caption(f"Red Area = Predicted {target_positive}. Blue Area = Predicted {target_negative}.")

# ==========================================
# ACTIVITY 3: THE SIMULATOR
# ==========================================
elif mode == "Activity 3: The 'What-If' Simulator":
    st.title("Activity 3: The Multi-Model Simulator")
    
    with st.expander("📝 Activity Instructions", expanded=True):
        st.write("""
        1. Use the sidebar sliders to adjust the standardized metrics for a hypothetical profile.
        2. Watch the 'Real-Time Model Consensus' panel to see how all four models evaluate the exact same data.
        3. Try creating edge cases (e.g., highly conflicting variables) to see when the models disagree based on their underlying logic.
        """)
    
    models = {
        "Logistic Regression": LogisticRegression().fit(X_train, y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=4).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=50).fit(X_train, y_train),
        "SVM": SVC(probability=True).fit(X_train, y_train)
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Variables")
    
    # Dynamically select variable names based on context toggle
    var1_name = "BMI" if scientific_context == "Clinical (Patient Care)" else "Organism Mass Index"
    var2_name = "Blood Pressure" if scientific_context == "Clinical (Patient Care)" else "Hydrostatic Pressure"
    var3_name = "Triglycerides" if scientific_context == "Clinical (Patient Care)" else "Target Metabolite (s5)"
    
    sim_var1 = st.sidebar.slider(f"{var1_name} (Standardized)", -3.0, 3.0, 0.0, help=f"Adjust the standardized {var1_name}. 0.0 represents the average in this dataset.")
    sim_var2 = st.sidebar.slider(f"{var2_name} (Standardized)", -3.0, 3.0, 0.0, help=f"Adjust the standardized {var2_name}.")
    sim_var3 = st.sidebar.slider(f"{var3_name} (Standardized)", -3.0, 3.0, 0.0, help=f"Adjust the standardized {var3_name}.")
    
    synthetic_profile = np.zeros((1, 10))
    synthetic_profile[0, feature_names.index(var1_name)] = sim_var1
    synthetic_profile[0, feature_names.index(var2_name)] = sim_var2
    synthetic_profile[0, feature_names.index(var3_name)] = sim_var3
    
    st.subheader("Real-Time Model Consensus")
    cols = st.columns(4)
    
    for idx, (name, model) in enumerate(models.items()):
        prediction = model.predict(synthetic_profile)[0]
        status = target_positive if prediction == 1 else target_negative
        color = "red" if prediction == 1 else "green"
        
        with cols[idx]:
            st.markdown(f"**{name}**")
            st.markdown(f"<h3 style='color: {color};'>{status}</h3>", unsafe_allow_html=True)
            
    st.markdown("---")
    st.info("**Why do they disagree?** Random Forests and SVMs handle complex, non-linear interactions better than Logistic Regression. If the models disagree, it highlights why selecting the right model requires balancing feature interpretability against raw algorithmic complexity.")
