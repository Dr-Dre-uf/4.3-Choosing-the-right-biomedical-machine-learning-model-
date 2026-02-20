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

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_prep_data():
    data = load_diabetes(as_frame=True)
    df = data.frame.copy()
    # Binary target: 1 = High Risk, 0 = Low Risk
    df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
    X = df.drop(columns=['target', 'Outcome'])
    y = df['Outcome']
    return X, y, data.feature_names

X, y, feature_names = load_and_prep_data()

# Scale data globally for the session
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("ML Learning Labs")
mode = st.sidebar.radio(
    "Select a Lab Environment:",
    [
        "Lab 1: The Feature Inspector (LogReg)", 
        "Lab 2: Decision Boundary Visualizer", 
        "Lab 3: The Clinical 'What-If' Simulator"
    ]
)

# ==========================================
# LAB 1: FEATURE INSPECTOR (Logistic Regression)
# ==========================================
if mode == "Lab 1: The Feature Inspector (LogReg)":
    st.title("Lab 1: Feature Engineering & Log-Odds")
    st.markdown("""
    **The Goal:** The video script highlights Logistic Regression for its *feature interpretability*. 
    Instead of using all data, select specific clinical features below to see how the **Beta Coefficients** change and impact your total accuracy.
    """)
    
    # Interactive Feature Selection
    selected_features = st.multiselect(
        "Select Clinical Features to include in the model:",
        options=feature_names,
        default=["bmi", "bp", "s1", "s2", "s5"]
    )
    
    if len(selected_features) > 0:
        # Filter data based on selection
        feature_indices = [feature_names.index(f) for f in selected_features]
        X_train_sub = X_train[:, feature_indices]
        X_test_sub = X_test[:, feature_indices]
        
        # Train Model dynamically
        log_reg = LogisticRegression()
        log_reg.fit(X_train_sub, y_train)
        acc = log_reg.score(X_test_sub, y_test)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Custom Model Accuracy", f"{acc:.2%}", help="Accuracy based ONLY on the features you selected.")
            st.info("Notice how adding or removing highly correlated features (like different blood serum measurements) shifts the importance of the others.")
            
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
# LAB 2: DECISION BOUNDARY VISUALIZER
# ==========================================
elif mode == "Lab 2: Decision Boundary Visualizer":
    st.title("Lab 2: Visualizing How Models 'Think'")
    st.markdown("""
    **The Goal:** The script mentions SVM finds the "optimal decision boundary," while Decision Trees "recursively split" data. 
    Here, we squish the 10-dimensional clinical data down to 2 dimensions (using PCA) so you can actually *see* the boundaries each model draws.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Model Tuning")
        model_type = st.radio("Choose Model to Visualize:", ["Decision Tree", "Random Forest", "SVM"])
        
        if model_type == "Decision Tree":
            param = st.slider("Max Tree Depth", 1, 10, 3)
            clf = DecisionTreeClassifier(max_depth=param)
        elif model_type == "Random Forest":
            param = st.slider("Number of Trees", 1, 50, 10)
            clf = RandomForestClassifier(n_estimators=param, max_depth=3)
        else:
            param = st.select_slider("SVM Margin Regularization (C)", [0.01, 0.1, 1, 10, 100], value=1)
            clf = SVC(C=param, kernel='rbf')

    with col2:
        # Reduce data to 2D for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_train)
        clf.fit(X_pca, y_train)
        
        # Create a mesh grid
        x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
        y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        
        # Predict across the grid
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plotting
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolor='k', cmap='coolwarm', s=20)
        ax.set_title(f"{model_type} Decision Boundary (2D PCA)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        st.pyplot(fig)
        st.caption("Red Area = Predicted High Risk. Blue Area = Predicted Low Risk. Notice how Trees draw blocky boxes, while SVM draws smooth, complex curves.")

# ==========================================
# LAB 3: CLINICAL "WHAT-IF" SIMULATOR
# ==========================================
elif mode == "Lab 3: The Clinical 'What-If' Simulator":
    st.title("Lab 3: The Multi-Model Patient Simulator")
    st.markdown("""
    **The Goal:** Train all four models in the background. Then, adjust the clinical sliders for a hypothetical patient. 
    Watch how the different models might disagree on the same patient based on their underlying math.
    """)
    
    # Train all models quietly
    models = {
        "Logistic Regression": LogisticRegression().fit(X_train, y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=4).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=50).fit(X_train, y_train),
        "SVM": SVC(probability=True).fit(X_train, y_train)
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Patient Vitals")
    
    # Sliders for the most impactful standardized features
    sim_bmi = st.sidebar.slider("Body Mass Index (Standardized)", -3.0, 3.0, 0.0)
    sim_bp = st.sidebar.slider("Blood Pressure (Standardized)", -3.0, 3.0, 0.0)
    sim_s5 = st.sidebar.slider("Serum Measure S5 (Standardized)", -3.0, 3.0, 0.0)
    
    # Construct a synthetic patient array (filling unselected features with 0/mean)
    synthetic_patient = np.zeros((1, 10))
    synthetic_patient[0, feature_names.index('bmi')] = sim_bmi
    synthetic_patient[0, feature_names.index('bp')] = sim_bp
    synthetic_patient[0, feature_names.index('s5')] = sim_s5
    
    st.subheader("Real-Time Model Consensus")
    cols = st.columns(4)
    
    for idx, (name, model) in enumerate(models.items()):
        prediction = model.predict(synthetic_patient)[0]
        status = "High Risk" if prediction == 1 else "Low Risk"
        color = "red" if prediction == 1 else "green"
        
        with cols[idx]:
            st.markdown(f"**{name}**")
            st.markdown(f"<h3 style='color: {color};'>{status}</h3>", unsafe_allow_html=True)
            
    st.markdown("---")
    st.info("**Why do they disagree?** As mentioned in the video, Random Forests handle complex interactions better than Logistic Regression. If you set BMI extremely high but BP extremely low, the models will weigh those conflicting signals differently.")
