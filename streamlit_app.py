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
import urllib.request

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
        "Activity 3: The 'What-If' Simulator",
        "Activity 4: Cross-Validation & Metrics"
    ],
    help="Navigate through the interactive activities to explore different machine learning models."
)

# --- DATA LOADING & CACHING ---
@st.cache_data
def load_and_prep_data(context):
    # Try loading local CSV, fallback to web URL if not present
    try:
        df = pd.read_csv("diabetes.csv")
    except FileNotFoundError:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(url, names=names)
    
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Contextual Column Mapping (Aligning with Pima Indians Dataset)
    if context == "Clinical (Patient Care)":
        rename_map = {
            'Pregnancies': 'Pregnancies', 'Glucose': 'Glucose', 
            'BloodPressure': 'Blood Pressure', 'SkinThickness': 'Skin Thickness', 
            'Insulin': 'Insulin', 'BMI': 'BMI', 
            'DiabetesPedigreeFunction': 'Diabetes Pedigree', 'Age': 'Age'
        }
    else:
        # Foundational Science Abstractions
        rename_map = {
            'Pregnancies': 'Reproductive Cycles', 'Glucose': 'Metabolite Alpha (Glucose)', 
            'BloodPressure': 'Hydrostatic Pressure', 'SkinThickness': 'Epidermal Thickness', 
            'Insulin': 'Hormone Level', 'BMI': 'Organism Mass Index', 
            'DiabetesPedigreeFunction': 'Genetic Lineage Factor', 'Age': 'Specimen Age'
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
target_positive = "High Risk / Mortality" if scientific_context == "Clinical (Patient Care)" else "Positive Assay"
target_negative = "Low Risk / Survival" if scientific_context == "Clinical (Patient Care)" else "Negative Assay"

# ==========================================
# ACTIVITY 1: FEATURE INSPECTOR
# ==========================================
if mode == "Activity 1: The Feature Inspector (LogReg)":
    st.title("Activity 1: Feature Engineering and Log-Odds")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Review the available features in the dropdown below.
        2. Add or remove features to see how the Logistic Regression model adapts.
        3. Observe the bar chart to see how the Beta Coefficients change dynamically based on your feature selection.
        """)
    
    selected_features = st.multiselect(
        "Select Features to include in the model:",
        options=feature_names,
        default=feature_names[1:6], 
        help="Logistic regression relies on these features to calculate the log-odds of the outcome."
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
            st.metric("Custom Model Accuracy", f"{acc:.2%}", help="Accuracy based ONLY on selected features.")
            st.info("Logistic Regression is highly interpretable because we can extract the exact mathematical weight (Beta Coefficient) assigned to every single feature.")
            
        with col2:
            st.subheader("Beta Coefficients (Log-Odds Impact)")
            coef_df = pd.DataFrame({
                'Feature': selected_features,
                'Weight (Beta)': log_reg.coef_[0]
            }).sort_values(by='Weight (Beta)')
            
            st.bar_chart(coef_df.set_index('Feature'))
            st.caption("Bar chart displaying the calculated Beta Coefficients. Positive values increase the likelihood of a positive outcome, while negative values decrease it.")
    else:
        st.warning("Please select at least one feature to train the model.")

# ==========================================
# ACTIVITY 2: DECISION BOUNDARY VISUALIZER
# ==========================================
elif mode == "Activity 2: Decision Boundary Visualizer":
    st.title("Activity 2: Visualizing How Models Decide")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write(f"""
        1. Select a machine learning model from the radio buttons.
        2. Adjust the model's specific hyperparameters using the sliders.
        3. Watch the 2D plot update to see exactly how the model draws its decision boundary between {target_positive} and {target_negative}.
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
            param = st.slider("Max Tree Depth", 1, 10, 3, help="Higher depth increases accuracy on training data but risks overfitting.")
            clf = DecisionTreeClassifier(max_depth=param, random_state=42)
        elif model_type == "Random Forest":
            param = st.slider("Number of Trees", 1, 50, 10, help="More trees generally improve accuracy and reduce overfitting.")
            clf = RandomForestClassifier(n_estimators=param, max_depth=3, random_state=42)
        else:
            param = st.select_slider("SVM Margin Regularization (C)", [0.01, 0.1, 1, 10, 100], value=1, help="Controls the trade-off between maximizing the margin and minimizing misclassifications.")
            clf = SVC(C=param, kernel='rbf', random_state=42)

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
        
        # ADA Update: Using 'viridis' colormap which is colorblind-safe
        contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, edgecolor='k', cmap='viridis', s=20)
        ax.set_title(f"{model_type} Decision Boundary (2D PCA)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        st.pyplot(fig)
        st.caption(f"A scatter plot visualizing the decision boundary. The lighter background region represents predicted {target_positive}, while the darker background region represents predicted {target_negative}. Points indicate individual records.")

# ==========================================
# ACTIVITY 3: THE SIMULATOR
# ==========================================
elif mode == "Activity 3: The 'What-If' Simulator":
    st.title("Activity 3: The Multi-Model Simulator")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Use the sidebar sliders to adjust the standardized metrics for a hypothetical profile.
        2. Watch the Real-Time Model Consensus panel to see how all four models evaluate the exact same data.
        3. Try creating edge cases (for example, highly conflicting variables) to see when the models disagree.
        """)
    
    models = {
        "Logistic Regression": LogisticRegression().fit(X_train, y_train),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train),
        "SVM": SVC(probability=True, random_state=42).fit(X_train, y_train)
    }
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Adjust Variables")
    
    var1_name = feature_names[1] 
    var2_name = feature_names[5] 
    var3_name = feature_names[7] 
    
    sim_var1 = st.sidebar.slider(f"{var1_name} (Standardized)", -3.0, 3.0, 0.0, help=f"Adjust the value for {var1_name}. 0.0 represents the dataset mean.")
    sim_var2 = st.sidebar.slider(f"{var2_name} (Standardized)", -3.0, 3.0, 0.0, help=f"Adjust the value for {var2_name}. 0.0 represents the dataset mean.")
    sim_var3 = st.sidebar.slider(f"{var3_name} (Standardized)", -3.0, 3.0, 0.0, help=f"Adjust the value for {var3_name}. 0.0 represents the dataset mean.")
    
    synthetic_profile = np.zeros((1, X_train.shape[1]))
    synthetic_profile[0, 1] = sim_var1
    synthetic_profile[0, 5] = sim_var2
    synthetic_profile[0, 7] = sim_var3
    
    st.subheader("Real-Time Model Consensus")
    cols = st.columns(4)
    
    for idx, (name, model) in enumerate(models.items()):
        prediction = model.predict(synthetic_profile)[0]
        status = target_positive if prediction == 1 else target_negative
        
        with cols[idx]:
            st.markdown(f"**{name}**")
            # ADA Update: Removed hardcoded CSS colors, relying on semantic markdown headers
            st.markdown(f"### {status}")
            
    st.markdown("---")
    st.info("Why do they disagree? Random Forests and SVMs handle complex, non-linear interactions better than Logistic Regression. If the models disagree, it highlights why selecting the right model requires balancing feature interpretability against raw algorithmic complexity.")

# ==========================================
# ACTIVITY 4: CROSS-VALIDATION & METRICS
# ==========================================
elif mode == "Activity 4: Cross-Validation & Metrics":
    st.title("Activity 4: Evaluating Model Stability")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("""
        1. Select a model below to evaluate.
        2. Run a 5-Fold Cross-Validation. This splits the data into 5 chunks, training the model 5 separate times to ensure its performance is robust.
        3. Compare the Average Sensitivity, Specificity, and Precision between the models to answer the final notebook question.
        """)
        
    model_choice = st.radio("Select Model for 5-Fold CV:", ["Decision Tree", "Random Forest"], help="Select which algorithm to evaluate across 5 distinct data splits.")
    
    if st.button(f"Run Cross-Validation on {model_choice}", help="Initiates the model training and evaluation process."):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        if model_choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
            
        acc_scores, sens_scores, spec_scores, prec_scores = [], [], [], []
        
        progress_bar = st.progress(0, text="Initializing Cross-Validation...")
        X_np = X.values 
        y_np = y.values
        
        # 5-Fold CV Loop
        for i, (train_index, val_index) in enumerate(kf.split(X_np)):
            X_train_fold, X_val_fold = X_np[train_index], X_np[val_index]
            y_train_fold, y_val_fold = y_np[train_index], y_np[val_index]
            
            fold_scaler = StandardScaler()
            X_train_fold = fold_scaler.fit_transform(X_train_fold)
            X_val_fold = fold_scaler.transform(X_val_fold)
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            
            tn, fp, fn, tp = confusion_matrix(y_val_fold, y_pred).ravel()
            
            acc = (tp + tn) / (tp + tn + fp + fn)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0 
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0 
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0 
            
            acc_scores.append(acc)
            sens_scores.append(sens)
            spec_scores.append(spec)
            prec_scores.append(prec)
            
            progress_bar.progress((i + 1) / 5, text=f"Processing Fold {i+1} of 5...")
            
        st.success("5-Fold Cross Validation Complete!")
        
        
        # Display Results
        st.subheader("Average Performance Across 5 Folds")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Accuracy", f"{np.mean(acc_scores):.3f}")
        col2.metric("Avg Sensitivity", f"{np.mean(sens_scores):.3f}", help="True Positive Rate: Ability to correctly identify positive cases.")
        col3.metric("Avg Specificity", f"{np.mean(spec_scores):.3f}", help="True Negative Rate: Ability to correctly identify negative cases.")
        col4.metric("Avg Precision", f"{np.mean(prec_scores):.3f}", help="Positive Predictive Value: Proportion of positive predictions that were correct.")
        
        # Display per-fold variance to show stability
        st.markdown("---")
        st.write(f"**Per-Fold Accuracy Variance ({model_choice} Stability Check):**")
        fold_df = pd.DataFrame({"Fold": ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"], "Accuracy": acc_scores}).set_index("Fold")
        st.line_chart(fold_df)
        st.caption("A line chart displaying the accuracy score across all 5 folds. A flatter line indicates a more stable, reliable algorithm across different data subpopulations. Compare this variance against other models to answer the final question in Notebook 3.")
