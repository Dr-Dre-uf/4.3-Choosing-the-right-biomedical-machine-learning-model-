import streamlit as st
import pandas as pd
import numpy as np
import psutil
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- MONITORING UTILITY ---
def display_system_monitor():
    """Tracks CPU and RAM usage to show the cost of Deep Learning training."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Watch this spike when you click 'Train Model'.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Memory currently used by the app and dataset.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# Navigation
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Clinical Scenario",
    "Activity 2: Base Performance & Accuracy",
    "Activity 3: Advanced Clinical Metrics",
    "Activity 4: CNN vs Decision Tree"
], help="Navigate through the module activities.")

display_system_monitor()

@st.cache_data
def load_data():
    try:
        # Attempt to load local file
        df = pd.read_csv("diabetes.csv")
    except:
        # Fallback to sklearn dataset if file not found
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        # Create a synthetic binary outcome for classification (1 = Death, 0 = Survival)
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df.drop(columns='target', inplace=True)
    return df

df = load_data()

# ==========================================
# ACTIVITY 1
# ==========================================
if activity == "Activity 1: Clinical Scenario":
    st.title("Activity 1: Applied Fundamentals of ML")
    st.info("**Overview:** This section outlines the clinical context and raw data layout of the Deep Learning task.")
    
    st.header("Clinical Scenario")
    st.write("""
    This demo showcases a **1D Convolutional Neural Network (CNN)** designed to predict **in-hospital mortality** using data from the eICU Collaborative Research Database. 
    
    The dataset includes patient demographics and selected lab results such as glucose, creatinine, and potassium.
    """)
    
    st.markdown("### Dataset Class Distribution")
    st.write("Notice the heavy class imbalance. The vast majority of records represent 'Survival'. This will impact how we evaluate the model in later activities.")
    
    # Visualizing the Data Distribution
    class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
    st.bar_chart(class_counts, color="#FF4B4B")

    with st.expander("Explore the Dataset Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption("Target Variable: 'Outcome' (0 = Survival, 1 = Death)")

    st.subheader("Task Details")
    st.markdown("""
    - **Job Task:** Binary Classification.
    - **CNN Advantage:** 1D CNNs can identify complex sequential or spatial relationships between different lab values and demographics without requiring extensive manual feature engineering.
    """)

# ==========================================
# ACTIVITY 2
# ==========================================
elif activity == "Activity 2: Base Performance & Accuracy":
    st.title("Activity 2: Evaluating Base Performance")
    
    st.success("**Demo Focus:** Adjust basic hyperparameters and train the model. Watch the training graphs update to observe how the network learns over time.")

    # Interactive Sidebar for Hyperparameters
    st.sidebar.header("Model Hyperparameters")
    epochs = st.sidebar.slider("Epochs", 10, 50, 20, help="Number of times the model sees the entire dataset.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32, 64], value=16, help="Number of samples processed before the model updates its internal weights.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Train the CNN Model")
        if st.button("Train Single Fold", help="Trains the model on an 80/20 split of the data and logs the training history."):
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            # Simple split
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).reshape(len(X_train), X.shape[1], 1)
            X_val = scaler.transform(X_val).reshape(len(X_val), X.shape[1], 1)
            
            model = Sequential([
                Input(shape=(X.shape[1], 1)),
                Conv1D(32, kernel_size=2, activation='relu'),
                Flatten(),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            with st.spinner("Training model... Check the CPU monitor on the left."):
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
            
            y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            acc = (tp+tn)/(tp+tn+fp+fn)
            
            st.session_state['act2_acc'] = acc
            st.session_state['act2_history'] = history.history
            st.session_state['act2_trained'] = True
            st.success("Training Complete")
            
    with col2:
        st.subheader("Training Metrics")
        if st.session_state.get('act2_trained', False):
            # Graphing the training loss and accuracy
            hist_df = pd.DataFrame({
                'Train Accuracy': st.session_state['act2_history']['accuracy'],
                'Val Accuracy': st.session_state['act2_history']['val_accuracy'],
            })
            
            # Removed the `help` argument here to fix the TypeError
            st.line_chart(hist_df)
            st.caption("Notice how accuracy converges over epochs.")
            
            st.metric("Final Total Accuracy", f"{st.session_state['act2_acc']*100:.1f}%", help="Percentage of overall correct predictions.")
            
            st.warning("**Note on Total Accuracy:** Because our dataset is highly imbalanced (mostly 'Survivals'), a model that simply predicts 'Survival' for everyone will achieve high total accuracy but miss 100% of the at-risk patients. Because of this, accuracy is an insufficient metric for clinical deployment.")

# ==========================================
# ACTIVITY 3
# ==========================================
elif activity == "Activity 3: Advanced Clinical Metrics":
    st.title("Activity 3: Sensitivity, Specificity & Precision")
    
    st.info("**Demo Focus:** Run the evaluation, then dynamically adjust the **Decision Threshold** to visualize the inverse relationship between Sensitivity and Specificity.")

    st.subheader("5-Fold Cross Validation Evaluation")
    if st.button("Run Full K-Fold Evaluation", help="Executes a 5-Fold cross validation to gather robust probabilities."):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        progress_bar = st.progress(0)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx]).reshape(len(train_idx), X.shape[1], 1)
            X_val = scaler.transform(X[val_idx]).reshape(len(val_idx), X.shape[1], 1)
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = Sequential([
                Input(shape=(X.shape[1], 1)),
                Conv1D(32, kernel_size=2, activation='relu'),
                Flatten(),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
            
            y_prob = model.predict(X_val, verbose=0)
            st.session_state[f'probs_fold_{fold}'] = (y_val, y_prob)
            progress_bar.progress((fold + 1) / 5)
            
        st.session_state['cv_done'] = True
        st.success("Evaluation complete.")

    if st.session_state.get('cv_done', False):
        st.markdown("---")
        # Interactive threshold mapping
        threshold = st.slider("Probability Decision Threshold", 0.05, 0.95, 0.50, 0.05, 
                              help="Lowering the threshold makes the model more sensitive (catches more at-risk patients) but increases false positives.")
        
        metrics_list = []
        for fold in range(5):
            y_val, y_prob = st.session_state[f'probs_fold_{fold}']
            y_pred = (y_prob > threshold).astype(int).flatten()
            
            # Handle edge cases where tn, fp, fn, tp might not unpack perfectly if only one class is predicted
            if len(np.unique(y_val)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0 # Fallback for safety in edge cases
                
            metrics_list.append({
                'Sensitivity (Recall)': tp/(tp+fn) if (tp+fn)>0 else 0,
                'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
                'Precision': tp/(tp+fp) if (tp+fp)>0 else 0
            })
            
        res_df = pd.DataFrame(metrics_list)
        avg_df = res_df.mean()
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### Trade-off Metrics")
            st.metric("Avg Sensitivity", f"{avg_df['Sensitivity (Recall)']:.3f}", help="True Positive Rate: The proportion of actual deaths we successfully predicted.")
            st.metric("Avg Specificity", f"{avg_df['Specificity']:.3f}", help="True Negative Rate: The proportion of actual survivals we correctly predicted.")
            st.metric("Avg Precision", f"{avg_df['Precision']:.3f}", help="Positive Predictive Value: When we predict death, how often are we right?")
            
        with col2:
            st.markdown("### Visual Metrics Comparison")
            # Create a dataframe for the bar chart
            chart_data = pd.DataFrame({
                "Score": [avg_df['Sensitivity (Recall)'], avg_df['Specificity'], avg_df['Precision']]
            }, index=["Sensitivity", "Specificity", "Precision"])
            
            st.bar_chart(chart_data)
            st.caption("Notice how sliding the threshold inversely affects Sensitivity vs Specificity.")

# ==========================================
# ACTIVITY 4
# ==========================================
elif activity == "Activity 4: CNN vs Decision Tree":
    st.title("Activity 4: Model Comparison")
    
    st.info("**Demo Focus:** Review the quantifiable trade-offs between model architectures for clinical settings.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Decision Trees (Previous Model)")
        st.write("""
        - **Interpretability:** Highly interpretable. Doctors can easily follow the logic.
        - **Performance:** Often plateaus on highly complex, high-dimensional temporal data.
        - **Feature Engineering:** Requires significant manual effort from clinicians to create good features.
        """)
    with col2:
        st.markdown("### 1D CNNs (Current Model)")
        st.write("""
        - **Interpretability:** Functions as a 'Black Box'. Hard to mathematically explain the prediction.
        - **Performance:** Usually superior on raw, complex datasets. Capable of finding hidden non-linear patterns.
        - **Feature Engineering:** Learns automatic representations via convolutional filters.
        """)
    
    st.markdown("---")
    st.subheader("Comparative Scoring Matrix")
    
    # Grouped Bar chart showing the comparison visually
    comp_df = pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automated Feature Extraction'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric')
    
    # Removed the `help` argument here to fix the TypeError
    st.bar_chart(comp_df)
    st.caption("A visual representation of model trade-offs. Scored 1-10.")
        
    st.error("**Clinical AI Reality:** Interpretability is often a regulatory requirement in clinical AI deployment. While CNNs may have better overall metrics (like Sensitivity), a hospital might be forced to choose a Decision Tree if they cannot legally or ethically deploy a 'black box' model without explainability features.")
