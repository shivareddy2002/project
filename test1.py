import streamlit as st
import sqlite3
import bcrypt
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib  # For saving and loading models

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier

# Create a folder for storing models
MODEL_FOLDER = "models"
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Database setup
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# Hash password
def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

# Verify password
def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password)

# Add user to database
def add_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

# Check if user exists
def user_exists(username):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT username FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    return result is not None

# Validate username
def validate_username(username):
    if len(username) < 4:
        return "Username must be at least 4 characters long."
    if not re.match("^[a-zA-Z0-9_]+$", username):
        return "Username can only contain letters, numbers, and underscores."
    return None

# Validate password
def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search("[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search("[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search("[0-9]", password):
        return "Password must contain at least one digit."
    if not re.search("[!@#$%^&*()]", password):
        return "Password must contain at least one special character."
    return None

# Initialize database
init_db()

def main():
    # Configure page settings
    st.set_page_config(
        page_title="Tomato Stress Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS styling
    st.markdown("""
    <style>
        .header {
            background: white;
            padding: 1rem;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
            position: fixed;
            width: 100%;
            top: 0;
            z-index: 1000;
        }
        .nav-link {
            margin: 0 1rem;
            color: #333;
            text-decoration: none;
        }
        .main-banner {
            padding: 6rem 0 2rem;
            background: #f8f9fa;
        }
        .service-card {
            padding: 2rem;
            margin: 1rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        }
        .footer {
            background: #333;
            color: white;
            padding: 2rem;
            text-align: center;
            margin-top: 4rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state for authentication and page routing
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "page" not in st.session_state:
        st.session_state.page = "signup"  # Start with the Sign Up page

    # Sign Up Page
    if st.session_state.page == "signup":
        st.title("Sign Up")
        new_username = st.text_input("Choose a Username", key="signup_username")
        new_password = st.text_input("Choose a Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up", key="signup_button"):
            username_error = validate_username(new_username)
            password_error = validate_password(new_password)
            
            if username_error:
                st.error(username_error)
            elif password_error:
                st.error(password_error)
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif user_exists(new_username):
                st.error("Username already exists. Please choose another.")
            else:
                add_user(new_username, new_password)
                st.success("Account created successfully! Please sign in.")
                st.session_state.page = "login"  # Redirect to the Login page
                st.rerun()

    # Login Page
    elif st.session_state.page == "login":
        st.title("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_button"):
            if user_exists(username):
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                c.execute("SELECT password FROM users WHERE username = ?", (username,))
                hashed_password = c.fetchone()[0]
                conn.close()
                if verify_password(password, hashed_password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.page = "home"  # Redirect to the Home page
                    st.rerun()
                else:
                    st.error("Invalid password")
            else:
                st.error("Username does not exist")

    # Home Page
    elif st.session_state.page == "home":
        # Create a layout with columns for the Sign Out button
        col1, col2 = st.columns([4, 1])  # Adjust the ratio as needed
        with col1:
            st.markdown("<div class='main-banner'>", unsafe_allow_html=True)
            
            # Banner section
            st.title("Welcome to Dashboard")
            st.markdown("""
            Classification and Forecasting of Water Stress in Tomato Plants 
            Using Bioristor Data
            """)
            st.image("static/123.webp", width=500)
            
            st.markdown("</div>", unsafe_allow_html=True)

        # Sign Out Button at the top-right corner
        with col2:
            if st.button("Sign Out", key="sign_out_button"):
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.page = "signup"  # Redirect to the Sign Up page
                st.rerun()

    # Machine Learning Section
    if st.session_state.authenticated and st.session_state.page == "home":
        st.title("Tomato Drought Stress Prediction App")
        st.write("This app predicts tomato drought stress using sensor data ")

        # --- Load Dataset ---
        st.header("Data Upload and Preprocessing")
        uploaded_file = st.file_uploader("Upload your Bioristor Dataset", type=["csv"])

        if uploaded_file is not None:
            dataset = pd.read_csv(uploaded_file)
            dataset.fillna(0, inplace=True)
            # st.dataframe(dataset) dataset preview

            le = LabelEncoder()
            status = dataset['status'].ravel()
            drought = dataset['y'].ravel()
            dataset.drop(['y', 'status'], axis=1, inplace=True)
            X = dataset.values

            normalized = MinMaxScaler()
            X = normalized.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(X, drought, test_size=0.2, random_state=42)

            # Reshape for MLPClassifier (2D array)
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            st.write("Dataset loaded and preprocessed successfully!")

            # --- Model Training and Evaluation ---
            st.header("Model Training and Evaluation")

            @st.cache_resource
            def load_models():
                # Check if models already exist
                if os.path.exists(os.path.join(MODEL_FOLDER, "dt_gini_model.pkl")):
                    dt_cls_gini = joblib.load(os.path.join(MODEL_FOLDER, "dt_gini_model.pkl"))
                else:
                    # Decision Tree (GINI)
                    dt_cls_gini = DecisionTreeClassifier(criterion='gini')
                    dt_cls_gini.fit(X_train, y_train)
                    joblib.dump(dt_cls_gini, os.path.join(MODEL_FOLDER, "dt_gini_model.pkl"))

                if os.path.exists(os.path.join(MODEL_FOLDER, "dt_info_model.pkl")):
                    dt_cls_info = joblib.load(os.path.join(MODEL_FOLDER, "dt_info_model.pkl"))
                else:
                    # Decision Tree (Info Gain)
                    dt_cls_info = DecisionTreeClassifier(criterion='entropy')
                    dt_cls_info.fit(X_train, y_train)
                    joblib.dump(dt_cls_info, os.path.join(MODEL_FOLDER, "dt_info_model.pkl"))

                if os.path.exists(os.path.join(MODEL_FOLDER, "rf_model.pkl")):
                    rf_cls = joblib.load(os.path.join(MODEL_FOLDER, "rf_model.pkl"))
                else:
                    # Random Forest
                    rf_cls = RandomForestClassifier(class_weight='balanced')
                    rf_cls.fit(X_train, y_train)
                    joblib.dump(rf_cls, os.path.join(MODEL_FOLDER, "rf_model.pkl"))

                if os.path.exists(os.path.join(MODEL_FOLDER, "mlp_model.pkl")):
                    mlp_model = joblib.load(os.path.join(MODEL_FOLDER, "mlp_model.pkl"))
                else:
                    # MLPClassifier (scikit-learn's Neural Network)
                    mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42, max_iter=300)
                    mlp_model.fit(X_train, y_train)
                    joblib.dump(mlp_model, os.path.join(MODEL_FOLDER, "mlp_model.pkl"))

                return dt_cls_gini, dt_cls_info, rf_cls, mlp_model

            dt_cls_gini, dt_cls_info, rf_cls, mlp_model = load_models()

            st.success("Model training and evaluation successfully completed!")

            # --- Prediction on Test Data ---
            st.header("Prediction on New Test Data")
            test_file = st.file_uploader("Upload your Test Data CSV file", type=["csv"])

            if test_file is not None:
                testData = pd.read_csv(test_file)
                
                # Ensure the test data has the same columns as the training data
                expected_columns = dataset.columns  # Columns from the training data
                if not all(col in testData.columns for col in expected_columns):
                    st.error(f"Test data must have the following columns: {expected_columns}")
                else:
                    # Preprocess the test data in the same way as the training data
                    testData.fillna(0, inplace=True)
                    testData = testData[expected_columns]  # Ensure the same column order
                    td = testData.values
                    temp_test = normalized.transform(td)  # Use the same scaler as training data
                    temp_test = temp_test.reshape(temp_test.shape[0], -1)

                    # Make predictions
                    status_predict_test = rf_cls.predict(temp_test)
                    drought_predict_test = mlp_model.predict(temp_test)

                    # Create a DataFrame for predictions
                    predictions_df = pd.DataFrame({
                        "Row": range(1, len(status_predict_test) + 1),
                        "Number of Features": [len(td[i]) for i in range(len(status_predict_test))],  # Length of features
                        "Predicted Status": status_predict_test,
                        "Drought Prediction": ["Drought Stress Detected" if pred == 1 else "No Drought Stress" for pred in drought_predict_test]
                    })

                    # Display predictions in a table format
                    st.subheader("Test Data Predictions:")
                    st.dataframe(predictions_df)  # Use st.dataframe for an interactive table

                    # Evaluate predictions (if true labels are available in the test data)
                    if 'y' in testData.columns:
                        true_labels = testData['y'].values
                        accuracy = accuracy_score(true_labels, drought_predict_test)
                        st.write(f"**Accuracy on Test Data:** {accuracy * 100:.2f}%")
                        
                        # Display confusion matrix
                        st.subheader("Confusion Matrix:")
                        cm = confusion_matrix(true_labels, drought_predict_test)
                        st.write(cm)
                    else:
                        st.warning("")
        else:
            st.info("Please upload a CSV dataset to proceed.")

if __name__ == "__main__":
    main()