import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier  # Import MLPClassifier for scikit-learn alternative

st.title("Tomato Drought Stress Prediction App")
st.write("This app predicts tomato drought stress using sensor data and machine learning models.")

# --- Load Dataset ---
st.header("Data Upload and Preprocessing")
uploaded_file = st.file_uploader("Upload your Tomato data CSV file", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    dataset.fillna(0, inplace=True)
    st.dataframe(dataset)

    le = LabelEncoder()
    status = dataset['status'].to_numpy()
    drought = dataset['y'].to_numpy()
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
        # Decision Tree (GINI)
        param_grid_dt_gini = {'max_leaf_nodes': [None], 'max_depth': [None]}
        dt_cls_gini = GridSearchCV(DecisionTreeClassifier(criterion='gini'), param_grid_dt_gini, cv=5)
        dt_cls_gini.fit(X_train, y_train)

        # Decision Tree (Info Gain)
        param_grid_dt_info = {'max_leaf_nodes': [None], 'max_depth': [None]}
        dt_cls_info = GridSearchCV(DecisionTreeClassifier(criterion='entropy'), param_grid_dt_info, cv=5)
        dt_cls_info.fit(X_train, y_train)

        # Random Forest
        param_grid_rf = {'n_estimators': [5, 15, 18, 100], 'max_depth': [2, 5, 7, 9]}
        rf_cls = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid_rf, cv=5)
        rf_cls.fit(X_train, y_train)

        # MLPClassifier (scikit-learn's Neural Network)
        mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42, max_iter=300)  # Example hyperparameters
        mlp_model.fit(X_train, y_train)  # Train MLPClassifier

        return dt_cls_gini, dt_cls_info, rf_cls, mlp_model

    dt_cls_gini, dt_cls_info, rf_cls, mlp_model = load_models()

    labels_status = np.unique(status)
    labels_drought = ['No Drought', 'Drought']  # Labels are now directly 0 and 1, adjust labels_drought accordingly
    algorithm_names = ["Decision Tree with Gini", "Decision Tree with Info Gain", "Random Forest", "MLPClassifier"]  # Remove CNN/LSTM, add MLP
    precision = []
    recall = []
    fscore = []
    accuracy = []

    def calculate_metrics_streamlit(algorithm, predict, testY, col):  # No changes needed in metric calculation
        p = precision_score(testY, predict, average='macro') * 100
        r = recall_score(testY, predict, average='macro') * 100
        f = f1_score(testY, predict, average='macro') * 100
        a = accuracy_score(testY, predict) * 100

        col.write(f"**{algorithm} Metrics:**")
        col.write(f"Accuracy: {a:.2f}%")
        col.write(f"Precision: {p:.2f}%")
        col.write(f"Recall: {r:.2f}%")
        col.write(f"F1-Score: {f:.2f}%")

        accuracy.append(a)
        precision.append(p)
        recall.append(r)
        fscore.append(f)

    # Evaluate Decision Tree (GINI)
    predict_dt_gini = dt_cls_gini.predict(X_test)
    col1, col2 = st.columns(2)
    calculate_metrics_streamlit("Decision Tree with Gini", predict_dt_gini, y_test, col1)

    # Evaluate Decision Tree (Info Gain)
    predict_dt_info = dt_cls_info.predict(X_test)
    calculate_metrics_streamlit("Decision Tree with Info Gain", predict_dt_info, y_test, col2)

    # Evaluate Random Forest
    predict_rf = rf_cls.predict(X_test)
    calculate_metrics_streamlit("Random Forest", predict_rf, y_test, col1)

    # Evaluate MLPClassifier
    predict_mlp = mlp_model.predict(X_test)
    calculate_metrics_streamlit("MLPClassifier", predict_mlp, y_test, col2)  # Use labels_status as MLP predicts status labels

    st.header("Algorithm Comparison")
    df = pd.DataFrame([
        ['Decision Tree with Gini', 'Precision', precision[0]], ['Decision Tree with Gini', 'Recall', recall[0]], ['Decision Tree with Gini', 'F1 Score', fscore[0]], ['Decision Tree with Gini', 'Accuracy', accuracy[0]],
        ['Decision Tree with Info Gain', 'Precision', precision[1]], ['Decision Tree with Info Gain', 'Recall', recall[1]], ['Decision Tree with Info Gain', 'F1 Score', fscore[1]], ['Decision Tree with Info Gain', 'Accuracy', accuracy[1]],
        ['Random Forest', 'Precision', precision[2]], ['Random Forest', 'Recall', recall[2]], ['Random Forest', 'F1 Score', fscore[2]], ['Random Forest', 'Accuracy', accuracy[2]],
        ['MLPClassifier', 'Precision', precision[3]], ['MLPClassifier', 'Recall', recall[3]], ['MLPClassifier', 'F1 Score', fscore[3]], ['MLPClassifier', 'Accuracy', accuracy[3]],  # Updated to MLPClassifier
    ], columns=['Parameters', 'Algorithms', 'Value'])

    st.dataframe(df.pivot(index="Parameters", columns="Algorithms", values="Value"))
    st.bar_chart(df.pivot(index="Parameters", columns="Algorithms", values="Value"))

    # --- Prediction on Test Data ---
    st.header("Prediction on New Test Data")
    test_file = st.file_uploader("Upload your Test Data CSV file", type=["csv"])

    if test_file is not None:
        testData = pd.read_csv(test_file)
        testData.fillna(0, inplace=True)
        td = testData.values
        temp_test = testData.values
        temp_test = normalized.transform(temp_test)
        temp_test = temp_test.reshape(temp_test.shape[0], -1)  # Reshape test data for MLPClassifier

        status_predict_test = rf_cls.predict(temp_test)  # Random Forest for Status
        drought_predict_test = mlp_model.predict(temp_test)  # MLPClassifier for Drought

        st.subheader("Test Data Predictions:")
        for i in range(len(status_predict_test)):
            drought_pred_text = "Drought Stress Detected" if drought_predict_test[i] == 1 else "No Drought Stress"
            st.write(f"**Test Data Row {i + 1}:** {td[i][0:20]}  \n**Predicted Status:** {status_predict_test[i]}  \n**Drought Prediction:** {drought_pred_text}")
else:
    st.info("Please upload a CSV dataset to proceed.")