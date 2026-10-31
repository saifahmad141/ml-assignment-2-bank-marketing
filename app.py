import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦", layout="wide")
st.title("Bank Marketing Subscription Prediction")
st.write(
    "Upload the held-out test CSV generated in the BITS Virtual Lab, select a trained "
    "classifier, and review its evaluation metrics. The app loads saved models; it does "
    "not retrain them on uploaded data."
)

MODEL_DIR = Path(__file__).parent / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors (KNN)": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine (SVM)": "support_vector_machine.joblib",
}

uploaded_file = st.file_uploader("Upload test data (CSV)", type=["csv"])
selected_model = st.selectbox("Select a trained model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.subheader("Uploaded test data")
        st.write(f"Rows: {data.shape[0]} | Columns: {data.shape[1]}")
        st.dataframe(data.head(), use_container_width=True)

        if "y" not in data.columns:
            st.error("The CSV must include the target column named 'y'.")
            st.stop()

        model_path = MODEL_DIR / MODEL_FILES[selected_model]
        if not model_path.exists():
            st.error(
                f"Saved model not found: {model_path.name}. Add all five .joblib model "
                "files to the repository's model/ folder before deployment."
            )
            st.stop()

        y_raw = data["y"]
        if y_raw.dtype == object:
            y_text = y_raw.astype(str).str.strip().str.lower()
            mapping = {"no": 0, "yes": 1, "0": 0, "1": 1}
            y_true = y_text.map(mapping)
        else:
            y_true = pd.to_numeric(y_raw, errors="coerce")

        if y_true.isna().any() or not set(y_true.unique()).issubset({0, 1}):
            st.error("Target column 'y' must contain only no/yes or 0/1 values.")
            st.stop()

        X_test = data.drop(columns=["y"])
        model = joblib.load(model_path)

        expected_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
        if expected_features:
            missing = sorted(set(expected_features) - set(X_test.columns))
            extra = sorted(set(X_test.columns) - set(expected_features))
            if missing:
                st.error(f"Missing required input columns: {', '.join(missing)}")
                st.stop()
            # Keep feature order consistent with training.
            X_test = X_test[expected_features]

        y_pred = model.predict(X_test)
        y_true = y_true.astype(int)

        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        metrics["AUC"] = (
            roc_auc_score(y_true, y_score)
            if y_score is not None and len(set(y_true)) == 2
            else float("nan")
        )

        st.subheader(f"Evaluation: {selected_model}")
        cols = st.columns(6)
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(
                confusion_matrix=confusion_matrix(y_true, y_pred, labels=[0, 1]),
                display_labels=["No (0)", "Yes (1)"]
            ).plot(ax=ax, colorbar=False)
            st.pyplot(fig)
            plt.close(fig)

        with right:
            st.subheader("Classification report")
            report = classification_report(
                y_true, y_pred, labels=[0, 1],
                target_names=["No (0)", "Yes (1)"],
                output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    except Exception as exc:
        st.error(f"Could not evaluate this file: {exc}")
else:
    st.info("Upload the test_data.csv file exported from your BITS Virtual Lab notebook to begin.")
