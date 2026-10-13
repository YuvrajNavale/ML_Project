import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Student Risk Predictor", page_icon="🎓", layout="wide")

SUBJECTS = ["DBMS", "ML", "AI", "CV"]
FEATURES_GENERAL = ["Prior_GPA", "Study_Hours_Week"]

@st.cache_data
def make_dataset():
    """Recreates the synthetic dataset described in the project notebook."""
    np.random.seed(42)
    n = 300
    aptitude = np.random.normal(loc=0.0, scale=1.0, size=n)
    student_ids = [f"STU_{i:03d}" for i in range(1, n + 1)]
    prior_gpa = np.clip(
        np.round(6.8 + 1.25 * aptitude + np.random.normal(0, 0.45, n), 2), 4.0, 10.0
    )
    study_hours = np.clip(
        np.round(15 + 4.5 * aptitude + np.random.normal(0, 2.0, n)), 3, 35
    ).astype(int)

    params = {
        "DBMS": {"att_base": 77, "mark_base": 64},
        "ML":   {"att_base": 73, "mark_base": 58},
        "AI":   {"att_base": 75, "mark_base": 61},
        "CV":   {"att_base": 71, "mark_base": 55},
    }
    data = {
        "Student_ID": student_ids,
        "Prior_GPA": prior_gpa,
        "Study_Hours_Week": study_hours,
    }
    targets = {}
    for subject, cfg in params.items():
        att = np.clip(
            np.round(cfg["att_base"] + 10 * aptitude + np.random.normal(0, 7.5, n)),
            35, 100
        ).astype(int)
        marks = np.clip(
            np.round(cfg["mark_base"] + 14 * aptitude + np.random.normal(0, 8.5, n)),
            12, 98
        ).astype(int)
        assign = np.clip(
            np.round(3.6 + 0.9 * aptitude + np.random.normal(0, 0.6, n)), 0, 5
        ).astype(int)

        data[f"Att_{subject}"] = att
        data[f"Marks_{subject}"] = marks
        data[f"Assign_{subject}"] = assign

        att_deficit = np.maximum(0, 75 - att)
        mark_deficit = np.maximum(0, 40 - marks)
        logit = (
            0.18 * mark_deficit
            + 0.12 * att_deficit
            - 0.60 * assign
            - 0.40 * (prior_gpa - 6.5)
            - 0.05 * (study_hours - 15)
            - 1.8
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        targets[f"Risk_{subject}"] = (prob >= 0.5).astype(int)

    return pd.DataFrame({**data, **targets})

@st.cache_resource
def train_models(df):
    models, scalers, metrics = {}, {}, []
    for subject in SUBJECTS:
        cols = FEATURES_GENERAL + [
            f"Att_{subject}", f"Marks_{subject}", f"Assign_{subject}"
        ]
        X, y = df[cols], df[f"Risk_{subject}"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:, 1]
        metrics.append({
            "Subject": subject,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-Score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob) if y_test.nunique() == 2 else np.nan,
        })
        models[subject], scalers[subject] = model, scaler
    return models, scalers, pd.DataFrame(metrics)

st.title("🎓 Student Subject-Wise Academic Risk Prediction")
st.caption("Logistic Regression • DBMS, Machine Learning, Artificial Intelligence and Computer Vision")
st.info("This prototype uses the synthetic 300-student dataset described in your project. Predictions are educational demonstrations, not validated decisions about real students.")

df = make_dataset()
models, scalers, metrics_df = train_models(df)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page", ["Predict Risk", "Dataset Explorer", "Model Performance", "About Project"])
    st.divider()
    st.metric("Dataset records", len(df))
    st.metric("Subjects", len(SUBJECTS))

if page == "Predict Risk":
    st.subheader("Predict a student's subject risk")
    st.write("Enter academic and engagement details, then select **Predict risk**.")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            student_id = st.text_input("Student ID (optional)", value="NEW_STUDENT")
            prior_gpa = st.slider("Prior GPA", min_value=4.0, max_value=10.0, value=7.2, step=0.1)
            study_hours = st.slider("Study hours per week", min_value=3, max_value=35, value=16)
        with c2:
            subject = st.selectbox("Subject to assess", SUBJECTS)
            attendance = st.slider("Attendance (%)", min_value=35, max_value=100, value=68)
            marks = st.slider("Midterm marks (0–100)", min_value=0, max_value=100, value=42)
            assignments = st.slider("Assignments submitted (0–5)", min_value=0, max_value=5, value=2)
        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if submitted:
        cols = FEATURES_GENERAL + [f"Att_{subject}", f"Marks_{subject}", f"Assign_{subject}"]
        row = pd.DataFrame([[
            prior_gpa, study_hours, attendance, marks, assignments
        ]], columns=cols)
        scaled = scalers[subject].transform(row)
        model = models[subject]
        risk_probability = float(model.predict_proba(scaled)[0, 1])
        prediction = int(risk_probability >= 0.5)
        left, right = st.columns([1, 1])
        with left:
            st.subheader(f"Prediction for {student_id or 'student'}")
            if prediction:
                st.error("⚠️ At Risk")
            else:
                st.success("✅ Not At Risk")
            st.metric("Estimated risk probability", f"{risk_probability:.1%}")
            st.progress(min(max(risk_probability, 0.0), 1.0))
            st.caption("Decision threshold: 50%. Probability is the model's estimate, not a certainty.")
        with right:
            st.subheader("Input summary")
            st.dataframe(pd.DataFrame({
                "Input": ["Subject", "Prior GPA", "Study hours/week", "Attendance", "Midterm marks", "Assignments"],
                "Value": [subject, prior_gpa, study_hours, f"{attendance}%", marks, assignments]
            }), hide_index=True, use_container_width=True)

        st.subheader("All-subject risk overview")
        all_results = []
        for sub in SUBJECTS:
            sub_cols = FEATURES_GENERAL + [f"Att_{sub}", f"Marks_{sub}", f"Assign_{sub}"]
            sub_row = pd.DataFrame([[
                prior_gpa, study_hours, attendance, marks, assignments
            ]], columns=sub_cols)
            p = float(models[sub].predict_proba(scalers[sub].transform(sub_row))[0, 1])
            all_results.append({
                "Subject": sub,
                "Prediction": "At Risk" if p >= 0.5 else "Not At Risk",
                "Risk probability": p,
            })
        result_df = pd.DataFrame(all_results)
        result_df["Risk probability"] = result_df["Risk probability"].map(lambda x: f"{x:.1%}")
        st.dataframe(result_df, hide_index=True, use_container_width=True)
        st.caption("For the all-subject overview, the same attendance, marks and assignment values are applied to each subject for demonstration. For a meaningful comparison, enter subject-specific values separately.")

elif page == "Dataset Explorer":
    st.subheader("Synthetic student dataset")
    st.write("300 records with general features, subject-wise academic features, and four target labels.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(df))
    c2.metric("Input columns", 15)
    c3.metric("Risk labels", 4)
    st.dataframe(df, use_container_width=True, height=420)
    st.download_button(
        "Download dataset as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_subject_risk_300.csv",
        mime="text/csv",
    )
    st.subheader("At-risk distribution by subject")
    counts = pd.DataFrame({
        "Subject": SUBJECTS,
        "At-risk students": [int(df[f"Risk_{s}"].sum()) for s in SUBJECTS],
    })
    counts["Percentage"] = counts["At-risk students"] / len(df) * 100
    st.bar_chart(counts.set_index("Subject")["At-risk students"])
    st.dataframe(counts, hide_index=True, use_container_width=True)

elif page == "Model Performance":
    st.subheader("Test-set performance")
    st.caption("Logistic Regression per subject • 80/20 stratified split • StandardScaler fitted on training data")
    display = metrics_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
        display[col] = display[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.subheader("Metric comparison")
    chart_df = metrics_df.set_index("Subject")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    st.bar_chart(chart_df)
    st.warning("The dataset is synthetic and the target labels are generated from rules that use related input features. High scores therefore do not establish real-world generalization. Evaluate on independent, real, appropriately consented data before any practical use.")
    st.markdown("""
    **Why ROC-AUC is useful here:** the at-risk class is relatively uncommon, so accuracy alone can be misleading. ROC-AUC measures how well the model ranks at-risk students above not-at-risk students across decision thresholds. Also inspect recall and precision to understand missed at-risk students and false alarms.
    """)

else:
    st.subheader("About the project")
    st.markdown("""
    **Aim:** Identify students who may be at risk of underperforming in DBMS, ML, AI and CV so educators can consider timely support.

    **Dataset:** 300 synthetic student records. General inputs include prior GPA and weekly study hours. Each subject has attendance, midterm marks and assignment submissions.

    **Methodology**
    1. Generate/load the dataset.
    2. Split each subject's data into 80% training and 20% testing using stratification.
    3. Standardize features using training data.
    4. Train a separate Logistic Regression model for each subject.
    5. Report Accuracy, Precision, Recall, F1-score and ROC-AUC.

    **Important limitation:** This dataset is synthetic. Treat predictions as a prototype demonstration, not as a definitive assessment of any student.
    """)
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
