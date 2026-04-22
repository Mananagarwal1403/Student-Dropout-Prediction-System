import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)


menu = st.sidebar.selectbox("Menu", ["Prediction", "Model Comparison"])

if menu == "Prediction":

    model = joblib.load("./model.pkl")

    st.markdown("""
        <style>
        .main {
            background-color: #0f172a;
            color: white;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
        .card {
            padding: 20px;
            border-radius: 15px;
            background-color: #1e293b;
            box-shadow: 0 0 10px rgba(0,0,0,0.4);
            margin-bottom: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Student Dropout Prediction System")
    st.markdown("### Predict student dropout risk using Machine Learning")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("👤 Personal Details")

        age = st.number_input("Age", 15, 60, 20)
        gender = st.selectbox("Gender (0 = Female, 1 = Male)", [0, 1])
        attendance = st.selectbox("Daytime Attendance", [0, 1])
        scholarship = st.selectbox("Scholarship Holder", [0, 1])

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📚 Academic Performance")

        gpa1 = st.slider("1st Semester Grade", 0.0, 20.0, 10.0)
        gpa2 = st.slider("2nd Semester Grade", 0.0, 20.0, 10.0)

        approved1 = st.number_input("Subjects Approved (Sem 1)", 0, 10, 5)
        approved2 = st.number_input("Subjects Approved (Sem 2)", 0, 10, 5)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Economic Indicators")

    unemployment = st.slider("Unemployment Rate", 0.0, 20.0, 5.0)
    inflation = st.slider("Inflation Rate", 0.0, 20.0, 2.5)
    gdp = st.slider("GDP", -10.0, 10.0, 1.0)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Predict Dropout Risk"):

        input_data = [[
            1, attendance, 1, 1, 1, 1, 1,
            1, 0, 0, 1, gender, scholarship,
            age, 0,
            5, 5, 5, approved1, gpa1, 0,
            5, 5, 5, approved2, gpa2, 0,
            unemployment, inflation, gdp
        ]]

        columns = [
            'Marital status', 'Daytime/evening attendance',
            'Previous qualification', "Mother's qualification",
            "Father's qualification", "Mother's occupation", "Father's occupation",
            'Displaced', 'Educational special needs', 'Debtor',
            'Tuition fees up to date', 'Gender', 'Scholarship holder',
            'Age at enrollment', 'International',
            'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
            'Curricular units 2nd sem (without evaluations)',
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]

        df = pd.DataFrame(input_data, columns=columns)

        prediction = model.predict(df)

        st.markdown("## 🎯 Prediction Result")

        if prediction[0] == 1:
            st.error("⚠️ High Risk of Dropout")
        else:
            st.success("✅ Low Risk — Student likely to continue")

        prob = model.predict_proba(df)[0][1]

        st.progress(prob)
        st.write(f"Dropout Probability: {prob:.2%}")


# -------------------------------
# NEW PAGE: MODEL COMPARISON (GRAPH ONLY)
# -------------------------------
elif menu == "Model Comparison":

    import matplotlib.pyplot as plt

    st.title("📊 Model Comparison")

    # Load results
    results_df = pd.read_csv("./model_results.csv")

    # Graph only
    fig = plt.figure()
    plt.bar(results_df["Model"], results_df["Accuracy"])
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0.75, 0.9)

    st.pyplot(fig)