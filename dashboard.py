# -*- coding: utf-8 -*-


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title and layout of the dashboard
st.set_page_config(page_title="Exam Performance Analytics Dashboard 📊", layout="wide", initial_sidebar_state="expanded")

st.title("Exam Performance Analytics Dashboard 🎓")
st.write("""
This dashboard provides a comprehensive analysis of student exam performance data,
utilizing **Descriptive**, **Diagnostic**, **Predictive**, and **Prescriptive** analytics.
""")

st.divider()

# --- Load Data ---
@st.cache_data # Cache the data loading for better performance
def load_data():
    try:
        df = pd.read_csv("cleaned_exam_data.csv")
        # Convert categorical columns to 'category' dtype for efficiency
        for col in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
            df[col] = df[col].astype('category')
        return df
    except FileNotFoundError:
        st.error("Error: 'cleaned_exam_data.csv' not found. Please ensure the file is in the same directory as the script.")
        return pd.DataFrame() # Return empty DataFrame on error

df = load_data()

if df.empty:
    st.stop() # Stop execution if data loading failed

# Define score_cols here so it's accessible globally
score_cols = ['math score', 'reading score', 'writing score']

# --- Global Data Preparation for Predictive Model ---
# Define target and features globally
target = 'math score'
features = [col for col in df.columns if col not in ['Student ID', target]]
X = df[features]
y = df[target]

# One-hot encode categorical features globally
X_encoded = pd.get_dummies(X, drop_first=True)

# Split data for training and testing once globally
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Cache the model training to run only once
@st.cache_resource
def train_model(X_train_data, y_train_data):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train_data, y_train_data)
    return model

model_gb = train_model(X_train, y_train)

# --- Sidebar Navigation ---
st.sidebar.header("Navigation")
analysis_type = st.sidebar.radio(
    "Choose Analysis Type:",
    ["Descriptive Analytics", "Diagnostic Analytics", "Predictive Analytics", "Prescriptive Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Overview")
st.sidebar.write(f"Total Students: **{len(df)}**")
st.sidebar.write(f"Columns: **{len(df.columns)}**")

# --- Main Content Area ---

# 1. Descriptive Analytics
if analysis_type == "Descriptive Analytics":
    st.header("1. Descriptive Analytics 📈")
    st.write("Understand the basic characteristics and distributions of the exam data.")

    st.subheader("Overall Score Statistics")
    st.dataframe(df[score_cols].describe().T.style.background_gradient(cmap='Blues'), use_container_width=True)
    st.write("Summary statistics for Math, Reading, and Writing scores.")

    st.subheader("Distribution of Scores")
    selected_score = st.selectbox("Select a score to visualize:", score_cols)
    fig_hist = px.histogram(df, x=selected_score, marginal="box",
                            title=f'Distribution of {selected_score.replace("_", " ").title()}',
                            color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_hist, use_container_width=True)
    st.write(f"This histogram and box plot show the spread and central tendency of {selected_score.replace('_', ' ').title()}.")

    st.subheader("Scores by Gender and Test Preparation")
    col1, col2 = st.columns(2)
    with col1:
        fig_gender = px.box(df, x='gender', y=selected_score, color='gender',
                            title=f'{selected_score.replace("_", " ").title()} by Gender',
                            color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_gender, use_container_width=True)
    with col2:
        fig_prep = px.box(df, x='test preparation course', y=selected_score, color='test preparation course',
                          title=f'{selected_score.replace("_", " ").title()} by Test Preparation Course',
                          color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_prep, use_container_width=True)
    st.write("Box plots illustrating score differences based on gender and test preparation course completion.")

    st.subheader("Correlation Matrix of Scores")
    corr_matrix = df[score_cols].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         color_continuous_scale=px.colors.sequential.RdBu,
                         title="Correlation Matrix of Exam Scores")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.write("This heatmap shows the correlation between Math, Reading, and Writing scores. Values closer to 1 indicate a strong positive correlation, and values closer to -1 indicate a strong negative correlation.")

# 2. Diagnostic Analytics
elif analysis_type == "Diagnostic Analytics":
    st.header("2. Diagnostic Analytics 🔍")
    st.write("Investigate the reasons behind observed patterns and outcomes in the data.")

    st.subheader("Impact of Parental Education on Scores")
    parent_edu_order = [
        "some high school", "high school", "some college",
        "associate's degree", "bachelor's degree", "master's degree"
    ]
    df['parental level of education'] = pd.Categorical(df['parental level of education'], categories=parent_edu_order, ordered=True)

    fig_parent_edu = px.bar(df.groupby('parental level of education')[score_cols].mean().reset_index(),
                            x='parental level of education', y=score_cols,
                            barmode='group',
                            title='Average Scores by Parental Level of Education',
                            labels={'value': 'Average Score', 'variable': 'Subject'},
                            color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_parent_edu, use_container_width=True)
    st.write("This chart shows how average scores vary with the parental level of education.")

    st.subheader("Lunch Type vs. Scores")
    # Define selected_score for Diagnostic Analytics if it's not already set by Descriptive Analytics
    # This ensures the selectbox is available even if Descriptive Analytics wasn't visited first
    if 'selected_score' not in st.session_state:
        st.session_state['selected_score'] = score_cols[0] # Default to math score

    selected_score_diagnostic = st.selectbox("Select a score to visualize for Lunch Type:", score_cols, key='diagnostic_score_select')
    fig_lunch = px.box(df, x='lunch', y=selected_score_diagnostic, color='lunch',
                       title=f'{selected_score_diagnostic.replace("_", " ").title()} by Lunch Type',
                       color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(fig_lunch, use_container_width=True)
    st.write("Examines the impact of lunch type (standard vs. free/reduced) on student scores.")

    st.subheader("Test Preparation Course Effect")
    avg_scores_prep = df.groupby('test preparation course')[score_cols].mean().reset_index()
    fig_prep_effect = px.bar(avg_scores_prep, x='test preparation course', y=score_cols,
                             barmode='group',
                             title='Average Scores for Test Preparation Course Completion',
                             labels={'value': 'Average Score', 'variable': 'Subject'},
                             color_discrete_sequence=px.colors.qualitative.Vivid)
    st.plotly_chart(fig_prep_effect, use_container_width=True)
    st.write("This chart clearly shows the average score difference between students who completed the test preparation course and those who did not.")

# 3. Predictive Analytics
elif analysis_type == "Predictive Analytics":
    st.header("3. Predictive Analytics 🔮")
    st.write("Forecast future outcomes based on historical data. Here, we predict Math Score using a Gradient Boosting Regressor.")

    st.subheader("Predict Math Score using Various Features")

    y_pred_gb = model_gb.predict(X_test)

    mae_gb = mean_absolute_error(y_test, y_pred_gb)
    r2_gb = r2_score(y_test, y_pred_gb)

    st.write(f"**Model Used:** Gradient Boosting Regressor")
    st.write(f"**Features Used:** {', '.join(X.columns)}")
    st.write(f"**Target:** Math Score")

    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae_gb:.2f}")
    st.metric(label="R-squared (R²)", value=f"{r2_gb:.2f}")
    st.write("MAE indicates the average magnitude of the errors in a set of predictions, without considering their direction. R² represents the proportion of variance in the dependent variable that can be predicted from the independent variables.")

    st.subheader("Top Feature Importances for Math Score Prediction")
    if hasattr(model_gb, 'feature_importances_'):
        feature_importances_gb = pd.Series(model_gb.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)

        plt.figure(figsize=(10, 7))
        sns.barplot(x=feature_importances_gb.head(10).values, y=feature_importances_gb.head(10).index, palette='viridis')
        plt.title('Top 10 Feature Importances for Math Score Prediction')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        st.pyplot(plt) # Display matplotlib plot in Streamlit
        plt.close() # Close the plot to free memory
        st.write("This chart highlights which features contributed most to the math score prediction.")

    st.subheader("Predicted vs. Actual Math Scores")
    plt.figure(figsize=(8, 7))
    sns.scatterplot(x=y_test, y=y_pred_gb, alpha=0.6)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Diagonal line
    plt.xlabel('Actual Math Scores')
    plt.ylabel('Predicted Math Scores')
    plt.title('Predicted vs. Actual Math Scores')
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(plt) # Display matplotlib plot in Streamlit
    plt.close() # Close the plot to free memory
    st.write("This scatter plot visualizes how well the predicted scores align with the actual scores.")

    st.subheader("Make a Prediction for a New Student")
    st.write("Enter the characteristics of a new student to predict their Math Score.")

    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        input_gender = st.selectbox("Gender:", df['gender'].unique())
        input_race = st.selectbox("Race/ethnicity:", df['race/ethnicity'].unique())
    with col_input2:
        input_parent_edu = st.selectbox("Parental Education:", df['parental level of education'].unique())
        input_lunch = st.selectbox("Lunch Type:", df['lunch'].unique())
    with col_input3:
        input_test_prep = st.selectbox("Test Prep Course:", df['test preparation course'].unique())
        input_reading = st.number_input("Reading Score:", min_value=0, max_value=100, value=70)
        input_writing = st.number_input("Writing Score:", min_value=0, max_value=100, value=65)

    if st.button("Predict Math Score for New Student"):
        new_student_data = {
            'gender': input_gender,
            'race/ethnicity': input_race,
            'parental level of education': input_parent_edu,
            'lunch': input_lunch,
            'test preparation course': input_test_prep,
            'reading score': input_reading,
            'writing score': input_writing
        }
        new_student_df = pd.DataFrame([new_student_data])

        # Ensure all columns from original features are present in new_student_df for one-hot encoding
        # This is crucial to match the columns of X_train
        new_student_processed = pd.get_dummies(new_student_df, drop_first=True)

        # Reindex to ensure all columns from X_train are present, filling missing with 0
        # This handles cases where a category might not be present in the single new student row
        new_student_processed = new_student_processed.reindex(columns=X_train.columns, fill_value=0)

        predicted_math = model_gb.predict(new_student_processed)[0]
        st.success(f"**Predicted Math Score for this student:** {predicted_math:.2f}")
        st.write("*(Note: This prediction is based on the trained Gradient Boosting Regressor model.)*")

# 4. Prescriptive Analytics
elif analysis_type == "Prescriptive Analytics":
    st.header("4. Prescriptive Analytics 💡")
    st.write("Provide actionable recommendations based on insights from descriptive, diagnostic, and predictive analyses.")

    st.subheader("Key Findings & Targeted Recommendations")

    st.markdown("""
    Based on the deeper diagnostic insights and the predictive model for math scores, we can formulate more targeted prescriptive actions:

    * **Test Preparation Course Impact:**
        * **Finding:** Students who **completed** the test preparation course consistently show **higher average scores** across all subjects compared to those who did not.
        * **Recommendation:** Actively **promote and encourage** students to enroll in and complete the test preparation course. For students identified as 'low performers' (e.g., math score < 40) who did NOT complete test preparation, a mandatory or highly incentivized math-focused test prep program should be implemented.

    * **Parental Education Influence:**
        * **Finding:** There is a **positive correlation** between the parental level of education and student exam scores. Students with parents holding higher degrees tend to perform better.
        * **Recommendation:** Implement **parental engagement programs** focusing on educational support and resources, especially for parents with lower educational backgrounds. Provide workshops on how parents can support their children's learning at home.

    * **Lunch Type Disparity:**
        * **Finding:** Students receiving **free/reduced lunch** tend to have **lower average scores** than those with standard lunch. This could indicate socioeconomic factors impacting performance.
        * **Recommendation:** Provide **targeted academic support and resources** (e.g., tutoring, after-school programs, access to learning materials) for students from lower socioeconomic backgrounds. Ensure adequate nutrition and a supportive learning environment for all students.

    * **Inter-Subject Correlation & Feature Importance:**
        * **Finding:** Math, Reading, and Writing scores are **highly correlated**. The feature importance analysis (in Predictive Analytics) reveals which specific factors most influence math scores.
        * **Recommendation:** If 'reading score' or 'writing score' are high importance features for math prediction, prescriptive action could be to focus on improving foundational literacy skills (reading comprehension, clear writing) as these may indirectly boost math performance. Tailor interventions based on the most impactful features identified by the model.

    * **Individualized Learning Plans:**
        * **Finding:** Even with general trends, there's significant variability in individual student performance.
        * **Recommendation:** Utilize **individual student data** to identify specific areas of weakness and provide **personalized learning plans or interventions**. For any student whose predicted math score falls significantly below their current performance or a desired benchmark, an individualized learning plan should be developed, focusing on their specific math weaknesses and leveraging high-impact factors identified by feature importance.

    """)

    st.subheader("Simulate Targeted Intervention for a New Student")
    st.write("Enter details for a hypothetical student to see predicted score and specific recommendations.")

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        sim_gender = st.selectbox("Simulate Gender:", df['gender'].unique(), key='sim_gender')
        sim_race = st.selectbox("Simulate Race/Ethnicity:", df['race/ethnicity'].unique(), key='sim_race')
    with col_sim2:
        sim_parent_edu = st.selectbox("Simulate Parental Education:", df['parental level of education'].unique(), key='sim_parent_edu')
        sim_lunch = st.selectbox("Simulate Lunch Type:", df['lunch'].unique(), key='sim_lunch')
    with col_sim3:
        sim_test_prep = st.selectbox("Simulate Test Prep Course:", df['test preparation course'].unique(), key='sim_test_prep')
        sim_reading = st.number_input("Simulate Reading Score:", min_value=0, max_value=100, value=50, key='sim_reading')
        sim_writing = st.number_input("Simulate Writing Score:", min_value=0, max_value=100, value=55, key='sim_writing')

    if st.button("Get Targeted Recommendations"):
        sim_student_data = {
            'gender': sim_gender,
            'race/ethnicity': sim_race,
            'parental level of education': sim_parent_edu,
            'lunch': sim_lunch,
            'test preparation course': sim_test_prep,
            'reading score': sim_reading,
            'writing score': sim_writing
        }
        sim_student_df = pd.DataFrame([sim_student_data])

        # Re-run the encoding and reindexing to match the training data's columns
        # Use the globally defined X_encoded.columns for reindexing
        sim_student_processed = pd.get_dummies(sim_student_df, drop_first=True)
        sim_student_processed = sim_student_processed.reindex(columns=X_encoded.columns, fill_value=0)

        # Use the globally trained model_gb
        predicted_math_score = model_gb.predict(sim_student_processed)[0]
        st.write(f"\n**Predicted Math Score for this simulated student:** {predicted_math_score:.2f}")

        st.subheader("Targeted Prescriptive Actions:")
        if predicted_math_score < 60: # Threshold for targeted math intervention
            st.warning("This student is predicted to have a low math score and exhibits characteristics that may require additional support.")
            st.markdown("""
            * **Recommendation 1:** Enroll in a specialized math workshop focusing on foundational skills (e.g., algebra basics, problem-solving strategies).
            * **Recommendation 2:** Provide one-on-one mentorship with a focus on improving math concepts and building confidence.
            * **Recommendation 3:** Connect family with community resources for math educational support and study groups.
            """)
        else:
            st.info("This student is predicted to perform adequately in math.")
            st.markdown("""
            * **Recommendation 1:** Encourage enrollment in advanced math electives or participation in math competitions to further challenge and develop their skills.
            * **Recommendation 2:** Consider opportunities for peer tutoring where stronger students can help others, reinforcing their own understanding.
            * **Recommendation 3:** Explore STEM-related extracurricular activities to foster continued interest in quantitative fields.
            """)
    st.markdown("---")
