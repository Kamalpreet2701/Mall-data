# app.py
import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# ===========================
# Load Model & Vectorizer
# ===========================
with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# File to store user feedbacks
CSV_FILE = "mall_feedback.csv"

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Mall Feedback Sentiment App", page_icon="🛍️", layout="centered")

st.title("🛍️ Mall Customer Feedback System")
st.write("Enter your details and feedback to check whether it's Positive or Negative.")

# Input Form
with st.form("feedback_form"):
    name = st.text_input("Enter your Name")
    gender = st.selectbox("Select Gender", ["Male", "Female", "Other"])
    age = st.number_input("Enter Age", min_value=10, max_value=100, step=1)
    email = st.text_input("Enter your Email")
    feedback = st.text_area("Write your Feedback here...")

    submitted = st.form_submit_button("Submit Feedback")

# ===========================
# Prediction & Save Feedback
# ===========================
if submitted:
    if feedback.strip() == "":
        st.warning("⚠️ Please enter feedback before submitting.")
    else:
        # Convert feedback into vector
        feedback_vec = vectorizer.transform([feedback])

        # Prediction
        prediction = model.predict(feedback_vec)[0]
        sentiment = "✅ Positive" if prediction == 1 else "❌ Negative"

        # Show result
        st.subheader(f"Sentiment: {sentiment}")

        # Save into CSV
        new_entry = pd.DataFrame([[name, gender, age, email, feedback, sentiment]],
                                 columns=["Name", "Gender", "Age", "Email", "Feedback", "Sentiment"])

        if os.path.exists(CSV_FILE):
            old = pd.read_csv(CSV_FILE)
            updated = pd.concat([old, new_entry], ignore_index=True)
        else:
            updated = new_entry

        updated.to_csv(CSV_FILE, index=False)
        st.success("🎉 Your feedback has been recorded successfully!")

# ===========================
# Feedback Reports
# ===========================
if os.path.exists(CSV_FILE):
    st.markdown("---")
    st.subheader("📊 Feedback Reports")

    data = pd.read_csv(CSV_FILE)

    if not data.empty:
        st.write("### Recent Feedbacks")
        st.write(data.tail(5))   # show last 5 feedbacks

        # Sentiment Counts
        sentiment_counts = data["Sentiment"].value_counts()

        # Bar Chart
        st.write("### Sentiment Bar Chart")
        st.bar_chart(sentiment_counts)

        # Pie Chart
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        sentiment_counts.plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)
