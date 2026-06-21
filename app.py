import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title("📄 Document Classifier")
st.write("Classify documents using Machine Learning")
st.markdown("### AI-Powered Text Classification System")
st.info("Enter any text document and click Classify.")

# Load dataset
data = fetch_20newsgroups(subset='train')

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(data.data)

# Train model
model = MultinomialNB()
model.fit(X, data.target)

# User Input
user_text = st.text_area("Enter your document text")

if st.button("Classify"):
    transformed = vectorizer.transform([user_text])
    prediction = model.predict(transformed)

    st.success(
        f"Predicted Category: {data.target_names[prediction[0]]}"
    )