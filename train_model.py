import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Step 1: Prepare Training Data
data = {
    "text": [
        "Hello", "Hi", "Hey", 
        "What's the weather like?", "Tell me the temperature", "Is it raining?",
        "Goodbye", "See you later", "Bye"
    ],
    "intent": [
        "greeting", "greeting", "greeting",
        "weather", "weather", "weather",
        "goodbye", "goodbye", "goodbye"
    ]
}

df = pd.DataFrame(data)

# Step 2: Convert Text into Numerical Representation (TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])  # Convert text to vectors
y = df["intent"]  # Target labels

# Step 3: Train the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X, y)

# Step 4: Save the Model and Vectorizer for Future Use
with open("intent_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model trained and saved successfully!")