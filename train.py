import json
import pickle
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# 1. Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab') # Added for compatibility with newer NLTK versions

# 2. Load Data
print("Loading data...")
with open('intents.json', 'r') as file:
    data = json.load(file)

patterns = []
tags = []

# 3. Organize Data
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# 4. Convert Text to Numbers (Vectorization)
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize, stop_words=None)
X = vectorizer.fit_transform(patterns)

# 5. Train the Model
print("Training model...")
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X, tags)

# 6. Save the Model and Vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Success! 'model.pkl' and 'vectorizer.pkl' created.")