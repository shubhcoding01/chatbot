# 🤖 AI Smart Chatbot (Python + NLP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![ML](https://img.shields.io/badge/AI-Scikit--Learn-orange)

An intelligent, multi-language AI Chatbot built using **Python** and **Machine Learning**. It uses Natural Language Processing (NLP) for intent classification and integrates with **Wikipedia** for real-time information retrieval.

## 🚀 Key Features
* **🧠 Intent Classification:** Uses `RandomForestClassifier` and `NLTK` to understand user intent (not just keyword matching).
* **🌐 Multi-Language Support:** Supports 50+ languages (Hindi, Spanish, French, etc.) using deep-translation layers.
* **📚 Hybrid Intelligence:** Combines predefined training data with **Wikipedia API** for answering general knowledge questions.
* **💾 Conversation History:** Automatically saves chats to an **SQLite Database**.
* **scra Export Options:** Users can download chat logs in **CSV, Excel, or PDF** formats.
* **🖥️ Pure Python UI:** Built entirely using **Streamlit**, requiring no HTML/CSS knowledge.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **ML/NLP:** Scikit-Learn, NLTK, CountVectorizer (Bag of Words)
* **External APIs:** Wikipedia, Deep Translator
* **Database:** SQLite3
* **Data Handling:** Pandas, FPDF, OpenPyXL

---

## 📂 Project Structure
```text
my_chatbot/
│
├── app.py                # Main Application (UI & Logic)
├── train.py              # Training Script (ML Model Generator)
├── intents.json          # Training Data (Patterns & Responses)
│
├── model.pkl             # Saved ML Model (Auto-generated)
├── vectorizer.pkl        # Saved NLP Vectorizer (Auto-generated)
└── database.db           # Chat History Storage (Auto-generated)

---

## ⚙️ Installation & Setup

1. Clone or Download the Repository
Navigate to the project folder in your terminal.

2. Install Dependencies
Run the following command to install all required libraries:

Bash

pip install streamlit nltk scikit-learn pandas openpyxl fpdf wikipedia deep-translator
3. Train the Model
Before running the bot, you must train the AI model. Run this once:

Bash

python train.py
You should see a message: ✅ Success! 'model.pkl' created.

▶️ How to Run
Start the web interface using the following command:

Bash

python -m streamlit run app.py
The application will automatically open in your default web browser at http://localhost:8501.

🧠 How it Works (Architecture)
Input Processing: The user inputs text (in any language).

Translation Layer: If the input is not English, it is translated to English using GoogleTranslator.

Intent Prediction:

The input is vectorized (converted to numbers).

The Random Forest Model predicts the "tag" (e.g., greeting, pricing, python_intro).

Confidence Check: If the confidence score is low (< 50%), the bot switches to "Search Mode".

Response Generation:

Small Talk: Returns a predefined response from intents.json.

Knowledge Query: Fetches a summary from Wikipedia.

Output: The response is translated back to the user's selected language and displayed.