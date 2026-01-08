# import streamlit as st
# import pickle
# import json
# import random
# import sqlite3
# import pandas as pd
# from fpdf import FPDF
# from datetime import datetime
# import io
# import os
# import nltk
# nltk.download('punkt')

# # --- PAGE SETUP ---
# st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
# st.title("🤖 Python AI Assistant")

# # --- 1. LOAD MODEL ---
# @st.cache_resource
# def load_resources():
#     # Load the saved model and vectorizer
#     if not os.path.exists('model.pkl'):
#         return None, None, None
    
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     with open('vectorizer.pkl', 'rb') as f:
#         vectorizer = pickle.load(f)
#     with open('intents.json', 'r') as f:
#         intents = json.load(f)
#     return model, vectorizer, intents

# model, vectorizer, intents = load_resources()

# if model is None:
#     st.error("Error: Model not found. Please run 'python train.py' first.")
#     st.stop()

# # --- 2. DATABASE SETUP ---
# def init_db():
#     conn = sqlite3.connect('database.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS history 
#                  (timestamp TEXT, user_msg TEXT, bot_response TEXT)''')
#     conn.commit()
#     conn.close()

# def save_to_db(user_text, bot_text):
#     conn = sqlite3.connect('database.db')
#     c = conn.cursor()
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     c.execute("INSERT INTO history VALUES (?, ?, ?)", (timestamp, user_text, bot_text))
#     conn.commit()
#     conn.close()

# init_db() # Initialize DB on launch

# # --- 3. CHAT LOGIC ---
# def get_bot_response(user_input):
#     try:
#         # Convert user text to numbers
#         user_input_vec = vectorizer.transform([user_input])
#         # Predict the intent tag
#         tag = model.predict(user_input_vec)[0]
        
#         # Find a response matching the tag
#         for i in intents['intents']:
#             if i['tag'] == tag:
#                 return random.choice(i['responses'])
#     except:
#         return "I'm sorry, I didn't understand that."
    
#     return "I'm sorry, I didn't understand that."

# # --- 4. UI INTERFACE ---

# # Initialize chat history in session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display previous messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Capture User Input
# if prompt := st.chat_input("Type your message here..."):
#     # 1. Show user message
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # 2. Get Bot Response
#     response = get_bot_response(prompt)
    
#     # 3. Show bot response
#     with st.chat_message("assistant"):
#         st.markdown(response)
#     st.session_state.messages.append({"role": "assistant", "content": response})

#     # 4. Save to Database
#     save_to_db(prompt, response)

# # --- 5. EXPORT SECTION (Sidebar) ---
# st.sidebar.title("🗄️ History Options")
# st.sidebar.info("Download conversation logs below.")

# if st.sidebar.button("Refresh Data"):
#     st.rerun()

# conn = sqlite3.connect('database.db')
# df = pd.read_sql_query("SELECT * FROM history", conn)
# conn.close()

# if not df.empty:
#     # CSV
#     csv = df.to_csv(index=False).encode('utf-8')
#     st.sidebar.download_button("Download CSV", csv, "chat_history.csv", "text/csv")

#     # Excel
#     buffer = io.BytesIO()
#     with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
#         df.to_excel(writer, index=False)
#     st.sidebar.download_button("Download Excel", buffer.getvalue(), "chat_history.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

#     # PDF
#     def generate_pdf(dataframe):
#         pdf = FPDF()
#         pdf.add_page()
#         pdf.set_font("Arial", size=10)
#         pdf.cell(200, 10, txt="Conversation History", ln=1, align='C')
        
#         for index, row in dataframe.iterrows():
#             # Clean text to prevent PDF errors
#             u_text = str(row['user_msg']).encode('latin-1', 'replace').decode('latin-1')
#             b_text = str(row['bot_response']).encode('latin-1', 'replace').decode('latin-1')
            
#             pdf.set_text_color(0, 0, 255) # Blue for user
#             pdf.multi_cell(0, 7, f"User ({row['timestamp']}): {u_text}")
#             pdf.set_text_color(0, 100, 0) # Green for bot
#             pdf.multi_cell(0, 7, f"Bot: {b_text}")
#             pdf.ln(2)
#         return pdf.output(dest='S').encode('latin-1')

#     if st.sidebar.button("Generate PDF"):
#         pdf_data = generate_pdf(df)
#         st.sidebar.download_button("Download PDF", pdf_data, "chat_history.pdf", "application/pdf")
# else:
#     st.sidebar.write("No history found.")


import streamlit as st
import pickle
import json
import random
import sqlite3
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import io
import os
import nltk
import wikipedia
from deep_translator import GoogleTranslator

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="Polyglot AI Bot", page_icon="🧠")
st.title("🧠 Advanced AI Chatbot (Multi-Language)")

# Fix for NLTK data (Downloads automatically if missing)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- 2. LOAD AI BRAIN ---
@st.cache_resource
def load_resources():
    if not os.path.exists('model.pkl'):
        return None, None, None
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    return model, vectorizer, intents

model, vectorizer, intents = load_resources()

if not model:
    st.error("⚠️ Error: Model not found. Please run 'python train.py' in your terminal first.")
    st.stop()

# --- 3. DATABASE MANAGEMENT ---
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history 
                 (timestamp TEXT, user_msg TEXT, bot_response TEXT)''')
    conn.commit()
    conn.close()

def save_to_db(user_text, bot_text):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history VALUES (?, ?, ?)", (ts, user_text, bot_text))
    conn.commit()
    conn.close()

init_db()

# --- 4. SMART LOGIC (THE BRAIN) ---
def get_english_response(text_input):
    try:
        # A. Pre-processing: Fix common typos
        clean_input = text_input.lower().strip()
        if clean_input in ["hii", "hiii", "helloo", "heya"]:
            clean_input = "hello"

        # B. Predict Intent using ML
        user_input_vec = vectorizer.transform([clean_input])
        tag = model.predict(user_input_vec)[0]
        probs = model.predict_proba(user_input_vec)[0]
        confidence = max(probs)

        # Debugging (Optional: View in terminal)
        print(f"Input: {clean_input} | Tag: {tag} | Confidence: {confidence}")

        # C. LOGIC RULES
        # Rule 1: High Confidence Greeting/Goodbye -> Return immediately
        if tag in ["greeting", "goodbye", "creator", "tech_stack"] and confidence > 0.4:
            for i in intents['intents']:
                if i['tag'] == tag:
                    return random.choice(i['responses'])

        # Rule 2: Low Confidence -> Assume it's a Knowledge Question (Wikipedia)
        if confidence < 0.6: 
            tag = "search_query"

        # Rule 3: Process the Tag
        for i in intents['intents']:
            if i['tag'] == tag:
                response = random.choice(i['responses'])
                
                # Check if we need to search Wikipedia
                if response == "SEARCH_WIKIPEDIA" or tag == "search_query":
                    try:
                        # Clean query for Wikipedia (remove "who is", "what is")
                        query = clean_input.replace("what is", "").replace("who is", "").replace("tell me about", "").strip()
                        
                        # Safety: Don't search if query is too short
                        if len(query) < 3:
                            return "Could you please be more specific?"

                        # Fetch Summary
                        wiki_summary = wikipedia.summary(query, sentences=2, auto_suggest=False)
                        return f"📚 **Wikipedia:** {wiki_summary}"
                    except wikipedia.exceptions.DisambiguationError:
                        return "That topic is a bit vague. Can you be more specific?"
                    except wikipedia.exceptions.PageError:
                        return "I couldn't find any information on that topic."
                    except:
                        return "I'm having trouble connecting to Wikipedia right now."
                
                return response
        
        return "I'm not sure I understand. Can you rephrase?"

    except Exception as e:
        return f"System Error: {e}"

# --- 5. TRANSLATION WRAPPER ---
def handle_chat(user_input, target_lang_code):
    # 1. If English, skip translation (Faster)
    if target_lang_code == 'en':
        return get_english_response(user_input)

    try:
        # 2. Translate User Input -> English
        translator = GoogleTranslator(source='auto', target='en')
        english_input = translator.translate(user_input)
        
        # 3. Get Smart Response
        english_response = get_english_response(english_input)
        
        # 4. Translate Response -> Target Language
        translator_back = GoogleTranslator(source='en', target=target_lang_code)
        translated_response = translator_back.translate(english_response)
        
        return translated_response
    except Exception as e:
        return f"Translation Error: {e}"

# --- 6. UI LAYOUT ---

# Sidebar: Language Selection
st.sidebar.header("🌐 Language Settings")
languages = {'English': 'en', 'Hindi': 'hi', 'Spanish': 'es', 'French': 'fr', 'German': 'de'}
selected_lang = st.sidebar.selectbox("Choose Language", list(languages.keys()))
lang_code = languages[selected_lang]

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input(f"Type here ({selected_lang})..."):
    # Show User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get Bot Response
    with st.spinner("Thinking..."):
        response = handle_chat(prompt, lang_code)
    
    # Show Bot Response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Save to Database
    save_to_db(prompt, response)

# --- 7. EXPORT SECTION (Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.header("🗄️ Admin Tools")

if st.sidebar.button("Refresh History"):
    st.rerun()

# Load Data for Export
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM history", conn)
conn.close()

if not df.empty:
    # CSV Button
    st.sidebar.download_button(
        "Download CSV", 
        df.to_csv(index=False).encode('utf-8'), 
        "chat_history.csv", 
        "text/csv"
    )

    # Excel Button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button(
        "Download Excel", 
        buffer.getvalue(), 
        "chat_history.xlsx", 
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # PDF Logic
    def generate_pdf(dataframe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Chat Logs", ln=1, align='C')
        for i, row in dataframe.iterrows():
            # Sanitize text for PDF (Latin-1 limitation)
            u_text = str(row['user_msg']).encode('latin-1', 'replace').decode('latin-1')
            b_text = str(row['bot_response']).encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 8, f"User: {u_text}")
            pdf.multi_cell(0, 8, f"Bot: {b_text}", border='B')
            pdf.ln(2)
        return pdf.output(dest='S').encode('latin-1')

    if st.sidebar.button("Prepare PDF"):
        pdf_data = generate_pdf(df)
        st.sidebar.download_button("Download PDF", pdf_data, "history.pdf", "application/pdf")