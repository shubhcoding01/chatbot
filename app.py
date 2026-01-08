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

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Python AI Assistant")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_resources():
    # Load the saved model and vectorizer
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

if model is None:
    st.error("Error: Model not found. Please run 'python train.py' first.")
    st.stop()

# --- 2. DATABASE SETUP ---
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history VALUES (?, ?, ?)", (timestamp, user_text, bot_text))
    conn.commit()
    conn.close()

init_db() # Initialize DB on launch

# --- 3. CHAT LOGIC ---
def get_bot_response(user_input):
    try:
        # Convert user text to numbers
        user_input_vec = vectorizer.transform([user_input])
        # Predict the intent tag
        tag = model.predict(user_input_vec)[0]
        
        # Find a response matching the tag
        for i in intents['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    except:
        return "I'm sorry, I didn't understand that."
    
    return "I'm sorry, I didn't understand that."

# --- 4. UI INTERFACE ---

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture User Input
if prompt := st.chat_input("Type your message here..."):
    # 1. Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Get Bot Response
    response = get_bot_response(prompt)
    
    # 3. Show bot response
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 4. Save to Database
    save_to_db(prompt, response)

# --- 5. EXPORT SECTION (Sidebar) ---
st.sidebar.title("🗄️ History Options")
st.sidebar.info("Download conversation logs below.")

if st.sidebar.button("Refresh Data"):
    st.rerun()

conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM history", conn)
conn.close()

if not df.empty:
    # CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download CSV", csv, "chat_history.csv", "text/csv")

    # Excel
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    st.sidebar.download_button("Download Excel", buffer.getvalue(), "chat_history.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # PDF
    def generate_pdf(dataframe):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        pdf.cell(200, 10, txt="Conversation History", ln=1, align='C')
        
        for index, row in dataframe.iterrows():
            # Clean text to prevent PDF errors
            u_text = str(row['user_msg']).encode('latin-1', 'replace').decode('latin-1')
            b_text = str(row['bot_response']).encode('latin-1', 'replace').decode('latin-1')
            
            pdf.set_text_color(0, 0, 255) # Blue for user
            pdf.multi_cell(0, 7, f"User ({row['timestamp']}): {u_text}")
            pdf.set_text_color(0, 100, 0) # Green for bot
            pdf.multi_cell(0, 7, f"Bot: {b_text}")
            pdf.ln(2)
        return pdf.output(dest='S').encode('latin-1')

    if st.sidebar.button("Generate PDF"):
        pdf_data = generate_pdf(df)
        st.sidebar.download_button("Download PDF", pdf_data, "chat_history.pdf", "application/pdf")
else:
    st.sidebar.write("No history found.")
