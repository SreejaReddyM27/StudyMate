import streamlit as st
from PyPDF2 import PdfReader
import requests
import json
import os
from datetime import datetime

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "granite3-dense:latest"

# JSON file paths for persistence
QA_HISTORY_FILE = "qa_history.json"
CHAT_HISTORY_FILE = "chat_history.json"
PROCESSED_FILES_FILE = "processed_files.json"

def save_to_json(data, filename):
    """Save data to JSON file with error handling"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        return True
    except Exception as e:
        st.error(f"Error saving to {filename}: {str(e)}")
        return False

def load_from_json(filename, default=None):
    """Load data from JSON file with error handling"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return default if default is not None else []
    except Exception as e:
        st.error(f"Error loading from {filename}: {str(e)}")
        return default if default is not None else []

def save_qa_history():
    """Save Q&A history to JSON file"""
    if st.session_state.qa_history:
        # Add timestamp to each entry if not present
        timestamped_history = []
        for entry in st.session_state.qa_history:
            if len(entry) == 4:  # Original format: (chunk_num, question, answer, filename)
                chunk_num, question, answer, filename = entry
                timestamped_entry = {
                    "chunk_num": chunk_num,
                    "question": question,
                    "answer": answer,
                    "filename": filename,
                    "timestamp": datetime.now().isoformat()
                }
            elif len(entry) == 5:  # Already has timestamp
                chunk_num, question, answer, filename, timestamp = entry
                timestamped_entry = {
                    "chunk_num": chunk_num,
                    "question": question,
                    "answer": answer,
                    "filename": filename,
                    "timestamp": timestamp
                }
            else:  # Handle unexpected formats
                continue
            timestamped_history.append(timestamped_entry)

        return save_to_json(timestamped_history, QA_HISTORY_FILE)
    return True

def load_qa_history():
    """Load Q&A history from JSON file"""
    data = load_from_json(QA_HISTORY_FILE, [])
    formatted_history = []

    for entry in data:
        if isinstance(entry, dict):
            formatted_entry = (
                entry.get("chunk_num", 1),
                entry.get("question", ""),
                entry.get("answer", ""),
                entry.get("filename", "unknown"),
                entry.get("timestamp", datetime.now().isoformat())
            )
            formatted_history.append(formatted_entry)

    return formatted_history

def save_chat_history():
    """Save chat history to JSON file"""
    if st.session_state.chat_messages:
        # Add timestamps to chat messages if not present
        timestamped_messages = []
        for msg in st.session_state.chat_messages:
            if isinstance(msg, dict):
                msg_copy = msg.copy()
                if "timestamp" not in msg_copy:
                    msg_copy["timestamp"] = datetime.now().isoformat()
                timestamped_messages.append(msg_copy)

        return save_to_json(timestamped_messages, CHAT_HISTORY_FILE)
    return True

def load_chat_history():
    """Load chat history from JSON file"""
    return load_from_json(CHAT_HISTORY_FILE, [])

def save_processed_files():
    """Save processed files list to JSON file"""
    if st.session_state.processed_files:
        file_data = []
        for filename in st.session_state.processed_files:
            if isinstance(filename, str):
                file_entry = {
                    "filename": filename,
                    "processed_date": datetime.now().isoformat()
                }
                file_data.append(file_entry)

        return save_to_json(file_data, PROCESSED_FILES_FILE)
    return True

def load_processed_files():
    """Load processed files from JSON file"""
    data = load_from_json(PROCESSED_FILES_FILE, [])
    return [entry.get("filename", entry) if isinstance(entry, dict) else entry for entry in data]

def export_all_history():
    """Export all history data to a single JSON file"""
    export_data = {
        "export_timestamp": datetime.now().isoformat(),
        "qa_history": load_from_json(QA_HISTORY_FILE, []),
        "chat_history": load_from_json(CHAT_HISTORY_FILE, []),
        "processed_files": load_from_json(PROCESSED_FILES_FILE, [])
    }

    export_filename = f"pdf_qa_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if save_to_json(export_data, export_filename):
        return export_filename
    return None

@st.cache_resource
def test_ollama_connection():
    """Test if Ollama is running and model is available"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            available_models = [model['name'] for model in models]
            if MODEL_NAME in available_models:
                return True, f"✅ Ollama is running. Model {MODEL_NAME} is available."
            else:
                return False, f"❌ Model {MODEL_NAME} not found. Available models: {available_models}"
        else:
            return False, "❌ Ollama server is not responding"
    except Exception as e:
        return False, f"❌ Cannot connect to Ollama: {str(e)}"

def generate_with_ollama(prompt, max_tokens=300, temperature=0.7):
    """Generate text using Ollama API"""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    except Exception as e:
        st.error(f"Error calling Ollama: {str(e)}")
        return ""

def ollama_chat_stream(model_name, messages):
    """Generate streaming response using Ollama chat API"""
    try:
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True
        }

        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True, timeout=60)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    if chunk.get('message', {}).get('content'):
                        yield chunk['message']['content']
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        st.error(f"Error in streaming chat: {str(e)}")
        yield ""

# Initialize session state and load from JSON files
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = load_qa_history()
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = load_processed_files()
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = load_chat_history()
if 'pdf_context' not in st.session_state:
    st.session_state.pdf_context = ""

def extract_text_from_pdf(pdf_file, start_page=None, end_page=None):
    """Extract text from PDF pages"""
    reader = PdfReader(pdf_file)
    total_pages = len(reader.pages)

    # Default to all pages if not specified
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = total_pages

    # Adjust for 0-based indexing
    start_idx = max(0, start_page - 1)
    end_idx = min(total_pages, end_page)

    text = ""
    for page_num in range(start_idx, end_idx):
        text += reader.pages[page_num].extract_text() or ""

    return text, end_idx - start_idx

def chunk_text(text, max_words=400):
    """Split text into chunks for processing"""
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def generate_qa_pairs(text_chunk, num_questions=5):
    """Generate Q&A pairs using Ollama model"""
    prompt = f"""Based on the following text, generate {num_questions} clear questions with their corresponding answers.

Text: {text_chunk}

Format your response exactly as:
Q1: [Question]
A1: [Answer]

Q2: [Question] 
A2: [Answer]

And so on...

Questions and Answers:"""

    response = generate_with_ollama(prompt, max_tokens=400, temperature=0.7)
    if response:
        # Extract the part after "Questions and Answers:" if it exists
        if "Questions and Answers:" in response:
            return response.split("Questions and Answers:")[-1].strip()
        else:
            return response.strip()
    return ""

def parse_qa_pairs(qa_text):
    """Parse the generated text into question-answer pairs"""
    lines = qa_text.split('\n')
    qa_pairs = []
    current_question = ""
    current_answer = ""

    for line in lines:
        line = line.strip()
        if line.startswith('Q') and ':' in line:
            if current_question and current_answer:
                qa_pairs.append((current_question, current_answer))
            current_question = line[line.find(':')+1:].strip()
            current_answer = ""
        elif line.startswith('A') and ':' in line:
            current_answer = line[line.find(':')+1:].strip()

    # Add the last pair
    if current_question and current_answer:
        qa_pairs.append((current_question, current_answer))

    return qa_pairs

def generate_contextual_response(user_question, pdf_context):
    """Generate response based on PDF context"""
    if not pdf_context.strip():
        return "Please upload and process a PDF first to enable context-aware responses."

    # Create a contextual prompt
    prompt = f"""You are an AI assistant helping users understand a PDF document. 
    Use the following context from the PDF to answer the user's question. 
    If the answer is not available in the context, say so clearly.

    PDF Context:
    {pdf_context[:3000]}  # Limit context to avoid token limits

    User Question: {user_question}

    Answer based on the PDF context:"""

    return generate_with_ollama(prompt, max_tokens=500, temperature=0.3)

# Streamlit UI Configuration
st.set_page_config(
    page_title="📄 Enhanced PDF Q&A with AI Chatbot & JSON History",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Test Ollama connection
connection_status, connection_message = test_ollama_connection()

# Main Title
st.title("📄 Enhanced PDF Q&A Generator with AI Chatbot & JSON History")
st.write("Upload a PDF, generate Q&A pairs, chat with an AI assistant, and persist your history in JSON files.")

# Display connection status
if connection_status:
    st.success(connection_message)
else:
    st.error(connection_message)
    st.info("Make sure Ollama is running with: ollama serve and you have the granite3-dense:latest model installed")

# Sidebar - About Us and History Management
st.sidebar.title("ℹ About This App")
st.sidebar.markdown(f"""
### Features:
- ✅ *PDF Q&A Generation*: Automatic question-answer generation from PDF content
- ✅ *AI Chatbot*: Interactive chat with PDF context awareness  
- ✅ *Local LLM*: Powered by Ollama (no internet required for inference)
- ✅ *JSON Persistence*: History saved to JSON files
- ✅ *Session History*: Track Q&A pairs and chat conversations
- ✅ *Page Selection*: Choose specific pages to process
- ✅ *Export/Import*: Backup and restore your data

### Models & Frameworks:
*🤖 Language Model:*
- *Ollama {MODEL_NAME}*: Running locally

*🛠 Tech Stack:*
- *Streamlit*: Interactive web framework
- *Ollama*: Local LLM inference engine  
- *PyPDF2*: PDF text extraction
- *JSON*: Data persistence

### System Status:
- *Ollama Server*: {"🟢 Connected" if connection_status else "🔴 Disconnected"}
- *Model*: {MODEL_NAME}
- *Endpoint*: {OLLAMA_BASE_URL}
""")

# History Management Section
st.sidebar.markdown("---")
st.sidebar.subheader("📊 History Management")

# Display current statistics
qa_count = len(st.session_state.qa_history)
chat_count = len(st.session_state.chat_messages)
files_count = len(st.session_state.processed_files)

st.sidebar.metric("Q&A Pairs", qa_count)
st.sidebar.metric("Chat Messages", chat_count)
st.sidebar.metric("Processed Files", files_count)

# History control buttons
col1, col2 = st.sidebar.columns(2)

with col1:
    if st.button("💾 Save All", help="Save current session to JSON files"):
        success_count = 0
        if save_qa_history():
            success_count += 1
        if save_chat_history():
            success_count += 1
        if save_processed_files():
            success_count += 1

        if success_count == 3:
            st.sidebar.success("✅ All data saved!")
        else:
            st.sidebar.warning("⚠ Some data may not have saved properly")

with col2:
    if st.button("📤 Export", help="Export all history to single JSON file"):
        export_file = export_all_history()
        if export_file:
            st.sidebar.success(f"✅ Exported to: {export_file}")
        else:
            st.sidebar.error("❌ Export failed")

# Clear history section
st.sidebar.markdown("---")
st.sidebar.subheader("🗑 Clear Data")

clear_option = st.sidebar.selectbox(
    "Choose what to clear:",
    ["Select option", "Clear Session Only", "Clear JSON Files", "Clear Everything"]
)

if st.sidebar.button("🗑 Clear Selected", type="secondary"):
    if clear_option == "Clear Session Only":
        st.session_state.qa_history = []
        st.session_state.processed_files = []
        st.session_state.chat_messages = []
        st.session_state.pdf_context = ""
        st.sidebar.success("Session cleared!")

    elif clear_option == "Clear JSON Files":
        try:
            for file in [QA_HISTORY_FILE, CHAT_HISTORY_FILE, PROCESSED_FILES_FILE]:
                if os.path.exists(file):
                    os.remove(file)
            st.sidebar.success("JSON files cleared!")
        except Exception as e:
            st.sidebar.error(f"Error clearing files: {str(e)}")

    elif clear_option == "Clear Everything":
        # Clear session
        st.session_state.qa_history = []
        st.session_state.processed_files = []
        st.session_state.chat_messages = []
        st.session_state.pdf_context = ""

        # Clear files
        try:
            for file in [QA_HISTORY_FILE, CHAT_HISTORY_FILE, PROCESSED_FILES_FILE]:
                if os.path.exists(file):
                    os.remove(file)
            st.sidebar.success("Everything cleared!")
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

# Main layout with two columns
col1, col2 = st.columns([1.2, 0.8], gap="medium")

# Left Column - PDF Q&A Generator
with col1:
    st.header("📋 PDF Q&A Generator")

    # File upload
    st.subheader("📁 Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF file", type=["pdf"], key="pdf_upload")

    if pdf_file:
        # Get total pages for reference
        reader = PdfReader(pdf_file)
        total_pages = len(reader.pages)
        st.info(f"📖 PDF has *{total_pages}* pages")

        # Page selection
        st.subheader("📖 Page Selection")
        col_a, col_b = st.columns(2)

        with col_a:
            start_page = st.number_input(
                "Start Page", 
                min_value=1, 
                max_value=total_pages, 
                value=1,
                help="First page to process (inclusive)"
            )

        with col_b:
            end_page = st.number_input(
                "End Page", 
                min_value=start_page, 
                max_value=total_pages, 
                value=min(3, total_pages),
                help="Last page to process (inclusive)"
            )

        # Question settings
        st.subheader("⚙ Generation Settings")
        num_questions = st.slider("Number of questions per chunk", 1, 8, 4)

        # Auto-save option
        auto_save = st.checkbox("💾 Auto-save to JSON", value=True, help="Automatically save Q&A pairs to JSON file")

        # Submit button
        if st.button("🚀 Generate Questions & Answers", type="primary"):
            if not connection_status:
                st.error("❌ Cannot generate Q&A: Ollama is not connected")
            else:
                with st.spinner("🔄 Extracting text from selected pages..."):
                    text, pages_processed = extract_text_from_pdf(pdf_file, start_page, end_page)
                    # Store PDF context for chatbot
                    st.session_state.pdf_context = text

                if not text.strip():
                    st.error("❌ No text found in the selected pages. Please try different pages.")
                else:
                    st.success(f"✅ Text extracted from *{pages_processed}* pages (pages {start_page} to {end_page})")

                    chunks = chunk_text(text)
                    st.write(f"📝 Content divided into *{len(chunks)}* chunks for processing.")

                    # Generate Q&A for each chunk
                    all_qa_pairs = []
                    progress_bar = st.progress(0)
                    current_time = datetime.now().isoformat()

                    for i, chunk in enumerate(chunks):
                        with st.spinner(f"🤖 Generating Q&A for chunk {i+1}/{len(chunks)}..."):
                            qa_text = generate_qa_pairs(chunk, num_questions=num_questions)
                            qa_pairs = parse_qa_pairs(qa_text)
                            # Add timestamp to each Q&A pair
                            timestamped_pairs = [(i+1, q, a, pdf_file.name, current_time) for q, a in qa_pairs if q and a]
                            all_qa_pairs.extend(timestamped_pairs)

                        progress_bar.progress((i + 1) / len(chunks))

                    if all_qa_pairs:
                        # Add to history
                        st.session_state.qa_history.extend(all_qa_pairs)
                        if pdf_file.name not in st.session_state.processed_files:
                            st.session_state.processed_files.append(pdf_file.name)

                        # Auto-save if enabled
                        if auto_save:
                            save_qa_history()
                            save_processed_files()
                            st.success(f"🎉 Generated *{len(all_qa_pairs)}* Q&A pairs and saved to JSON!")
                        else:
                            st.success(f"🎉 Generated *{len(all_qa_pairs)}* Q&A pairs!")

                        # Display Current Q&A pairs
                        st.subheader("❓ Generated Questions & Answers")

                        for idx, (chunk_num, question, answer, filename, timestamp) in enumerate(all_qa_pairs, 1):
                            with st.container():
                                st.markdown(f"*Q{idx}:* {question}")
                                st.markdown(f"*A{idx}:* {answer}")
                                st.caption(f"📄 Source: {filename} (Chunk {chunk_num}) - Generated: {timestamp[:19]}")
                                st.markdown("---")
                    else:
                        st.warning("⚠ No valid Q&A pairs were generated. Try adjusting settings.")

    # Q&A History section
    if st.session_state.qa_history:
        st.subheader("📚 Q&A History")
        st.write(f"Total questions generated: *{len(st.session_state.qa_history)}*")

        # Show recent history in an expander
        with st.expander(f"Recent Q&A History (Last 5)", expanded=False):
            recent_qa = st.session_state.qa_history[-5:]
            for idx, qa_entry in enumerate(reversed(recent_qa), 1):
                if len(qa_entry) >= 4:
                    chunk_num, question, answer, filename = qa_entry[:4]
                    timestamp = qa_entry[4] if len(qa_entry) > 4 else "Unknown time"
                    st.markdown(f"*Q:* {question}")
                    st.markdown(f"*A:* {answer}")
                    st.caption(f"📄 {filename} - {timestamp[:19] if isinstance(timestamp, str) else timestamp}")
                    st.markdown("---")

# Right Column - AI Chatbot
with col2:
    st.header("🤖 AI Chatbot")
    st.caption("Chat about your PDF content")

    # Auto-save chat option
    auto_save_chat = st.checkbox("💾 Auto-save chat", value=True, help="Automatically save chat messages to JSON file")

    # Chat container with fixed height
    chat_container = st.container(height=400)

    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "timestamp" in message:
                    st.caption(f"🕒 {message['timestamp'][:19]}")

    # Chat input
    if prompt := st.chat_input("Ask about the PDF content...", key="chat_input"):
        if not connection_status:
            st.error("❌ Ollama is not connected")
        else:
            current_time = datetime.now().isoformat()

            # Add user message to chat history
            user_message = {"role": "user", "content": prompt, "timestamp": current_time}
            st.session_state.chat_messages.append(user_message)

            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                    st.caption(f"🕒 {current_time[:19]}")

            # Generate and display assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    # Prepare messages for streaming
                    messages = []
                    if st.session_state.pdf_context:
                        # Add system message with PDF context
                        system_msg = f"""You are an AI assistant helping users understand a PDF document. 
                        Use the following context from the PDF to answer questions accurately.
                        If the answer is not in the context, say so clearly.

                        PDF Context: {st.session_state.pdf_context[:2000]}"""
                        messages.append({"role": "system", "content": system_msg})

                    # Add conversation history (last 4 messages to keep context manageable)
                    recent_messages = st.session_state.chat_messages[-4:]
                    for msg in recent_messages[:-1]:  # Exclude the just-added user message
                        messages.append({"role": msg["role"], "content": msg["content"]})

                    # Add current user message
                    messages.append({"role": "user", "content": prompt})

                    # Stream the response
                    try:
                        response = st.write_stream(ollama_chat_stream(MODEL_NAME, messages))
                        response_time = datetime.now().isoformat()
                        st.caption(f"🕒 {response_time[:19]}")
                    except Exception as e:
                        response = f"Error generating response: {str(e)}"
                        response_time = datetime.now().isoformat()
                        st.error(response)

            # Add assistant response to chat history
            assistant_message = {"role": "assistant", "content": response, "timestamp": response_time}
            st.session_state.chat_messages.append(assistant_message)

            # Auto-save if enabled
            if auto_save_chat:
                save_chat_history()

    # Chat status
    if st.session_state.pdf_context:
        st.success("✅ PDF context loaded - chatbot ready!")
    else:
        st.info("💡 Upload and process a PDF to enable context-aware chat")

    # Chat history controls
    if len(st.session_state.chat_messages) > 0:
        col_clear, col_count = st.columns(2)
        with col_clear:
            if st.button("🧹 Clear Chat", key="clear_chat"):
                st.session_state.chat_messages = []
                # Also clear the JSON file if exists
                if os.path.exists(CHAT_HISTORY_FILE):
                    save_to_json([], CHAT_HISTORY_FILE)
                st.rerun()
        with col_count:
            st.caption(f"💬 {len(st.session_state.chat_messages)} messages")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666;'>
    <p><strong>Enhanced PDF Q&A with AI Chatbot & JSON Persistence</strong> 🚀</p>
    <p>Powered by Ollama + {MODEL_NAME} | Local LLM with persistent history</p>
    <p>📁 Data files: {QA_HISTORY_FILE}, {CHAT_HISTORY_FILE}, {PROCESSED_FILES_FILE}</p>
</div>
""", unsafe_allow_html=True)

# Enhanced CSS styling
st.markdown("""
<style>
/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1rem;
    font-weight: bold;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #FF5252 0%, #26C6DA 100%);
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Progress bar styling */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
}

/* Sidebar styling */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
}

/* Container styling */
.stContainer > div {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #4ECDC4;
    margin: 0.5rem 0;
}

/* Chat message styling */
div[data-testid="stChatMessage"] {
    margin-bottom: 1rem;
}

/* Metric styling */
div[data-testid="metric-container"] {
    background: linear-gradient(45deg, #f8f9fa, #ffffff);
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Column gap adjustment */
.block-container {
    padding-top: 1rem;
}

/* Custom chat container */
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}
</style>
""", unsafe_allow_html=True)