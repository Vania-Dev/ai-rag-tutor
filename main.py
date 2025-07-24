import streamlit as st
import os

from prompt_engineering.agent import chat_with_agent  # Ahora usamos esta función
from utils.rag_utils import load_and_index_pdf
PDF_DIR  = "data/pdfs"

st.set_page_config(page_title="Asesor IA RAG", layout="wide")

st.title("📚 Asesor de Investigación con RAG + Ollama")
st.markdown("Carga libros especializados en PDF y haz preguntas al agente IA entrenado con LangChain y LangGraph.")

# 📄 Subida e indexación de PDFs
with st.sidebar:
    st.header("📄 Subir documento PDF")
    uploaded_file = st.file_uploader("Selecciona un archivo PDF", type=["pdf"])
    if uploaded_file:
        os.makedirs(PDF_DIR, exist_ok=True)
        path = os.path.join(PDF_DIR, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        load_and_index_pdf(path)
        st.success("✅ Documento indexado correctamente")

# 💬 Estado del chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown("### 💬 Chat con tu Asesor IA")
user_input = st.text_input("Tu pregunta:", key="user_input")

if st.button("Preguntar") and user_input.strip():
    with st.spinner("Pensando..."):
        # Pasamos la pregunta y el historial
        response = chat_with_agent(user_input, st.session_state.chat_history)

        # Guardamos en el historial
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# 📝 Mostrar historial
for message in st.session_state.chat_history:
    role = "🧑‍🎓 Tú" if message["role"] == "user" else "🤖 Asesor IA"
    st.markdown(f"**{role}**: {message['content']}")
