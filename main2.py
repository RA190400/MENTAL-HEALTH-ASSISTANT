import streamlit as st
from llama_cpp import Llama
import os
from huggingface_hub import hf_hub_download

# Function to download the model
@st.cache_resource
def download_model():
    model_path = hf_hub_download(
        repo_id="RAKS19/Llama-2-7b-Mental-health-chatbot-finetune2-Q4_K_M-GGUF",
        filename="llama-2-7b-mental-health-chatbot-finetune2-q4_k_m.gguf"
    )
    return model_path

# Load the model
@st.cache_resource
def load_model(model_path):
    return Llama(model_path=model_path)

def main():
    # Streamlit UI setup
    st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
    st.title("Mental Health Chatbot 🤖")
    st.markdown("""
    Welcome to the Mental Health Chatbot! Share your thoughts or concerns, and I'll provide empathetic advice.
    """)

    # Download and load the model
    st.info("Loading the model. This may take some time...")
    model_path = download_model()
    llm = load_model(model_path)
    response_guidelines = """
        You are a psychologist. Respond empathetically in clear, numbered points:
        1. Show empathy for feelings.
        2. Provide actionable advice in short points.
        3. Be clear, concise, and supportive.
        4. For serious issues, encourage professional help kindly.
        """
    prompt = f"{response_guidelines}\n\n"
    # Input box for user message
    user_input = st.text_input("Your message:", placeholder="Type your message here...")
    prompt += f"User: {user_input}\nBot:"
    print(prompt)
    if st.button("Send"):
        if user_input.strip():
            with st.spinner("The bot is thinking..."):
                response = llm(prompt, max_tokens=200)
                bot_response = response["choices"][0]["text"].strip()
                st.markdown(f"**🤖 Bot:** {bot_response}")
        else:
            st.warning("Please enter a message before clicking send!")

if __name__ == "__main__":
    main()


















