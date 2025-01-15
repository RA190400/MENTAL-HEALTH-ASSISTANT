import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Title and description
st.title("Mental Health Chatbot")
st.write("This app uses a fine-tuned Llama-2 model for mental health-related conversations.")

# Load the model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("RAKS19/Llama-2-7b-Mental-health-chatbot-finetune2")
    model = AutoModelForCausalLM.from_pretrained("RAKS19/Llama-2-7b-Mental-health-chatbot-finetune2")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

chatbot_pipeline = load_model_and_tokenizer()

# User input
user_input = st.text_input("Enter your message:", "I feel stressed and anxious.")

# Generate response
if st.button("Get Response"):
    with st.spinner("Thinking..."):
        response = chatbot_pipeline(user_input, max_length=100, num_return_sequences=1)
        chatbot_reply = response[0]['generated_text']
        st.subheader("Chatbot's Response:")
        st.write(chatbot_reply)

# Instructions for deployment
st.markdown("---")
st.markdown("### Deploying on Streamlit Cloud")
st.markdown("1. Save this script as `app.py`.")
st.markdown("2. Create a `requirements.txt` file with the following content:")
st.code("""streamlit\ntransformers""", language="text")
st.markdown("3. Push your code to a GitHub repository.")
st.markdown("4. Connect your repository to Streamlit Cloud and deploy the app.")



















