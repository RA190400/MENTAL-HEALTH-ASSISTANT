import streamlit as st
from llama_cpp import Llama

def format_response(raw_response):
    """
    Formats the chatbot's raw response into a structured and readable format.
    """
    suggestions = raw_response.split("\n")  # Split response into lines for better formatting
    formatted_response = "**🤖 Bot:**\n\n"
    
    for idx, suggestion in enumerate(suggestions, start=1):
        if suggestion.strip():  # Avoid blank lines
            formatted_response += f"{idx}. {suggestion.strip()}\n\n"
    
    return formatted_response

def llm_answer(user_input, chat_history, max_context_messages=2):
    """
    Interacts with the GGUF model using llama_cpp to generate a response.

    Args:
        user_input (str): The user's input message.
        chat_history (list): The chat history as a list of dictionaries with keys 'role' and 'content'.
        max_context_messages (int): Maximum number of chat history messages to include in the prompt.

    Returns:
        str: The model's response.
    """
    try:
        # Load the GGUF model
        llm = Llama.from_pretrained(
            repo_id="RAKS19/Llama-2-7b-Mental-health-chatbot-finetune2-Q4_K_M-GGUF",
            filename="llama-2-7b-mental-health-chatbot-finetune2-q4_k_m.gguf",
        )

        # Define concise response guidelines
        response_guidelines = """
        You are a psychologist. Respond empathetically in clear, numbered points:
        1. Show empathy for feelings.
        2. Provide actionable advice in short points.
        3. Be clear, concise, and supportive.
        4. For serious issues, encourage professional help kindly.
        """

        # Prepare the prompt
        prompt = f"{response_guidelines}\n\n"
        for msg in chat_history[-max_context_messages:]:  # Include only the last N messages
            if msg['role'] == 'user':
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Bot: {msg['content']}\n"
        prompt += f"User: {user_input}\nBot:"

        # Print the prompt for debugging
        print("\n========== Prompt Sent to Model ==========\n")
        print(prompt)
        print("\n==========================================\n")

        # Generate response with reduced max_tokens
        response = llm(prompt, max_tokens=300)  # Limit response to fit within context
        bot_response = response["choices"][0]["text"].strip()
        return bot_response

    except Exception as e:
        return f"Error: {str(e)}"




def main():
    # Streamlit UI setup
    st.set_page_config(page_title="Mental Health Chatbot", layout="wide")
    st.markdown(
        """
        <style>
            /* Set a background gradient */
            .stApp {
                background: linear-gradient(to bottom, #4e54c8, #8f94fb); /* Blue to purple gradient */
                background-size: cover;
                color: #000000; /* Set all text to black */
            }
            /* Style the chat history */
            .chat-history {
                background: rgba(255, 255, 255, 0.1); /* White with slight transparency */
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
                color: #000000; /* Set chat text to black */
            }
            /* Style the input box */
            .stTextInput > div {
                background: rgba(255, 255, 255, 0.2); /* Slightly transparent white */
                border-radius: 5px;
                padding: 10px;
                border: 1px solid #ffffff;
                color: #000000; /* Set input text to black */
            }
            /* Style the buttons */
            .stButton button {
                background-color: #6a5acd; /* Slate blue */
                color: white;
                border-radius: 5px;
                padding: 10px 20px;
                border: none;
                font-size: 16px;
                cursor: pointer;
            }
            .stButton button:hover {
                background-color: #7b68ee; /* Light slate blue */
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Mental Health Chatbot 🤖")
    st.markdown("""
    Welcome to the Mental Health Chatbot! 🤗 Share your thoughts or concerns, 
    and I'll provide empathetic and helpful advice. This chatbot is here to support you.
    """)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for clearing chat
    with st.sidebar:
        st.subheader("Options")
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared successfully!")

    # Chat history section (scrollable above the input box)
    with st.container():
        chat_placeholder = st.empty()  # Placeholder for dynamic chat display
        chat_display = ""
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_display += f"**🧑 You:** {msg['content']}\n\n"
            else:
                chat_display += f"**🤖 Bot:** {msg['content']}\n\n"
        chat_placeholder.markdown(chat_display, unsafe_allow_html=True)

    # Spacer to ensure input box stays at the bottom
    st.write("")  # Empty line
    st.write("")  # Additional empty line

    # Input box at the bottom
    user_input = st.text_input(
        "Your message:",
        placeholder="Type your response here...",
        key="user_input"  # Adding a key to track the state of the input box
    )

    # Send button to submit input
    if st.button("Send", key="send_button"):
        if user_input.strip():
            # Append user's message to chat history
            

            # Fetch chatbot response
            with st.spinner("The bot is thinking..."):
                bot_response = llm_answer(user_input, st.session_state.chat_history)
                #bot_response = format_response(bot_response)
            st.subheader("🤖 Bot's Response:")
            st.markdown(f"{bot_response}")
            # Append bot's response to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
            

        else:
            st.warning("Please enter a message before clicking send!")

    # Display bot's latest response below the input box
    

    # Footer disclaimer
    st.markdown("""
    ---
    **Disclaimer:** This chatbot is for informational purposes only and is not a substitute for professional mental health advice.
    """)

# Run the Streamlit app
if __name__ == "__main__":
    main()









