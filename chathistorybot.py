import boto3
import time
import streamlit as st

# ---------------------------------------
# BEDROCK CLIENT
# ---------------------------------------
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "us.amazon.nova-pro-v1:0"

SYSTEM_PROMPT = "You are an AI assistant, and you need to answer the user's question."

# ---------------------------------------
# CALL MODEL WITH CHAT HISTORY
# ---------------------------------------
def call_model(messages):
    params = {
        "modelId": MODEL_ID,
        "system": [
            {
                "text": SYSTEM_PROMPT
            }
        ],

        "messages": messages,

        "inferenceConfig": {
            "temperature": 0.4,
            "maxTokens": 1024
        }
    }

    try:
        start_time = time.time()
        response = bedrock_client.converse(**params)
        print(f"[INFO] Call took {time.time() - start_time:.2f}s")

        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        raise RuntimeError(f"Bedrock error: {str(e)}")


# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(page_title="Nova Chatbot", layout="centered")

st.title("🤖 Nova Chatbot (with memory)")

# ---------------------------------------
# SESSION STATE INIT (NO SYSTEM HERE)
# ---------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"][0]["text"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"][0]["text"])

# ---------------------------------------
# USER INPUT
# ---------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Add user message
    user_message = {
        "role": "user",
        "content": [{"text": user_input}]
    }
    st.session_state.messages.append(user_message)

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Call model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = call_model(st.session_state.messages)
            st.write(answer)

    # Add assistant response
    assistant_message = {
        "role": "assistant",
        "content": [{"text": answer}]
    }
    st.session_state.messages.append(assistant_message)


# ---------------------------------------
# CLEAR CHAT BUTTON
# ---------------------------------------
if st.button("Clear Chat"):
    st.session_state.messages = []
