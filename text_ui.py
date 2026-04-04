import boto3
import time
import streamlit as st


# ---------------------------------------
# BEDROCK CLIENT (uses aws configure creds)
# ---------------------------------------
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "us.anthropic.claude-sonnet-4-20250514-v1:0"


# ---------------------------------------
# CALL CLAUDE (CONVERSE API)
# ---------------------------------------
def call_claude(system_prompt: str, user_question: str) -> str:
    params = {
        "modelId": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": f"{system_prompt}\n\nUser question:\n{user_question}"
                    }
                ]
            }
        ],
        "inferenceConfig": {
            "temperature": 0.4,
            "maxTokens": 1024
        }
    }

    try:
        start_time = time.time()
        response = bedrock_client.converse(**params)
        print(f"[INFO] Claude call took {time.time() - start_time:.2f}s")

        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        raise RuntimeError(f"Bedrock error: {str(e)}")


# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(page_title="Claude 3.7 Sonnet", layout="centered")

st.title("🤖 Claude LLM Demo chatbot")

SYSTEM_PROMPT = "You are an AI assistant, and you need to answer the user's question."

user_question = st.text_input("Ask Claude a question:")

if st.button("Submit") and user_question:
    with st.spinner("Thinking..."):
        answer = call_claude(SYSTEM_PROMPT, user_question)

    st.subheader("Claude says:")
    st.write(answer)