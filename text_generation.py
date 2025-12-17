import boto3
import time


# ---------------------------------------
# BEDROCK CLIENT (uses aws configure creds)
# ---------------------------------------
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"  # models region
)

MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"


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

        # Extract final answer
        return response["output"]["message"]["content"][0]["text"]

    except Exception as e:
        raise RuntimeError(f"Bedrock error: {str(e)}")


# ---------------------------------------
# SIMPLE DEMO EXECUTION
# ---------------------------------------
if __name__ == "__main__":
    SYSTEM_PROMPT = "You are an AI assistant, and you need to answer the user's question."

    print("Claude 3.7 Sonnet Demo")
    print("-" * 30)

    user_question = input("Ask Claude a question: ")

    answer = call_claude(SYSTEM_PROMPT, user_question)

    print("\nClaude says:\n")
    print(answer)
