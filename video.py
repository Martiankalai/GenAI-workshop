import boto3
import json
import base64
import time


# ---------------------------------------
# BEDROCK CLIENT
# ---------------------------------------
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "amazon.nova-reel-v1:1"


# ---------------------------------------
# GENERATE VIDEO USING NOVA REEL (ASYNC)
# ---------------------------------------
def generate_video(prompt: str, output_file: str = "output.mp4"):
    request_body = {
        "taskType": "TEXT_VIDEO",
        "textToVideoParams": {
            "text": prompt
        },
        "videoGenerationConfig": {
            "durationSeconds": 4,
            "fps": 8,
            "height": 512,
            "width": 512,
            "cfgScale": 7.5
        }
    }

    print("[INFO] Starting Nova Reel async job...")

    # 1️⃣ Start async job
    start_response = bedrock_client.start_async_invoke(
        modelId=MODEL_ID,
        body=json.dumps(request_body),
        accept="application/json",
        contentType="application/json"
    )

    invocation_arn = start_response["invocationArn"]
    print(f"[INFO] Invocation ARN: {invocation_arn}")

    # 2️⃣ Poll for completion
    while True:
        status_response = bedrock_client.get_async_invoke(
            invocationArn=invocation_arn
        )

        status = status_response["status"]
        print(f"[INFO] Job status: {status}")

        if status == "Completed":
            break
        elif status == "Failed":
            raise RuntimeError(f"Nova Reel job failed: {status_response}")

        time.sleep(5)

    # 3️⃣ Extract video
    response_body = json.loads(status_response["output"]["body"])
    video_base64 = response_body["videos"][0]
    video_bytes = base64.b64decode(video_base64)

    # 4️⃣ Save video
    with open(output_file, "wb") as f:
        f.write(video_bytes)

    print(f"[SUCCESS] Video saved as {output_file}")


# ---------------------------------------
# SIMPLE DEMO RUN
# ---------------------------------------
if __name__ == "__main__":
    print("Amazon Nova Reel Video Generator Demo")
    print("-" * 40)

    user_prompt = input("Enter video prompt: ")

    generate_video(user_prompt)
