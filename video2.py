import boto3
import json
import time
import os


# ---------------------------------------
# CONFIG
# ---------------------------------------
REGION = "us-east-1"
MODEL_ID = "amazon.nova-reel-v1:0"

S3_BUCKET = "your_s3_bucket_name"   # must exist
S3_PREFIX = "videos/"
OUTPUT_FILE = "output.mp4"


# ---------------------------------------
# CLIENTS
# ---------------------------------------
bedrock = boto3.client("bedrock-runtime", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)


# ---------------------------------------
# GENERATE VIDEO (ASYNC)
# ---------------------------------------
def generate_video(prompt: str):
    print("[INFO] Starting Nova Reel async job...")

    start_response = bedrock.start_async_invoke(
        modelId=MODEL_ID,
        modelInput={
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {
                "text": prompt
            },
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720",
                "seed": 1
            }
        },
        outputDataConfig={
            "s3OutputDataConfig": {
                "s3Uri": f"s3://{S3_BUCKET}/{S3_PREFIX}"
            }
        }
    )

    invocation_arn = start_response["invocationArn"]
    print(f"[INFO] Invocation ARN: {invocation_arn}")

    # ---------------------------------------
    # POLL STATUS
    # ---------------------------------------
    while True:
        response = bedrock.get_async_invoke(invocationArn=invocation_arn)
        status = response["status"]
        print(f"[INFO] Job status: {status}")

        if status == "Completed":
            break
        if status == "Failed":
            raise RuntimeError(response)

        time.sleep(5)

    # ---------------------------------------
    # DOWNLOAD VIDEO FROM S3 (CORRECT WAY)
    # ---------------------------------------
    s3_uri = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    print(f"[INFO] Video stored at: {s3_uri}")

    # s3_uri example:
    # s3://nova-demo-workshop/videos/mvokszpb9acw
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1]

    video_key = f"{prefix}/output.mp4"

    print(f"[INFO] Downloading from s3://{bucket}/{video_key}")

    s3.download_file(bucket, video_key, OUTPUT_FILE)

    print(f"[SUCCESS] Video downloaded as {OUTPUT_FILE}")



# ---------------------------------------
# DEMO RUN
# ---------------------------------------
if __name__ == "__main__":
    print("Amazon Nova Reel Video Generator Demo")
    print("-" * 40)

    user_prompt = input("Enter video prompt: ")
    generate_video(user_prompt)
