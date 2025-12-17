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

MODEL_ID = "amazon.titan-image-generator-v2:0"


# ---------------------------------------
# GENERATE IMAGE USING TITAN
# ---------------------------------------
def generate_image(prompt: str, output_file: str = "output.png"):
    request_body = {
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {
            "text": prompt
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
            "seed": 42
        }
    }

    try:
        start_time = time.time()

        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(request_body),
            accept="application/json",
            contentType="application/json"
        )

        print(f"[INFO] Titan image generation took {time.time() - start_time:.2f}s")

        response_body = json.loads(response["body"].read())

        # Extract base64 image
        image_base64 = response_body["images"][0]
        image_bytes = base64.b64decode(image_base64)

        # Save image
        with open(output_file, "wb") as f:
            f.write(image_bytes)

        print(f"[SUCCESS] Image saved as {output_file}")

    except Exception as e:
        raise RuntimeError(f"Bedrock Titan error: {str(e)}")


# ---------------------------------------
# SIMPLE DEMO RUN
# ---------------------------------------
if __name__ == "__main__":
    print("AWS Titan Image Generator Demo")
    print("-" * 35)

    prompt = input("Enter image prompt: ")

    generate_image(prompt)
