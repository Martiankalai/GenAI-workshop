import time
import boto3


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS


# =========================
# ENV + AWS CONFIG
# =========================

REGION = "us-east-1"
MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

bedrock_client = boto3.client("bedrock-runtime", region_name=REGION)


# =========================
# CLAUDE CONVERSE CALL
# =========================
def call_claude(prompt: str) -> str:
    params = {
        "modelId": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "temperature": 0.0,
            "maxTokens": 4000
        },
    }

    try:
        t0 = time.time()
        resp = bedrock_client.converse(**params)
        print(f"[INFO] Claude call took {time.time() - t0:.2f}s")

        return resp["output"]["message"]["content"][0]["text"]

    except Exception as err:
        raise RuntimeError(f"Bedrock Converse error: {err}")


# =========================
# BUILD VECTOR STORE
# =========================
def build_vector_store(pdf_path: str) -> FAISS:
    """
    Load PDF → split → embed → FAISS
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=REGION
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# =========================
# ASK QUESTION (RAG)
# =========================
def ask_pdf_question(
    vector_store: FAISS,
    question: str,
    k: int = 3
) -> str:
    docs = vector_store.similarity_search(question, k=k)

    context = "\n\n".join(
        f"Page {doc.metadata.get('page')}:\n{doc.page_content}"
        for doc in docs
    )
    print(context)

    prompt = f"""
You are a document-based assistant.

Answer the question strictly using the context below.
If the answer is not present, say:
"I don't know based on the document."

Context:
{context}

Question:
{question}
"""

    return call_claude(prompt)


# =========================
# MAIN (CLI DEMO)
# =========================
if __name__ == "__main__":
    PDF_PATH = "resume.pdf"

    print("Building FAISS index...")
    vector_store = build_vector_store(PDF_PATH)

    print("RAG system ready. Ask questions!\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        answer = ask_pdf_question(vector_store, query)
        print("\nAnswer:\n", answer)
        print("-" * 60)
