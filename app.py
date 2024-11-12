from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import requests

app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
MEDI_API = os.environ.get('MEDI_API')

# Set Pinecone and Groq API keys in the environment
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Download embeddings for the model
embeddings = download_hugging_face_embeddings()

# Set Pinecone index name
index_name = "medicalbot"

# Create a Pinecone vector store from the existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Set up the retriever for similar documents
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define system prompt for consistent chatbot behavior
system_prompt = (
    "You are a medical assistant chatbot for question-answering tasks. "
    "Use concise and clear language to provide helpful information based on context. "
    "If you don't know the answer, say that you don't know."
)

# Function to interact with the Groq API
def get_groq_response(user_input):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MEDI_API}",
        "Content-Type": "application/json"
    }
    
    # Retrieve similar documents as context
    context_documents = retriever.get_relevant_documents(user_input)

    # Use the correct attribute to access content, e.g., 'page_content'
    context = "\n".join([doc.page_content for doc in context_documents])  # Changed to 'page_content'

    # Format prompt with system prompt and context
    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + context},
        {"role": "user", "content": user_input}
    ]

    data = {
        "model": "llama3-70b-8192",  # Replace with your specific model ID
        "messages": messages,
        "temperature": 0.7
    }

    # Send request to Groq API
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Sorry, there was an error with the Groq API."

# Define the main route
@app.route("/")
def index():
    return render_template('chat.html')

# Define the chatbot endpoint
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User Input:", msg)
    response = get_groq_response(msg)
    print("Response:", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
