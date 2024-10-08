# Importing basic libraries
import os
import pickle
import pandas as pd

# LlamaIndex - 0.10.65
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document, VectorStoreIndex, Settings

csv_path = "Main_Data.csv"  # Path to the CSV file

try:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"The file does not exist.")

    df = pd.read_csv(csv_path)

    document_content = df.to_string(index=False)

    docs = [Document(text=document_content)]  # Convert text content into a list of Document objects

    try:
        llm = Ollama(model="llama3.1:8b-instruct-q4_K_M", request_timeout=300.0) # Initialize LLM with spcific configuration
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the language model: {e}")

    try:
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5") # Initialize embedding model for document indexing
        # According to MTEB leaderboard, Many tasks -> Different parameters to test it
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the embedding model: {e}")

    # Configure indexing settings
    Settings.llm = llm # Setting llm as Ollama
    Settings.embed_model = embed_model # Settings for Embedding Model
    Settings.node_parser = SentenceSplitter(chunk_size=512) # Breaking down sentences into chunks, with size of 512 words
    Settings.num_output = 256 # No. of words model will give output at once.
    Settings.context_window = 4000 # Model's maximum capacity to accept token while processing or generating text 

    # Building VectorStoreIndex object from the documents
    try:
        index = VectorStoreIndex.from_documents(
            docs,
            llm=llm,
            embed_model=embed_model
        )
    except Exception as e:
        raise RuntimeError(f"Failed to build the VectorStoreIndex: {e}")

    # Save the created index to a file using pickle for later use
    try:
        with open("index.pkl", "wb") as f:
            pickle.dump(index, f)
        print("Indexing completed and saved as index.pkl")
    except Exception as e:
        raise RuntimeError(f"Failed to save the index to a file: {e}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
