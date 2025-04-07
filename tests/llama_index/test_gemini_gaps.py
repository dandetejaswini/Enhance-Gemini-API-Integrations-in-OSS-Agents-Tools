import os
from dotenv import load_dotenv
# from llama_index.llms.google import GoogleGenerativeAI
from llama_index.llms.gemini import Gemini
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings

load_dotenv()

def test_llamaindex_capabilities():
    """Test LlamaIndex's Gemini support with minimal dependencies"""
    try:
        # Initialize LLM with current recommended client
        llm = Gemini(model="models/gemini-1.5-pro-latest")
        
        # Test 1: Basic completion
        response = llm.complete("Explain AI like I'm 5")
        print("Basic completion works:", response.text[:100] + "...")
        
        # Test 2: Multimodal (expected to fail)
        try:
            llm.complete(["Describe this image", "image_data"])
            print("Unexpected multimodal success - may need verification")
        except Exception as e:
            print("Multimodal blocked (expected):", str(e))
        
        # Test 3: Retrieval with default embeddings
        os.makedirs("test_data", exist_ok=True)
        with open("test_data/sample.txt", "w") as f:
            f.write("AI is machines doing smart things like humans.")
        
        # Use default local embeddings (no extra dependencies)
        Settings.embed_model = "local"
        documents = SimpleDirectoryReader("test_data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query("What can AI do?")
        
        print("Retrieval works:", str(response)[:100] + "...")
        return True
        
    except Exception as e:
        print("Test failed:", str(e))
        print("\nTroubleshooting steps:")
        print("1. Install required packages:")
        print("   pip install llama-index-llms-google-genai")
        print("2. Verify .env has:")
        print("   GOOGLE_API_KEY=your_key")
        print("3. Try alternative model names:")
        print("   - 'models/gemini-1.0-pro'")
        print("   - 'models/gemini-1.5-pro-latest'")
        return False

if __name__ == "__main__":
    print("Running Gemini Integration Tests")
    test_llamaindex_capabilities()