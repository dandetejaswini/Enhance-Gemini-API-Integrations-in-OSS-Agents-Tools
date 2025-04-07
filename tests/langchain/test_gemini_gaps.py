import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_current_capabilities():
    """Test what currently works with LangChain's Gemini integration"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    
    # Test basic text generation (Works)
    response = llm.invoke("Explain quantum computing simply")
    print("Basic text generation works:", bool(response.content))
    
    # Multimodal
    try:
        llm.invoke(["What's in this image?", "image_data_here"])
        print("Multimodal support: FAIL")
    except Exception as e:
        print("Multimodal support missing (expected):", str(e))
    
    # Function calling 
    try:
        llm.bind_functions([{"name": "get_weather", "description": "Get weather", "parameters": {}}])
        print("Function calling: FAIL")
    except Exception as e:
        print("Function calling missing (expected):", str(e))

if __name__ == "__main__":
    test_current_capabilities()