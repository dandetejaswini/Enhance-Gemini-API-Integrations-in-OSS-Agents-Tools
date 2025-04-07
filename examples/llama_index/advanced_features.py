import os
from dotenv import load_dotenv
from llama_index.llms import ChatMessage, MessageRole
from gemini_enhanced import EnhancedGeminiLLM
from llama_index import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()

def demo_enhanced_features():
    # Initialize with system prompt
    llm = EnhancedGeminiLLM(
        model="gemini-1.5-pro-latest",
        temperature=0.7,
        max_tokens=2048
    ).with_system_prompt("You are an expert research assistant. Provide detailed, accurate information.")
    
    # Example 1: Enhanced RAG
    def enhanced_rag():
        print("\n--- Enhanced RAG Example ---")
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(llm=llm)
        
        response = query_engine.query("What are the key points in the documents?")
        print("Enhanced RAG response:", response.response)
    
    # Example 2: Specialized knowledge
    def specialized_knowledge():
        print("\n--- Specialized Knowledge Example ---")
        tech_llm = EnhancedGeminiLLM(
            model="gemini-1.5-pro-latest"
        ).with_system_prompt("You are a computer science professor. Explain concepts technically.")
        
        simple_llm = EnhancedGeminiLLM(
            model="gemini-1.5-pro-latest"
        ).with_system_prompt("Explain like I'm 5 years old.")
        
        tech_response = tech_llm.complete("Explain quantum computing")
        simple_response = simple_llm.complete("Explain quantum computing")
        
        print("Technical explanation:", tech_response.text[:200] + "...")
        print("Simple explanation:", simple_response.text[:200] + "...")
    
    enhanced_rag()
    specialized_knowledge()

if __name__ == "__main__":
    demo_enhanced_features()