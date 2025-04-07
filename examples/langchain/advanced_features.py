import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gemini_enhanced import EnhancedGeminiChat

load_dotenv()

def demo_enhanced_features():
    # Initialize with system instruction
    llm = EnhancedGeminiChat(
        model="gemini-1.5-pro-latest",
        temperature=0.7,
        max_output_tokens=2048
    ).with_system_instruction("You are a helpful AI assistant that specializes in technology topics.")
    
    # Example 1: Streaming
    def show_streaming():
        print("\n--- Streaming Example ---")
        llm.streaming = True
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Explain neural networks in detail")
        ])
        chain = prompt | llm | StrOutputParser()
        
        for chunk in chain.stream({}):
            print(chunk, end="", flush=True)
        print("\n")
    
    # Example 2: System instructions
    def show_system_instructions():
        print("\n--- System Instruction Example ---")
        tech_llm = EnhancedGeminiChat(
            model="gemini-1.5-pro-latest"
        ).with_system_instruction("You are a computer science expert. Provide detailed technical explanations.")
        
        bio_llm = EnhancedGeminiChat(
            model="gemini-1.5-pro-latest"
        ).with_system_instruction("You are a biology expert. Focus on biological concepts.")
        
        tech_response = tech_llm.invoke("Explain how transformers work")
        bio_response = bio_llm.invoke("Explain how transformers work")
        
        print("Technical response:", tech_response.content[:200] + "...")
        print("Biological response:", bio_response.content[:200] + "...")
    
    show_streaming()
    show_system_instructions()

if __name__ == "__main__":
    demo_enhanced_features()