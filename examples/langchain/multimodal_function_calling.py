import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from gemini_multimodal import GeminiMultiModal

load_dotenv()

# Initialize enhanced Gemini
llm = GeminiMultiModal(model="gemini-1.5-pro-latest")

# Example 1: Multimodal processing
def process_image_with_text():
    image_url = "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg"
    messages = [
        HumanMessage(content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ])
    ]
    response = llm.invoke(messages)
    print("Multimodal response:", response.content)

# Example 2: Function calling
def use_function_calling():
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("human", "What's the weather like in San Francisco?")
    ])
    
    chain = prompt | llm.bind_functions(tools) | StrOutputParser()
    response = chain.invoke({})
    print("Function calling response:", response)

if __name__ == "__main__":
    process_image_with_text()
    use_function_calling()