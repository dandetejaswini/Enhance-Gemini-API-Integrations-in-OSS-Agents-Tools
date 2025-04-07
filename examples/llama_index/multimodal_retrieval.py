import os
from dotenv import load_dotenv
from llama_index.llms import ChatMessage, MessageRole
from gemini_multimodal import GeminiMultiModal
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.multi_modal_engine import MultiModalVectorStoreIndex

load_dotenv()

# Initialize enhanced Gemini
llm = GeminiMultiModal(model="gemini-1.5-pro-latest")

# Example 1: Multimodal chat
def multimodal_chat():
    image_url = "https://storage.googleapis.com/generativeai-downloads/images/scones.jpg"
    messages = [
        ChatMessage(
            role=MessageRole.USER,
            content=[
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        )
    ]
    response = llm.chat(messages)
    print("Multimodal chat response:", response.message.content)

# Example 2: Multimodal document indexing
def multimodal_indexing():
    # Create a simple multimodal index
    documents = [
        {
            "text": "This is a document about AI",
            "image": "https://example.com/ai-image.jpg"  # Would need actual image
        }
    ]
    
    # In practice, you'd implement proper document loading
    index = MultiModalVectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(llm=llm)
    
    response = query_engine.query(
        [
            {"type": "text", "text": "What does this image show about AI?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/ai-image.jpg"}}
        ]
    )
    print("Multimodal query response:", response)

if __name__ == "__main__":
    multimodal_chat()
    # multimodal_indexing()  # Would need actual implementation