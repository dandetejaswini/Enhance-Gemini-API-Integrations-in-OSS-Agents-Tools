import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def test_crewai_capabilities():
    """Test CrewAI's current Gemini support"""
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    
    researcher = Agent(
        role="Researcher",
        goal="Make amazing research",
        backstory="You're an expert researcher",
        llm=llm
    )
    
    task = Task(
        description="Investigate AI trends",
        agent=researcher
    )
    
    crew = Crew(agents=[researcher], tasks=[task])
    result = crew.kickoff()
    print("Basic agent execution works:", bool(result))
    
    # Try function calling - should fail
    try:
        researcher.bind_functions([{"name": "search", "description": "Search the web"}])
        print("Function calling: FAIL")
    except Exception as e:
        print("Function calling missing (expected):", str(e))

if __name__ == "__main__":
    test_crewai_capabilities()