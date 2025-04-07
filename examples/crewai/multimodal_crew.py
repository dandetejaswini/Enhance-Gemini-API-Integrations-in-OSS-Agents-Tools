import os
from dotenv import load_dotenv
from crewai import Crew, Task
from gemini_agent import GeminiAgent

load_dotenv()

def run_multimodal_crew():
    # Create multimodal researcher
    researcher = GeminiAgent(
        role="Senior Researcher",
        goal="Analyze complex data including images and text",
        backstory="Expert in multimodal AI analysis",
        model="gemini-1.5-pro-latest",
        multimodal=True
    )
    
    # Create standard writer agent
    writer = GeminiAgent(
        role="Content Writer",
        goal="Write engaging content",
        backstory="Skilled writer who creates compelling narratives",
        model="gemini-1.5-pro-latest"
    )
    
    # Multimodal research task
    research_task = Task(
        description=[
            {"type": "text", "text": "Analyze this image of a historical event"},
            {"type": "image_url", "image_url": {"url": "https://example.com/historical-event.jpg"}}
        ],
        expected_output="Detailed analysis of the historical event shown in the image",
        agent=researcher
    )
    
    # Writing task
    write_task = Task(
        description="Write a blog post about the historical event analysis",
        expected_output="Engaging 500-word blog post suitable for general audience",
        agent=writer,
        context=[research_task]
    )
    
    # Create and run crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=2
    )
    
    result = crew.kickoff()
    print("Crew execution result:", result)

def run_function_calling_crew():
    # Create agent with function calling
    analyst = GeminiAgent(
        role="Data Analyst",
        goal="Fetch and analyze data",
        backstory="Expert in data analysis and API integrations",
        model="gemini-1.5-pro-latest"
    ).function_calling([
        {
            "name": "get_stock_data",
            "description": "Fetch stock market data",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol"},
                    "days": {"type": "integer", "description": "Number of days"}
                },
                "required": ["symbol"]
            }
        }
    ])
    
    analysis_task = Task(
        description="Analyze Apple stock performance over last 30 days",
        expected_output="Detailed analysis with key insights",
        agent=analyst
    )
    
    crew = Crew(agents=[analyst], tasks=[analysis_task])
    result = crew.kickoff()
    print("Function calling crew result:", result)

if __name__ == "__main__":
    # run_multimodal_crew()  # Would need actual image URL
    run_function_calling_crew()