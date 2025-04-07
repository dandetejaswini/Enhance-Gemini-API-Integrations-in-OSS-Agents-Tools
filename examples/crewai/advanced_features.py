import os
from dotenv import load_dotenv
from crewai import Crew, Task
from gemini_enhanced import EnhancedGeminiAgent

load_dotenv()

def demo_enhanced_features():
    # Example 1: Specialized agents
    def specialized_agents():
        print("\n--- Specialized Agents Example ---")
        researcher = EnhancedGeminiAgent(
            role="AI Research Scientist",
            goal="Make breakthrough discoveries in AI",
            backstory="Expert in cutting-edge AI research",
            model="gemini-1.5-pro-latest"
        ).with_system_instruction("You are a top-tier AI researcher. Provide detailed, technical explanations.")
        
        writer = EnhancedGeminiAgent(
            role="Technical Writer",
            goal="Create engaging technical content",
            backstory="Skilled at explaining complex topics simply",
            model="gemini-1.5-pro-latest"
        ).with_system_instruction("Write in clear, accessible language for general audience.")
        
        research_task = Task(
            description="Research the latest advancements in transformer architectures",
            expected_output="Detailed technical report on 3 key advancements",
            agent=researcher
        )
        
        write_task = Task(
            description="Write a blog post about transformer advancements for non-experts",
            expected_output="Engaging 800-word blog post",
            agent=writer,
            context=[research_task]
        )
        
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, write_task],
            verbose=2
        )
        
        result = crew.kickoff()
        print("Crew result:", result)
    
    # Example 2: Model configuration
    def model_configuration():
        print("\n--- Model Configuration Example ---")
        analyst = EnhancedGeminiAgent(
            role="Data Analyst",
            goal="Analyze complex datasets",
            backstory="Expert in data analysis and visualization",
            model="gemini-1.5-pro-latest"
        ).with_model_config({
            "temperature": 0.3,
            "max_output_tokens": 1024,
            "top_p": 0.95
        })
        
        task = Task(
            description="Analyze this sales data and identify key trends",
            expected_output="Detailed analysis with 3 key insights",
            agent=analyst
        )
        
        result = analyst.execute_task(task.description)
        print("Analysis result:", result)
    
    specialized_agents()
    model_configuration()

if __name__ == "__main__":
    demo_enhanced_features()