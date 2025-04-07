# Gemini Integration Project

## Overview

This project enhances the integration of Google's Gemini AI models across three popular AI frameworks:
- LangChain
- LlamaIndex
- CrewAI

The implementation provides:
1. **Multimodal support** (text + images)
2. **Function calling** capabilities
3. **Advanced configurations** (safety settings, generation parameters)
4. **System instructions** support
5. **Performance optimizations** for each framework

## Table of Contents

- [Setup Instructions](#setup-instructions)
- [Framework Integrations](#framework-integrations)
  - [LangChain](#langchain-integration)
  - [LlamaIndex](#llamaindex-integration)
  - [CrewAI](#crewai-integration)
- [Key Features](#key-features)
- [Examples](#examples)
- [Testing](#testing)
- [Contributing](#contributing)


## Setup Instructions

### Prerequisites
- Python 3.9+
- Google API key with Gemini access
- Git

### Installation

```bash
# Create project directory
mkdir gemini-integration-project
cd gemini-integration-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install google-generativeai python-dotenv

# Clone all repositories
git clone https://github.com/langchain-ai/langchain.git
git clone https://github.com/jmorganca/llamaindex.git
git clone https://github.com/joaomdmoura/crewai.git

# Install each package in development mode
cd langchain && pip install -e . && cd ..
cd llamaindex && pip install -e . && cd ..
cd crewai && pip install -e . && cd ..
```

### Environment Configuration
Create a `.env` file:
```
GOOGLE_API_KEY=your_api_key_here
```

## Framework Integrations

### LangChain Integration

#### Features Added:
- Multimodal processing (text + images)
- Function calling support
- System instructions
- Streaming responses
- Enhanced configuration options

#### Usage Examples:

**Basic Text Generation:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro")
response = llm.invoke("Explain quantum computing")
print(response.content)
```

**Multimodal Processing:**
```python
from gemini_multimodal import GeminiMultiModal

llm = GeminiMultiModal(model="gemini-1.5-pro-latest")
image_url = "https://example.com/image.jpg"
response = llm.invoke([
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": image_url}}
])
```

### LlamaIndex Integration

#### Features Added:
- Multimodal document indexing
- System prompt support
- Enhanced retrieval capabilities
- Specialized knowledge configurations

#### Usage Examples:

**Basic Query Engine:**
```python
from llama_index.llms import Gemini
from llama_index import VectorStoreIndex, SimpleDirectoryReader

llm = Gemini(model="gemini-pro")
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What's in the documents?")
```

**Multimodal Indexing:**
```python
from gemini_multimodal import GeminiMultiModal

llm = GeminiMultiModal(model="gemini-1.5-pro-latest")
documents = [
    {
        "text": "This is a document about AI",
        "image": "https://example.com/ai-image.jpg"
    }
]
index = MultiModalVectorStoreIndex.from_documents(documents)
```

### CrewAI Integration

#### Features Added:
- Multimodal agent tasks
- Function calling in agents
- System instructions per agent
- Enhanced model configuration

#### Usage Examples:

**Basic Crew Setup:**
```python
from crewai import Agent, Task, Crew
from gemini_agent import GeminiAgent

researcher = GeminiAgent(
    role="Researcher",
    goal="Make amazing research",
    backstory="You're an expert researcher",
    model="gemini-pro"
)

task = Task(description="Investigate AI trends", agent=researcher)
crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

**Multimodal Crew:**
```python
researcher = GeminiAgent(
    role="Senior Researcher",
    goal="Analyze complex data including images and text",
    backstory="Expert in multimodal AI analysis",
    model="gemini-1.5-pro-latest",
    multimodal=True
)

research_task = Task(
    description=[
        {"type": "text", "text": "Analyze this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ],
    agent=researcher
)
```

## Key Features

### 1. Multimodal Support
- Process both text and images in all frameworks
- Unified interface across LangChain, LlamaIndex, and CrewAI
- Support for local files and remote URLs

### 2. Function Calling
- Define and bind functions to models
- Automatic parameter extraction
- Integration with agent workflows

### 3. Advanced Configurations
- System instructions/prompts
- Fine-tuned generation parameters
- Safety settings customization
- Streaming responses

### 4. Performance Optimizations
- Efficient message handling
- Batch processing support
- Caching mechanisms
- Error handling and retries

## Examples

See the `examples/` directory for complete implementation examples:

```
examples/
├── langchain/
│   ├── multimodal_function_calling.py
│   └── advanced_features.py
├── llamaindex/
│   ├── multimodal_retrieval.py
│   └── advanced_features.py
└── crewai/
    ├── multimodal_crew.py
    └── advanced_features.py
```

## Testing

Run tests to verify functionality:

```bash
# LangChain tests
python -m pytest tests/langchain/

# LlamaIndex tests
python -m pytest tests/llamaindex/

# CrewAI tests
python -m pytest tests/crewai/
```

Test coverage includes:
- Basic functionality
- Multimodal processing
- Function calling
- Error handling
- Performance benchmarks

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

### Contribution Guidelines:
- Follow existing code style
- Include tests for new features
- Update documentation
- Maintain backward compatibility


## Roadmap

### Short-term
- [ ] Add video support
- [ ] Improve error handling
- [ ] Add more examples

### Long-term
- [ ] Support for Gemini Ultra
- [ ] Integration with additional frameworks
- [ ] Benchmarking suite

## Support

For issues or questions, please open an issue on the GitHub repository.