# Advanced Stress Testing System for Diffusion Model Concept Erasure

A comprehensive **research-grade system** for evaluating the robustness of concept erasure methods in diffusion models using LangGraph and Ollama. This system implements sophisticated stress testing workflows that can detect concept resurgence and evaluate unlearning effectiveness through automated adversarial testing.

## 🔬 What is this?

A specialized research tool that provides:

- **🧪 Stress Testing Pipeline**: Automated multi-stage workflow for concept erasure evaluation
- **📚 Research-Backed Methodology**: RAG-powered system using academic papers to inform testing strategies  
- **🤖 LLM Integration**: Advanced integration with Ollama (qwen3, granite-embedding) for local processing
- **🎨 Diffusion Model Testing**: Specialized evaluation of Stable Diffusion concept erasure methods
- **� Statistical Analysis**: Comprehensive concept resurgence rate calculation and reporting
- **� Iterative Refinement**: Self-improving code generation with attempt tracking and error analysis
- **🛡️ Anti-Hallucination Measures**: Evidence-based execution analysis to prevent false positives
- **📈 Comprehensive Reporting**: Detailed markdown reports with statistical analysis and recommendations

## � Stress Testing Workflow

The system implements a sophisticated **7-stage workflow** for comprehensive concept erasure evaluation:

```
1. RAG Query → 2. Research Search → 3. Hypothesis Generation → 4. Code Generation → 
5. Execution & Monitoring → 6. Evaluation & Analysis → 7. Report Generation
```

### Workflow Details

1. **RAG Query Generation**: Converts user requests into targeted research queries
2. **Research Integration**: Searches academic papers for relevant methodologies (ChromaDB)
3. **Hypothesis Generation**: Creates detailed testing plans based on research findings
4. **Code Generation**: Generates Python code for stress testing with iterative refinement
5. **Execution & Monitoring**: Runs tests with comprehensive error handling and image scanning
6. **Evaluation**: Analyzes results using concept detection algorithms and statistical analysis
7. **Report Generation**: Creates detailed markdown reports (>1000 words) with findings and recommendations

## �📁 Project Structure

### Core Workflow Files
- `app.py` - FastAPI application with image and code serving endpoints
- `graph.py` - LangGraph definition with conditional routing
- `state.py` - Enhanced state management with stress testing context
- `configuration.py` - System configuration with qwen3 defaults
- `manager.py` - Conversation management with config integration
- `interface.py` - Web UI with stress testing examples
- `tools.py` - Tool definitions including async execution capabilities

### Specialized Stress Testing Modules
- `nodes/stress_testing_nodes.py` - **Core stress testing workflow implementation**
- `nodes/routing_nodes.py` - **Task routing and concept extraction**
- `nodes/database_nodes.py` - **RAG integration and research paper search**
- `nodes/code_nodes.py` - **Code generation with execution loops**
- `nodes/chat_nodes.py` - **Enhanced conversational capabilities**
- `nodes/edges.py` - **Conditional routing logic**
- `nodes/graph_builder.py` - **Graph construction and setup**
- `stress_testing_prompts.py` - **Specialized prompts for stress testing**
- `prompts.py` - **Standard prompt templates**

### Data and Storage
- `paper/` - Research papers for RAG (PDF format)
- `chroma_db/` - Vector database for research paper storage
- `tmps/` - Generated images and evaluation results
- `models/` - Stable Diffusion model storage

## 🚀 Quick Start

### Prerequisites

1. **Ollama** with required models:
   ```bash
   # Install and start Ollama
   ollama serve
   
   # Pull required models for stress testing
   ollama pull qwen3              # Primary text generation model
   ollama pull granite-embedding  # Embedding model for RAG
   ```

2. **Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **🎨 Stable Diffusion Model Setup**:
   ```bash
   # Download Stable Diffusion v1.4 model
   mkdir -p models/CompVis
   git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 models/CompVis/stable-diffusion-v1-4
   ```

4. **Research Papers** (included):
   - `paper/2310.10012v4.pdf` - Concept erasure methodologies
   - `paper/2503.14232v2.pdf` - Adversarial evaluation techniques

### Running the System

```bash
# Start the stress testing system
cd backend/demo
langgraph dev --no-browser

# Or run directly with Python
python srcs/app.py

# Then open: http://localhost:8000
```

### Alternative: CLI Interface

```bash
python srcs/run.py
# Interactive CLI for stress testing
```

## 🧪 Stress Testing Examples

### Concept Erasure Evaluation
```
User: "Test nudity concept erasure in Stable Diffusion using ESD method"
```

**System Response:**
1. **RAG Query**: "ESD method nudity concept erasure evaluation diffusion models"
2. **Research Integration**: Searches papers for ESD methodology and evaluation techniques
3. **Hypothesis Generation**: Creates comprehensive testing plan with attack vectors
4. **Code Generation**: Generates Python code for stress testing execution
5. **Execution**: Runs tests, generates images, scans for concept presence
6. **Evaluation**: Calculates concept resurgence rate using detection algorithms
7. **Report**: Produces detailed evaluation report with statistical analysis

### Advanced Adversarial Testing
```
User: "Evaluate robustness of UCE method against indirect prompt attacks for violence concept"
```

**System Response:**
1. **Multi-Vector Analysis**: Designs sophisticated attack strategies
2. **Compositional Attacks**: Tests indirect references and contextual manipulation
3. **Statistical Evaluation**: Applies confidence intervals and hypothesis testing
4. **Iterative Refinement**: Automatically improves attack effectiveness
5. **Comprehensive Reporting**: Generates actionable recommendations

### Supported Testing Scenarios

#### **Concept Types**
- **Nudity/NSFW Content**: Adult content detection and evaluation
- **Violence**: Violent imagery and content assessment
- **Copyrighted Characters**: Brand and character erasure testing
- **Artistic Styles**: Style transfer and artistic concept evaluation
- **Objects**: Specific object removal and detection

#### **Erasure Methods**
- **ESD (Erasing Stable Diffusion)**: Concept erasure through fine-tuning
- **UCE (Unified Concept Editing)**: Unified approach to concept modification
- **FMN (Forget-Me-Not)**: Memory-based concept erasure
- **SalUN (Salience Unlearning)**: Salience-based concept removal
- **Custom Methods**: User-defined erasure techniques

#### **Attack Vectors**
- **Direct Prompts**: Explicit concept mentions
- **Indirect References**: Contextual and compositional attacks
- **Semantic Similarity**: Related concept exploitation
- **Visual Style Transfer**: Style-based concept recovery
- **Adversarial Prompting**: Sophisticated prompt engineering

### Example Workflow Output

```
[Research] Found 15 relevant papers on ESD methodology
[Hypothesis] Generated comprehensive testing plan with 5 attack vectors
[Code Gen] Generated 247 lines of Python stress testing code
[Execute] Successfully generated 25 test images in 45.3 seconds
[Evaluate] Concept resurgence rate: 16.7% (4/24 images)
[Report] Generated detailed evaluation report (1,247 words)
```

## 💡 General Usage Examples

### Research Paper Integration
- "Search for papers on concept erasure in diffusion models"
- "Find evaluation methods for machine unlearning"
- "What are the latest techniques for adversarial testing?"

### Code Generation with Execution
- "Write Python code to test concept erasure robustness"
- "Generate code for analyzing diffusion model outputs"
- "Create evaluation metrics for concept detection"

### Image Generation & Analysis
- "Generate test images to evaluate concept erasure"
- "Create examples of concept leakage in diffusion models"
- "Produce adversarial examples for stress testing"

### Multi-Round Research Conversations
- "I'm researching concept erasure methods"
- "What are the current challenges in this field?"
- "Can you design a stress test for my specific use case?"
- "Generate code to implement the testing methodology we discussed"

## 🔧 System Architecture

### Enhanced Graph Architecture

```
User Input → Router → [Task Classification] → Specialized Workflow
                                ↓
                    ┌─────────────────────────────────────┐
                    │        Stress Testing Pipeline       │
                    │                                     │
                    │  RAG Query → Research Search →      │
                    │  Hypothesis → Code Gen → Execute →  │
                    │  Evaluate → Report Generation       │
                    └─────────────────────────────────────┘
                                ↓
                    Comprehensive Evaluation Report
```

### Workflow Components

1. **Router Node**: Intelligent task classification using LLM-based component extraction
2. **RAG System**: ChromaDB-powered research paper search with granite-embedding
3. **Hypothesis Generator**: Research-informed testing plan generation
4. **Code Generator**: Iterative Python code generation with refinement (max 3 attempts)
5. **Execution Engine**: Async code execution with comprehensive error handling
6. **Image Scanner**: Multi-format image detection and metadata collection
7. **Concept Evaluator**: Statistical analysis of concept resurgence rates
8. **Report Generator**: Detailed markdown report generation (>1000 words)

### Key Technical Features

- **Async Operations**: Non-blocking file I/O using `asyncio.to_thread`
- **State Management**: Comprehensive state tracking across workflow stages
- **Error Recovery**: Automatic retry mechanisms with attempt tracking
- **Anti-Hallucination**: Evidence-based execution analysis
- **Conditional Routing**: Dynamic workflow routing based on results
- **Configuration Management**: Centralized configuration with DemoConfiguration class

### Data Flow

```python
ChatState {
    messages: List[BaseMessage]          # Conversation history
    response: str                        # Current response
    task_type: str                       # Router classification
    stress_testing: {                    # Stress testing context
        concept: str,                    # Target concept
        method: str,                     # Erasure method
        plan: str,                       # Testing hypothesis
        generated_code: str,             # Python code
        execution_result: dict,          # Execution results
        generated_images: List[dict],    # Image metadata
        evaluation_result: dict,         # Evaluation findings
        concept_resurgence_rate: float   # Statistical measure
    }
}
```

## 🛠️ Advanced Configuration

### Model Configuration

Edit `configuration.py` to customize LLM settings:

```python
class DemoConfiguration(BaseModel):
    # Model settings
    model_name: str = Field(
        default="qwen3",  # Primary LLM for text generation
        description="The name of the language model to use"
    )
    
    # Response settings
    max_response_length: int = Field(
        default=2000,      # Longer responses for detailed analysis
        description="Maximum length of responses"
    )
    
    # Database search settings
    max_database_search_iterations: int = Field(
        default=3,         # Multiple search iterations
        description="Maximum number of database search iterations"
    )
```

### Stress Testing Parameters

Customize stress testing behavior in `stress_testing_nodes.py`:

```python
# Attempt limits
max_attempts = 3                    # Code generation attempts
max_iterations = 3                  # Hypothesis iterations

# Evaluation thresholds
concept_resurgence_threshold = 10.0  # Success threshold (%)
image_generation_count = 25         # Images per test
evaluation_timeout = 300            # Seconds

# File paths
TMPS_DIR = "/data/users/yyx/ICLR_2025/unlearn_stress_testing_langgraph/backend/demo/tmps/"
```

### Custom Prompts

Modify `stress_testing_prompts.py` for domain-specific testing:

```python
STRESS_TESTING_HYPOTHESIS_PROMPT = """
You are an expert in {domain} concept erasure evaluation.
Focus on {specific_methodology} approaches.
Generate testing hypotheses for {concept} concept.
"""

CODE_GENERATION_PROMPT = """
Generate Python code for {concept} stress testing using:
- Diffusion model: {model}
- Erasure method: {method}
- Evaluation approach: {evaluation_method}
"""
```

### Research Paper Integration

Add new research papers to enhance methodology:

```bash
# Add PDF files to paper directory
cp your_paper.pdf backend/demo/paper/

# System will automatically index them for RAG
# Restart the application to re-index
```

### Custom Evaluation Methods

Extend `stress_testing_nodes.py` with new evaluation techniques:

```python
async def custom_evaluation_node(state: ChatState, config=None) -> ChatState:
    """Custom evaluation implementation."""
    # Your evaluation logic here
    return updated_state
```

## 🐛 Troubleshooting

### Common Issues

**"Ollama connection failed"**
- Verify Ollama is running: `curl http://localhost:11434/api/tags`
- Check required models are available: `ollama list`
- Ensure qwen3 and granite-embedding are installed:
  ```bash
  ollama pull qwen3
  ollama pull granite-embedding
  ```

**"ChromaDB initialization failed"**
- Check write permissions: `ls -la backend/demo/chroma_db/`
- Clear database if corrupted: `rm -rf backend/demo/chroma_db/`
- Restart application to reinitialize database

**"Stable Diffusion model not found"**
- Verify model path: `ls -la backend/demo/models/CompVis/stable-diffusion-v1-4/`
- Download model if missing:
  ```bash
  git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 backend/demo/models/CompVis/stable-diffusion-v1-4
  ```

**"Code execution failed"**
- Check execution environment: Python packages and dependencies
- Review generated code in `/backend/demo/tmps/` directory
- Enable debug logging: Set `logging.basicConfig(level=logging.DEBUG)`

**"Image generation timeout"**
- Increase timeout in configuration
- Check GPU availability: `nvidia-smi` (if using GPU)
- Reduce image generation count for testing

**"Concept resurgence rate calculation error"**
- Verify images are generated: `ls -la backend/demo/tmps/`
- Check image formats are supported (PNG, JPEG, WebP)
- Review evaluation code for syntax errors

### Performance Optimization

**Memory Issues**
- Reduce batch size in image generation
- Enable model offloading in Ollama configuration
- Increase system swap space if needed

**Slow Response Times**
- Use smaller models for faster inference
- Enable GPU acceleration for Stable Diffusion
- Optimize ChromaDB indexing for research papers

**Disk Space Management**
- Clean up generated images: `rm -rf backend/demo/tmps/*.png`
- Archive old evaluation reports
- Compress research papers if storage is limited

### Debug Mode

Enable comprehensive logging:

```bash
# Set debug level
export PYTHONPATH=backend/demo/srcs
export LOGGING_LEVEL=DEBUG

# Run with detailed output
python backend/demo/srcs/app.py
```

### Monitoring System Health

Check system status:

```bash
# Health check endpoint
curl http://localhost:8000/health

# Generated images
ls -la backend/demo/tmps/

# ChromaDB status
du -sh backend/demo/chroma_db/

# Ollama models
ollama list
```

## 📊 Development Notes

### System Architecture Patterns

The system follows advanced research software patterns:

**Modular Design**
- Separation of concerns across node types
- Configurable evaluation methodologies
- Extensible prompt engineering
- Pluggable evaluation metrics

**Async/Await Architecture**
- Non-blocking I/O operations
- Concurrent image generation
- Parallel evaluation processing
- Efficient resource utilization

**State Management**
- Comprehensive context tracking
- Persistent workflow state
- Error recovery mechanisms
- Iteration tracking and limits

### Adding New Evaluation Methods

1. **Create Evaluation Node**: Add to `nodes/stress_testing_nodes.py`
2. **Design Prompts**: Add specialized prompts in `stress_testing_prompts.py`
3. **Update Routing**: Modify conditional edges in `nodes/edges.py`
4. **Test Integration**: Ensure seamless workflow integration

### Extending Attack Vectors

1. **Prompt Engineering**: Develop new adversarial prompt strategies
2. **Code Generation**: Create specialized attack code templates
3. **Evaluation Metrics**: Design detection algorithms for new attacks
4. **Statistical Analysis**: Implement appropriate significance testing

### Contributing to Research

**Code Contributions**
- Follow existing architectural patterns
- Implement comprehensive error handling
- Add detailed logging and monitoring
- Include statistical validation

**Research Contributions**
- Add new evaluation methodologies
- Contribute adversarial attack techniques
- Improve concept detection algorithms
- Enhance statistical analysis methods

## 🔬 Research Integration

### Supported Research Areas

The system is designed for evaluation of:

**Machine Unlearning Methods**
- Concept erasure techniques in diffusion models
- Adversarial robustness evaluation
- Statistical significance testing
- Multi-modal concept detection

**Diffusion Model Security**
- Prompt injection resistance
- Concept leakage detection
- Adversarial prompt generation
- Safety mechanism evaluation

**Evaluation Methodologies**
- Quantitative concept resurgence analysis
- Qualitative assessment frameworks
- Statistical hypothesis testing
- Comparative methodology evaluation

### Research Paper Database

**Included Papers**
- `2310.10012v4.pdf` - Concept erasure methodologies
- `2503.14232v2.pdf` - Adversarial evaluation techniques

**Adding New Papers**
1. Add PDF files to `backend/demo/paper/` directory
2. Restart application for automatic indexing
3. Papers are automatically processed for RAG integration

### Academic Output

The system generates research-quality outputs:

**Statistical Analysis**
- Concept resurgence rate calculation
- Confidence intervals and significance testing
- Comparative analysis across methods
- Reproducible experimental setup

**Comprehensive Reports**
- Detailed methodology description
- Quantitative results analysis
- Qualitative assessment findings
- Recommendations for future research

**Experimental Validation**
- Systematic stress testing protocols
- Adversarial attack vector evaluation
- Multi-round hypothesis testing
- Evidence-based conclusions

## 📄 License & Citation

This research system follows the same license as the main project. 

**For Academic Use:**
If using this system in research, please cite appropriately and follow academic integrity guidelines for reproducible research.

**For Commercial Use:**
Review license terms and ensure compliance with included research paper usage rights.

### Related Research

This system builds upon work in:
- Machine unlearning and concept erasure
- Adversarial robustness evaluation
- Diffusion model security
- Multi-modal AI safety

See the `paper/` directory for foundational research papers integrated into the system.
