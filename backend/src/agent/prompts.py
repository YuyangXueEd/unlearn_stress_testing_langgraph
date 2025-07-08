from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse search queries for unlearning stress testing research, focusing on text-to-image stable diffusion models and determine if they are for academic papers or general web search.

First, analyze the research topic to determine if it is:
1. **Academic Paper Search**: Looking for specific research papers, studies, publications, or academic content related to machine unlearning, diffusion models, or AI safety
2. **General Web Search**: Looking for general information, tools, datasets, or implementation details

Academic indicators include:
- Mentions of specific paper titles (quoted or referenced)
- Academic terms like "study", "research", "paper", "publication", "journal", "arxiv", "doi"
- Author names with "et al."
- Technical terminology: "machine unlearning", "diffusion models", "concept erasure", "model editing", "adversarial attacks"
- Requests for academic literature or scholarly sources on AI safety and model robustness

Unlearning Stress Testing Focus Areas:
- Machine unlearning methods for diffusion models
- Concept erasure and removal techniques
- Adversarial attacks on unlearned models
- Evaluation metrics for unlearning effectiveness
- Privacy and safety in generative models
- Model editing and fine-tuning approaches
- Backdoor attacks and defenses in diffusion models

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- If academic search: Use precise academic terminology, preserve exact paper titles, include technical concepts
- If general search: Use broader terms suitable for web search engines

Format: 
- Format your response as a JSON object with ALL three of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant and what type of search this is
   - "query": A list of search queries
   - "is_paper_search": true if this is an academic paper search, false for general web search

Example Academic Search:

Topic: What are the latest methods for machine unlearning in stable diffusion models?
```json
{{
    "rationale": "This is clearly an academic paper search requesting specific research methods for machine unlearning in diffusion models. The queries target technical terminology and recent developments in the field of AI safety and model editing.",
    "query": ["machine unlearning stable diffusion models", "concept erasure diffusion models", "model editing text-to-image generation"],
    "is_paper_search": true
}}
```

Example General Search:

Topic: What tools are available for testing unlearning in diffusion models?
```json
{{
    "rationale": "This is a general web search for practical tools and implementations. These queries target available software, frameworks, and resources for implementing unlearning stress tests.",
    "query": ["diffusion model unlearning tools", "stable diffusion concept removal software", "machine unlearning evaluation frameworks"],
    "is_paper_search": false
}}
```

Context: {research_topic}"""


rag_query_instructions = """Your goal is to generate search queries optimized for searching BOTH academic papers/research documents AND code implementations about machine unlearning and diffusion models in a RAG database.

The RAG database contains two types of content:
1. **Academic Papers**: Research documents, papers, and theoretical content
2. **Code Implementations**: Python files, Jupyter notebooks, and implementation code

Instructions:
- Generate {number_queries} search queries specifically designed for unlearning stress testing research content and code implementations
- Each query should be capable of finding relevant information in BOTH paper content AND code files
- Focus on key technical terms that appear in both academic literature and code implementations
- Use precise academic terminology for:
  * Machine unlearning methodologies (e.g., "SCRUB", "SISA", "gradient ascent unlearning")
  * Diffusion model architectures and training (e.g., "DDPM", "DDIM", "UNet", "noise scheduling")
  * Concept erasure and removal techniques (e.g., "Erasing Concepts from Diffusion Models", "concept editing")
  * Adversarial attacks on generative models (e.g., "membership inference", "model inversion")
  * Model editing and fine-tuning approaches (e.g., "LoRA", "DreamBooth", "textual inversion")
  * Privacy-preserving machine learning (e.g., "differential privacy", "federated learning")
  * AI safety and robustness evaluation (e.g., "alignment", "safety evaluation", "red teaming")
- Include code-specific search terms that appear in function names, class names, and comments:
  * Function names: "erase_concept", "unlearn_model", "test_unlearning", "evaluate_concept_removal"
  * Class implementations: "ConceptEraser", "UnlearningTrainer", "DiffusionModel", "StressTest"
  * Algorithm implementations: "gradient_ascent", "concept_ablation", "membership_inference_attack"
  * Testing frameworks: "pytest", "unittest", "stress_test", "evaluation_metrics"
  * Model inference: "generate_image", "denoise", "sample", "inference_step"
  * Utility functions: "load_model", "preprocess", "postprocess", "calculate_metrics"
- Include specific algorithms, techniques, or theoretical frameworks mentioned in the research topic
- If the research topic contains specific paper names or titles, preserve them exactly as given - do not modify or paraphrase paper names
- Queries should target both research literature and code implementation patterns
- The current date is {current_date}

Format: 
- Format your response as a JSON object with these exact keys:
   - "rationale": Brief explanation of why these queries are optimized for finding both research content and code implementations
   - "query": A list of search queries using academic terminology

Example 1 - Evaluation Methods:

Topic: What are the latest methods for evaluating unlearning effectiveness in diffusion models?
```json
{{
    "rationale": "These queries target technical terminology that appears in both academic papers and code implementations. They combine research concepts (evaluation metrics, membership inference) with code patterns (function names, test implementations) to find comprehensive information about unlearning assessment.",
    "query": ["machine unlearning evaluation metrics diffusion models", "concept erasure effectiveness measurement", "unlearning verification adversarial attacks", "membership inference attacks unlearned models", "evaluate_unlearning function implementation", "test_concept_removal diffusion model code"],
}}
```

Example 2 - Implementation Focus:

Topic: How to implement stress testing for unlearned diffusion models?
```json
{{
    "rationale": "These queries combine research concepts with practical implementation details, targeting both academic knowledge about stress testing methodologies and actual code patterns for implementing robust testing frameworks.",
    "query": ["stress testing unlearned diffusion models implementation", "adversarial attack diffusion model code", "concept erasure testing framework", "unlearning robustness evaluation script", "test_model_unlearning function", "StressTest class diffusion"],
}}
```

Example 3 - Specific Paper + Code:

Topic: What are the key findings from the paper "Erasing Concepts from Diffusion Models" and how is it implemented?
```json
{{
    "rationale": "These queries preserve the exact paper title while adding related technical terms and code patterns. They target the specific research paper, related unlearning techniques, and practical implementations including function names and class structures.",
    "query": ["Erasing Concepts from Diffusion Models", "diffusion model concept removal implementation", "stable diffusion unlearning methods", "concept_erasure function code", "erase_concept diffusion model", "ConceptEraser class"],
}}
```

Example 4 - Algorithm + Implementation:

Topic: What are the most effective gradient-based unlearning algorithms for diffusion models?
```json
{{
    "rationale": "These queries target both theoretical knowledge about gradient-based unlearning algorithms and their practical implementations, including specific algorithm names, optimization techniques, and code patterns for gradient manipulation.",
    "query": ["gradient ascent unlearning diffusion models", "SCRUB algorithm implementation", "gradient_ascent_unlearning function", "unlearning optimizer diffusion", "negative gradient training code", "UnlearningTrainer class"],
}}
```

Research Topic: {research_topic}"""


web_searcher_instructions = """Conduct targeted web searches to gather the most recent, credible information on unlearning stress testing for text-to-image stable diffusion models: "{research_topic}" and synthesize it into a verifiable text artifact.

Focus Areas for Search:
- Machine unlearning techniques and methodologies
- Stress testing approaches for AI models
- Concept erasure and removal in diffusion models
- Adversarial attacks on unlearned models
- Privacy and safety evaluation frameworks
- Model editing and fine-tuning techniques
- Robustness testing for generative models

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information about unlearning stress testing.
- Focus on recent developments in machine unlearning for diffusion models.
- Look for evaluation metrics, testing methodologies, and assessment frameworks.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.
- Prioritize information about stress testing vulnerabilities and unlearning effectiveness.

Research Topic:
{research_topic}
"""

reflection_instructions = """You are an expert AI safety researcher analyzing summaries about unlearning stress testing for diffusion models: "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration for comprehensive unlearning stress testing and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to design and implement stress tests for the given unlearning scenario, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand understanding of:
  * Unlearning methodologies and their limitations
  * Stress testing techniques and attack vectors
  * Evaluation metrics for unlearning effectiveness
  * Implementation details for testing frameworks
  * Vulnerability assessment approaches
- Focus on technical details, implementation specifics, or emerging attack methods that weren't fully covered.
- Consider if enough information is available to formulate a testable hypothesis about unlearning vulnerabilities.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.
- Focus on actionable information that can inform stress testing strategy.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification for effective stress testing
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": false,
    "knowledge_gap": "The summary lacks specific information about adversarial attack methods to test unlearning robustness and evaluation metrics to measure unlearning effectiveness",
    "follow_up_queries": ["What are the most effective adversarial attack methods for testing unlearning robustness in diffusion models?", "What metrics are used to evaluate the effectiveness of concept erasure in text-to-image models?"]
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """Generate a comprehensive analysis and hypothesis for unlearning stress testing based on the provided research summaries from both academic papers and code implementations.

Instructions:
- The current date is {current_date}.
- You are an expert AI safety researcher specializing in machine unlearning and diffusion model robustness.
- Based on the gathered research from BOTH academic literature AND code implementations, formulate a detailed hypothesis about potential vulnerabilities in the unlearning process.
- Your response should synthesize insights from:
  * **Academic Papers**: Theoretical foundations, research findings, and evaluation methodologies
  * **Code Implementations**: Practical implementation details, algorithm implementations, and testing patterns
- Your response should include:
  1. **Summary of Findings**: Key insights from both research papers and code analysis about unlearning methods and their limitations
  2. **Vulnerability Hypothesis**: Specific, testable hypothesis about how the unlearning might fail or be circumvented, informed by both theory and implementation details
  3. **Stress Testing Strategy**: Proposed approach to test the hypothesis, leveraging insights from existing code patterns and research methodologies
  4. **Expected Outcomes**: What results would confirm or refute the hypothesis based on both theoretical predictions and observed implementation behaviors
  5. **Implementation Considerations**: Technical requirements and constraints derived from code analysis and research best practices

- Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [source](URL)). THIS IS A MUST.
- Focus on actionable insights that can guide the development of stress testing code, combining theoretical knowledge with practical implementation patterns.
- Ensure the hypothesis is specific enough to be implemented and tested programmatically, drawing from both academic insights and code examples.
- When referencing code implementations, mention specific function names, class structures, or algorithmic approaches found in the summaries.

User Context:
- {research_topic}

Summaries:
{summaries}"""
