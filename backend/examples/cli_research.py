from dotenv import load_dotenv
import argparse
import os
import logging
from langchain_core.messages import HumanMessage
from agent.graph import graph
from agent.rag_manager import initialize_rag_database

load_dotenv()

# Set up logger
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the research agent from the command line."""
    parser = argparse.ArgumentParser(description="Run the LangGraph research agent")
    parser.add_argument("question", help="Research question")
    parser.add_argument(
        "--initial-queries",
        type=int,
        default=3,
        help="Number of initial search queries",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        default=2,
        help="Maximum number of research loops",
    )
    parser.add_argument(
        "--reasoning-model",
        default="qwen3",
        help="Model for the final answer",
    )
    args = parser.parse_args()

    # Initialize RAG database before running the graph
    print("🔧 Setting up research agent...")
    initialize_rag_database(
        persist_directory="../chroma_db",  # Adjusted path for examples directory
        papers_directory="../paper",    # Adjusted path for examples directory
        code_directory="../code"        # Adjusted path for examples directory
    )

    state = {
        "messages": [HumanMessage(content=args.question)],
        "initial_search_query_count": args.initial_queries,
        "max_research_loops": args.max_loops,
        "reasoning_model": args.reasoning_model,
    }

    print(f"❓ Research question: {args.question}")
    print("🔬 Starting research...")
    result = graph.invoke(state)
    messages = result.get("messages", [])
    if messages:
        print("\n" + "="*50)
        print("📋 RESEARCH RESULT:")
        print("="*50)
        print(messages[-1].content)
    else:
        print("⚠️  No result generated")


if __name__ == "__main__":
    main()
