from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    rag_search_query: Annotated[list, operator.add]  # RAG search queries
    rag_search_result: Annotated[list, operator.add]  # RAG search results
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    rag_results: Annotated[list, operator.add]
    rag_found: bool
    # Paper search related fields
    title: Annotated[list, operator.add]  # Detected paper titles
    paper_found: bool  # Whether papers were found in search
    papers_indexed: int  # Number of papers indexed into RAG
    paper_search_attempted: bool  # Whether paper search has been attempted
    is_paper_search: bool  # Whether current search is for papers
    paper_search_indicators: Annotated[list, operator.add]  # List of indicators that triggered paper search


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]
    is_paper_search: bool
    paper_search_indicators: list[str]


class WebSearchState(TypedDict):
    search_query: str
    id: str


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
