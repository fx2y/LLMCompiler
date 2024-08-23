from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import StructuredTool
from typing_extensions import TypedDict


class SearchResult(TypedDict):
    content: str
    url: str


def search_engine() -> StructuredTool:
    def search(query: str) -> list[SearchResult]:
        """Search engine to the internet."""
        results = DuckDuckGoSearchAPIWrapper()._ddgs_text(query)
        return [{"content": r["body"], "url": r["href"]} for r in results]

    return StructuredTool.from_function(
        name="search",
        func=search,
        description="Tool that queries the DuckDuckGo search API and gets back json string."
    )
