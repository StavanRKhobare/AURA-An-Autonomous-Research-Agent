import arxiv
import time
from typing import List, Dict

# ArXiv API limit: maximum 1 request per 3 seconds
ARXIV_RATE_LIMIT = 3.0

class ArxivSkill:
    def __init__(self):
        self.last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < ARXIV_RATE_LIMIT:
            time.sleep(ARXIV_RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def search(self, query: str, max_results: int = 5) -> List[Dict]:
        """
        Search arXiv for papers matching the query.
        """
        self._rate_limit()
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        results = []
        try:
            for result in client.results(search):
                results.append({
                    "id": result.entry_id,
                    "title": result.title,
                    "summary": result.summary,
                    "authors": [a.name for a in result.authors],
                    "published": result.published.isoformat(),
                    "pdf_url": result.pdf_url
                })
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")
            
        return results

if __name__ == "__main__":
    # Quick test
    skill = ArxivSkill()
    print("Searching for 'Graph Neural Networks in Supply Chain'...")
    papers = skill.search("all:Graph Neural Networks Supply Chain", max_results=2)
    for p in papers:
        print(f"\nTitle: {p['title']}")
        print(f"URL: {p['pdf_url']}")
