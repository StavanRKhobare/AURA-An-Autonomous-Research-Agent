import arxiv
import time
import urllib.request
from typing import List, Dict

# Configure global urllib opener to use a custom User-Agent
# arXiv aggressively blocks default python-urllib user agents with HTTP 429
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'AuraResearchBot/1.0 (stavan@example.com)')]
urllib.request.install_opener(opener)

# ArXiv API limit: maximum 1 request per 5 seconds
ARXIV_RATE_LIMIT = 5.0

class ArxivSkill:
    def __init__(self):
        self.last_request_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < ARXIV_RATE_LIMIT:
            time.sleep(ARXIV_RATE_LIMIT - elapsed)
        self.last_request_time = time.time()

    def search(self, query: str, max_results: int = 5, sort_by_date: bool = False) -> List[Dict]:
        """
        Search arXiv for papers matching the query.
        """
        self._rate_limit()
        
        # Configure the built-in arxiv client with robust settings to avoid 429 errors
        client = arxiv.Client(
            page_size=max_results, # Only fetch what we need per page (instead of default 100)
            delay_seconds=ARXIV_RATE_LIMIT,
            num_retries=5
        )
        
        sort_criterion = arxiv.SortCriterion.SubmittedDate if sort_by_date else arxiv.SortCriterion.Relevance
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
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
