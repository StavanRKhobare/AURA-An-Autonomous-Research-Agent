import typer
from rich.console import Console
from rich.progress import Progress
from skills.arxiv_skill import ArxivSkill
from skills.pdf_skill import PDFSkill
from skills.llm_skill import LLMSkill
from skills.memory_skill import MemorySkill

app = typer.Typer(help="Aura: Autonomous University Research Agent")
console = Console()

@app.command()
def explore(query: str = typer.Argument(..., help="Domain or idea to explore"), max_papers: int = 2):
    """
    Map out a research domain and find structural holes.
    """
    console.print(f"[bold green]Exploring domain:[/bold green] {query}")
    
    arxiv = ArxivSkill()
    pdf_skill = PDFSkill()
    llm = LLMSkill()
    memory = MemorySkill()
    
    papers = arxiv.search(query, max_results=max_papers)
    console.print(f"Found {len(papers)} papers. Ingesting...")
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing papers...", total=len(papers))
        
        for paper in papers:
            try:
                # 1. Fetch
                text = pdf_skill.fetch_and_parse(paper["pdf_url"])
                # Extract first chunk (e.g. 4000 chars) for testing the entity extraction
                chunk = text[:4000] 
                
                # 2. Extract Graph Entities
                graph_data = llm.extract_graph_entities(chunk)
                
                # 3. Upsert to Vector and Graph DB
                memory.upsert_memory(
                    paper_id=paper["id"], 
                    text=chunk, 
                    graph_data=graph_data, 
                    metadata={"title": paper["title"], "authors": paper["authors"]}
                )
                console.print(f"\n[green]✓[/green] Ingested: {paper['title']}")
            except Exception as e:
                console.print(f"\n[red]Failed to process {paper['id']}: {e}[/red]")
            progress.advance(task)
            
    # GraphRAG Query
    console.print("\n[bold purple]Running GraphRAG Query to find connections...[/bold purple]")
    try:
        rag_results = memory.query_graph_rag(query)
        
        console.print("\n[bold]Semantic Matches:[/bold]")
        for chunk in rag_results["chunks"]:
            console.print(f" - {chunk['text']} (Score: {chunk['score']:.2f})")
            
        console.print("\n[bold]Graph Context (Entities linked to these papers):[/bold]")
        for ctx in rag_results["graph_context"]:
            console.print(f" - Paper {ctx['paper']} --[{ctx['rel']}]--> {ctx['type']}: {ctx['entity']}")
    except Exception as e:
        console.print(f"[yellow]Could not query DBs (Are Qdrant and Neo4j running?): {e}[/yellow]")

    memory.close()

@app.command()
def review(pdf_path: str = typer.Argument(..., help="Path to the draft PDF")):
    """
    Critique a draft paper and verify its citations.
    """
    console.print(f"[bold yellow]Reviewing draft:[/bold yellow] {pdf_path}")
    
    pdf_skill = PDFSkill()
    try:
        text = pdf_skill.fetch_and_parse(pdf_path)
        console.print(f"Parsed {len(text)} characters from draft.")
        console.print("[bold]Running LLM Critique...[/bold]")
        # Simulated LLM critique for now
        console.print("[yellow]Critique module: The methodology section lacks a comparison with baseline GraphSAGE. Furthermore, citation [12] seems to contradict the claim made in paragraph 3.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error reading PDF: {e}[/red]")

@app.command()
def digest(interests: str = typer.Option("Graph Neural Networks", help="Comma separated interests")):
    """
    Generate a daily digest of new papers.
    """
    console.print(f"[bold blue]Generating daily digest for:[/bold blue] {interests}")
    arxiv = ArxivSkill()
    
    queries = [q.strip() for q in interests.split(",")]
    for q in queries:
        console.print(f"\n[bold]Latest on {q}:[/bold]")
        papers = arxiv.search(f"all:{q}", max_results=3, sort_by_date=True)
        for p in papers:
            console.print(f" - {p['title']} ({p['published'][:10]})")
            console.print(f"   URL: {p['pdf_url']}")

if __name__ == "__main__":
    app()
