import typer
from rich.console import Console

app = typer.Typer(help="Aura: Autonomous University Research Agent")
console = Console()

@app.command()
def explore(query: str = typer.Argument(..., help="Domain or idea to explore")):
    """
    Map out a research domain and find structural holes.
    """
    console.print(f"[bold green]Exploring domain:[/bold green] {query}")

@app.command()
def review(pdf_path: str = typer.Argument(..., help="Path to the draft PDF")):
    """
    Critique a draft paper and verify its citations.
    """
    console.print(f"[bold yellow]Reviewing draft:[/bold yellow] {pdf_path}")

@app.command()
def digest():
    """
    Generate a daily digest of new papers.
    """
    console.print("[bold blue]Generating daily digest...[/bold blue]")

if __name__ == "__main__":
    app()
