import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from omni_ai.core import text_analysis, file_utils, safe_processing

console = Console()
app = typer.Typer(
    name="omni-ai",
    help="🤖 Safe, ethical AI toolkit for text analysis, file processing, and automation.",
    add_completion=False,
    rich_markup_mode="rich"
)

SAFE_BANNER = Panel(
    "[bold green]Omni AI (Safe Mode)[/bold green]\n\n"
    "This tool focuses on [bold cyan]ethical, lawful, and beneficial[/bold cyan] use cases.\n"
    "Harmful functionality (e.g., cyber attacks, malware, unauthorized access) "
    "is [bold red]strictly out of scope[/bold red] and not supported.",
    title="🛡️ Safety Notice",
    border_style="green"
)


@app.command()
def hello(
    name: Optional[str] = typer.Argument(None, help="Name to greet")
) -> None:
    """👋 Print a friendly greeting."""
    if name:
        console.print(f"[bold green]Hello {name}![/bold green] 🤖")
    else:
        console.print("[bold green]Hello from Omni AI![/bold green] 🤖")
    console.print("[italic]Staying safe and helpful![/italic] ✨")


@app.command()
def analyze_text(
    text: Optional[str] = typer.Argument(None, help="Text to analyze"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File to analyze"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed analysis")
) -> None:
    """📊 Analyze text for readability, complexity, and statistics."""
    if file:
        if not file.exists():
            console.print(f"[bold red]Error:[/bold red] File {file} does not exist.")
            raise typer.Exit(1)
        text = file.read_text(encoding='utf-8')
    
    if not text:
        console.print("[bold red]Error:[/bold red] No text provided. Use argument or --file option.")
        raise typer.Exit(1)
    
    results = text_analysis.analyze_text(text, detailed=detailed)
    text_analysis.display_analysis(results, console)


@app.command()
def process_file(
    input_file: Path = typer.Argument(..., help="Input file to process"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    operation: str = typer.Option("clean", "--operation", "-op", help="Operation: clean, format, extract")
) -> None:
    """📁 Process files safely (clean text, extract content, format)."""
    if not input_file.exists():
        console.print(f"[bold red]Error:[/bold red] Input file {input_file} does not exist.")
        raise typer.Exit(1)
    
    result = file_utils.process_file(input_file, operation)
    
    if output_file:
        output_file.write_text(result, encoding='utf-8')
        console.print(f"[bold green]✓[/bold green] Processed content saved to {output_file}")
    else:
        console.print("[bold cyan]Processed Content:[/bold cyan]")
        console.print(Panel(result[:500] + "..." if len(result) > 500 else result, border_style="cyan"))


@app.command()
def batch_process(
    directory: Path = typer.Argument(..., help="Directory containing files to process"),
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File pattern to match"),
    operation: str = typer.Option("analyze", "--operation", "-op", help="Operation to perform")
) -> None:
    """🔄 Process multiple files in a directory."""
    if not directory.exists() or not directory.is_dir():
        console.print(f"[bold red]Error:[/bold red] Directory {directory} does not exist.")
        raise typer.Exit(1)
    
    files = list(directory.glob(pattern))
    if not files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No files matching '{pattern}' found.")
        return
    
    console.print(f"[bold green]Processing {len(files)} files...[/bold green]")
    
    results = safe_processing.batch_process_files(files, operation)
    safe_processing.display_batch_results(results, console)


@app.command()
def safety_check() -> None:
    """🛡️ Display safety guidelines and ethical use policy."""
    console.print(SAFE_BANNER)
    
    guidelines = Panel(
        "[bold cyan]✓[/bold cyan] Text analysis and processing\n"
        "[bold cyan]✓[/bold cyan] File content extraction and formatting\n"
        "[bold cyan]✓[/bold cyan] Automated content organization\n"
        "[bold cyan]✓[/bold cyan] Educational and research assistance\n\n"
        "[bold red]✗[/bold red] Harmful content generation\n"
        "[bold red]✗[/bold red] Privacy violations or data theft\n"
        "[bold red]✗[/bold red] Malicious code or exploits\n"
        "[bold red]✗[/bold red] Unauthorized system access",
        title="📋 Supported Use Cases",
        border_style="blue"
    )
    console.print(guidelines)


def main() -> None:
    """Main entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()
