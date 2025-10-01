"""Safe batch processing utilities for multiple files."""

from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import text_analysis, file_utils


def batch_process_files(files: List[Path], operation: str) -> List[Dict[str, Any]]:
    """
    Process multiple files with the specified operation.
    
    Args:
        files: List of file paths to process
        operation: Operation to perform (analyze, clean, info)
    
    Returns:
        List of results for each file
    """
    results = []
    
    for file_path in files:
        try:
            if operation == "analyze":
                result = analyze_file(file_path)
            elif operation == "clean":
                result = clean_file(file_path)
            elif operation == "info":
                result = get_file_summary(file_path)
            else:
                result = {
                    'file': str(file_path),
                    'status': 'error',
                    'message': f'Unknown operation: {operation}'
                }
            
            results.append(result)
        
        except Exception as e:
            results.append({
                'file': str(file_path),
                'status': 'error',
                'message': f'Error processing file: {e}'
            })
    
    return results


def analyze_file(file_path: Path) -> Dict[str, Any]:
    """
    Analyze a single file and return summary statistics.
    
    Args:
        file_path: Path to the file to analyze
    
    Returns:
        Dictionary with analysis results
    """
    try:
        if not file_utils.is_text_file(file_path):
            return {
                'file': str(file_path),
                'status': 'skipped',
                'message': 'Not a text file'
            }
        
        content = file_path.read_text(encoding='utf-8')
        analysis = text_analysis.analyze_text(content)
        
        return {
            'file': str(file_path),
            'status': 'success',
            'analysis': {
                'word_count': analysis['word_count'],
                'character_count': analysis['character_count'],
                'reading_level': analysis['reading_level'],
                'flesch_score': analysis['flesch_reading_ease']
            }
        }
    
    except Exception as e:
        return {
            'file': str(file_path),
            'status': 'error',
            'message': str(e)
        }


def clean_file(file_path: Path) -> Dict[str, Any]:
    """
    Clean a file and return processing summary.
    
    Args:
        file_path: Path to the file to clean
    
    Returns:
        Dictionary with cleaning results
    """
    try:
        if not file_utils.is_text_file(file_path):
            return {
                'file': str(file_path),
                'status': 'skipped',
                'message': 'Not a text file'
            }
        
        original_content = file_path.read_text(encoding='utf-8')
        cleaned_content = file_utils.clean_text(original_content)
        
        # Create backup and save cleaned version
        backup_path = file_path.with_suffix(file_path.suffix + '.bak')
        backup_path.write_text(original_content, encoding='utf-8')
        file_path.write_text(cleaned_content, encoding='utf-8')
        
        return {
            'file': str(file_path),
            'status': 'success',
            'message': f'Cleaned and backed up to {backup_path.name}',
            'original_size': len(original_content),
            'cleaned_size': len(cleaned_content),
            'reduction': len(original_content) - len(cleaned_content)
        }
    
    except Exception as e:
        return {
            'file': str(file_path),
            'status': 'error',
            'message': str(e)
        }


def get_file_summary(file_path: Path) -> Dict[str, Any]:
    """
    Get summary information about a file.
    
    Args:
        file_path: Path to the file
    
    Returns:
        Dictionary with file summary
    """
    try:
        info = file_utils.get_file_info(file_path)
        
        return {
            'file': str(file_path),
            'status': 'success',
            'info': info
        }
    
    except Exception as e:
        return {
            'file': str(file_path),
            'status': 'error',
            'message': str(e)
        }


def display_batch_results(results: List[Dict[str, Any]], console: Console) -> None:
    """
    Display batch processing results in a formatted table.
    
    Args:
        results: List of processing results
        console: Rich console for output
    """
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return
    
    # Create summary statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    
    # Summary table
    summary_table = Table(title="📊 Batch Processing Summary", border_style="green")
    summary_table.add_column("Status", style="bold")
    summary_table.add_column("Count", style="cyan")
    summary_table.add_column("Percentage", style="yellow")
    
    total = len(results)
    summary_table.add_row("✅ Success", str(success_count), f"{success_count/total*100:.1f}%")
    summary_table.add_row("⏭️ Skipped", str(skipped_count), f"{skipped_count/total*100:.1f}%")
    summary_table.add_row("❌ Errors", str(error_count), f"{error_count/total*100:.1f}%")
    summary_table.add_row("📁 Total", str(total), "100.0%")
    
    console.print(summary_table)
    
    # Detailed results table
    details_table = Table(title="📋 Detailed Results", border_style="cyan")
    details_table.add_column("File", style="bold blue")
    details_table.add_column("Status", style="bold")
    details_table.add_column("Details", style="cyan")
    
    for result in results:
        file_name = Path(result['file']).name
        status = result['status']
        
        if status == 'success':
            status_display = "[green]✅ Success[/green]"
            if 'analysis' in result:
                # Analysis results
                analysis = result['analysis']
                details = f"Words: {analysis['word_count']}, Reading: {analysis['reading_level']}"
            elif 'info' in result:
                # File info results
                info = result['info']
                details = f"Size: {info.get('size_kb', 0)} KB, Type: {info.get('extension', 'unknown')}"
            elif 'original_size' in result:
                # Cleaning results
                reduction = result['reduction']
                details = f"Reduced by {reduction} chars, backup created"
            else:
                details = "Processed successfully"
        
        elif status == 'error':
            status_display = "[red]❌ Error[/red]"
            details = result.get('message', 'Unknown error')
        
        elif status == 'skipped':
            status_display = "[yellow]⏭️ Skipped[/yellow]"
            details = result.get('message', 'File skipped')
        
        else:
            status_display = f"[white]{status}[/white]"
            details = result.get('message', 'No details')
        
        details_table.add_row(file_name, status_display, details)
    
    console.print(details_table)
    
    # Show any errors in detail
    errors = [r for r in results if r['status'] == 'error']
    if errors:
        console.print("\n[bold red]Error Details:[/bold red]")
        for error in errors:
            console.print(f"[red]• {Path(error['file']).name}:[/red] {error.get('message', 'Unknown error')}")


def safe_bulk_operation(
    directory: Path,
    pattern: str,
    operation: str,
    max_files: int = 100,
    dry_run: bool = False
) -> List[Dict[str, Any]]:
    """
    Safely perform bulk operations with safety checks.
    
    Args:
        directory: Directory to process
        pattern: File pattern to match
        operation: Operation to perform
        max_files: Maximum number of files to process (safety limit)
        dry_run: If True, only show what would be done
    
    Returns:
        List of results
    """
    files = list(directory.glob(pattern))
    
    if len(files) > max_files:
        raise ValueError(f"Too many files ({len(files)}). Maximum allowed: {max_files}")
    
    if dry_run:
        return [{
            'file': str(f),
            'status': 'dry_run',
            'message': f'Would perform {operation} operation'
        } for f in files]
    
    return batch_process_files(files, operation)


def create_processing_report(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Create a detailed processing report and save to file.
    
    Args:
        results: Processing results
        output_path: Path to save the report
    """
    report_lines = [
        "# Omni AI Batch Processing Report",
        "",
        f"Total files processed: {len(results)}",
        f"Generated on: {Path().cwd()}",
        "",
        "## Summary",
        ""
    ]
    
    # Add summary statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = sum(1 for r in results if r['status'] == 'error')
    skipped_count = sum(1 for r in results if r['status'] == 'skipped')
    
    report_lines.extend([
        f"- ✅ Successful: {success_count}",
        f"- ⏭️ Skipped: {skipped_count}",
        f"- ❌ Errors: {error_count}",
        "",
        "## Detailed Results",
        ""
    ])
    
    # Add detailed results
    for result in results:
        file_name = Path(result['file']).name
        status = result['status']
        message = result.get('message', 'No details')
        
        report_lines.append(f"### {file_name}")
        report_lines.append(f"- Status: {status}")
        if message != 'No details':
            report_lines.append(f"- Details: {message}")
        report_lines.append("")
    
    # Save report
    report_content = "\n".join(report_lines)
    output_path.write_text(report_content, encoding='utf-8')