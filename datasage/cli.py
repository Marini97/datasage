"""
Command-line interface for DataSage.

This module provides the main CLI commands for profiling datasets
and generating data quality reports.
"""

import sys
from pathlib import Path
from typing import Optional, List
import logging

import typer
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

from .profiler import profile_df
from .prompt_builder import PromptBuilder
from .model import LocalLLMGenerator
from .report_formatter import ReportFormatter
from .verifier import OutputVerifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="datasage",
    help="🔮 DataSage - A fun local AI data quality checker",
    no_args_is_help=True
)
console = Console()


@app.command()
def profile(
    csv_path: Path = typer.Argument(
        ..., 
        help="Path to the CSV file to profile",
        exists=True,
        file_okay=True,
        dir_okay=False
    ),
    output: Optional[Path] = typer.Option(
        None, 
        "-o", 
        "--output",
        help="Output path for the Markdown report (default: <input>_report.md)"
    ),
    columns: Optional[str] = typer.Option(
        None,
        "--columns",
        help="Comma-separated list of specific columns to profile"
    ),
    summary_only: bool = typer.Option(
        False,
        "--summary-only",
        help="Generate only executive summary (faster, good for quick checks)"
    ),
    no_verify: bool = typer.Option(
        False,
        "--no-verify",
        help="Disable numeric verification of LLM outputs"
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show debug output (prompts, raw AI responses, etc.)"
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override the default LLM model"
    ),
    max_rows: Optional[int] = typer.Option(
        None,
        "--max-rows",
        help="Maximum number of rows to process (for large datasets)"
    )
):
    """Profile a CSV file and generate a data quality report."""
    
    # Set up output path
    if output is None:
        output = csv_path.parent / f"{csv_path.stem}_report.md"
    
    # Set debug logging if requested
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: Read CSV
            task1 = progress.add_task("Reading CSV file...", total=None)
            try:
                df = pd.read_csv(csv_path)
                if max_rows and len(df) > max_rows:
                    rprint(f"[yellow]Dataset has {len(df):,} rows. Sampling {max_rows:,} rows for analysis.[/yellow]")
                    df = df.sample(n=max_rows, random_state=42)
                progress.update(task1, description=f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                console.print(f"[red]Error reading CSV file: {e}[/red]")
                raise typer.Exit(1)
            
            # Filter columns if specified
            if columns:
                column_list = [col.strip() for col in columns.split(',')]
                missing_cols = [col for col in column_list if col not in df.columns]
                if missing_cols:
                    console.print(f"[red]Error: Columns not found: {', '.join(missing_cols)}[/red]")
                    raise typer.Exit(1)
                df = df[column_list]
                progress.update(task1, description=f"✓ Loaded {len(df):,} rows, {len(column_list)} selected columns")
            
            # Step 2: Profile the data
            task2 = progress.add_task("Computing data profile...", total=None)
            try:
                data_profile = profile_df(df)
                progress.update(task2, description="✓ Data profiling complete")
            except Exception as e:
                console.print(f"[red]Error profiling data: {e}[/red]")
                raise typer.Exit(1)
            
            # Step 3: Build prompts
            task3 = progress.add_task("Building LLM prompts...", total=None)
            try:
                prompt_builder = PromptBuilder()
                
                if summary_only:
                    prompts = [prompt_builder.build_summary_prompt(data_profile)]
                else:
                    prompts = prompt_builder.build_prompts(data_profile)
                
                progress.update(task3, description=f"✓ Generated {len(prompts)} prompt(s)")
                
                if debug:
                    import sys
                    for i, prompt in enumerate(prompts):
                        print(f"\n--- Prompt {i+1} ---", file=sys.stderr)
                        print(prompt, file=sys.stderr)
                        print(f"--- End Prompt {i+1} ---\n", file=sys.stderr)
                
            except Exception as e:
                console.print(f"[red]Error building prompts: {e}[/red]")
                raise typer.Exit(1)
            
            # Step 4: Generate LLM responses
            task4 = progress.add_task("Generating insights with local LLM...", total=None)
            try:
                llm_generator = LocalLLMGenerator(model_name=model_name)
                llm_outputs = []
                
                for i, prompt in enumerate(prompts):
                    progress.update(
                        task4, 
                        description=f"Generating insights... (chunk {i+1}/{len(prompts)})"
                    )
                    output_text = llm_generator.generate_markdown(prompt)
                    llm_outputs.append(output_text)
                    
                    if debug:
                        print(f"\n--- LLM Output {i+1} ---", file=sys.stderr)
                        print(output_text, file=sys.stderr)
                        print(f"--- End Output {i+1} ---\n", file=sys.stderr)
                
                progress.update(task4, description="✓ LLM generation complete")
                
            except Exception as e:
                console.print(f"[red]Error generating LLM responses: {e}[/red]")
                console.print(f"[yellow]Note: Make sure you have the required dependencies installed:[/yellow]")
                console.print(f"[yellow]  pip install transformers torch[/yellow]")
                raise typer.Exit(1)
            
            # Step 5: Format report
            task5 = progress.add_task("Formatting report...", total=None)
            try:
                formatter = ReportFormatter()
                report_title = f"Data Quality Report: {csv_path.name}"
                report_text = formatter.assemble_report(
                    llm_outputs, 
                    data_profile, 
                    title=report_title
                )
                progress.update(task5, description="✓ Report formatting complete")
            except Exception as e:
                console.print(f"[red]Error formatting report: {e}[/red]")
                raise typer.Exit(1)
            
            # Step 6: Verify and annotate (optional)
            if not no_verify:
                task6 = progress.add_task("Verifying numeric claims...", total=None)
                try:
                    verifier = OutputVerifier()
                    report_text, verification_results = verifier.verify_and_annotate_report(
                        report_text, data_profile, add_annotations=True
                    )
                    
                    verification_summary = verifier.get_verification_summary()
                    if verification_summary['total_claims_checked'] > 0:
                        success_rate = verification_summary['success_rate']
                        progress.update(
                            task6, 
                            description=f"✓ Verified {verification_summary['total_claims_checked']} claims ({success_rate:.1f}% accurate)"
                        )
                    else:
                        progress.update(task6, description="✓ No numeric claims to verify")
                        
                except Exception as e:
                    logger.warning(f"Verification failed: {e}")
                    progress.update(task6, description="⚠ Verification skipped due to error")
            
            # Step 7: Write output
            task7 = progress.add_task("Writing report...", total=None)
            try:
                with open(output, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                progress.update(task7, description=f"✓ Report saved to {output}")
            except Exception as e:
                console.print(f"[red]Error writing report: {e}[/red]")
                raise typer.Exit(1)
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)
    
    # Success message
    console.print(f"\n[green]✓ Data quality report generated successfully![/green]")
    console.print(f"[green]📄 Report saved to: {output}[/green]")
    
    # Display basic stats
    dataset_stats = data_profile.get('dataset', {})
    console.print(f"\n[dim]Dataset: {dataset_stats.get('n_rows', 0):,} rows × {dataset_stats.get('n_cols', 0)} columns[/dim]")
    
    if not no_verify and 'verification_summary' in locals():
        if verification_summary['total_claims_checked'] > 0:
            console.print(f"[dim]Verification: {verification_summary['success_rate']:.1f}% of numeric claims verified[/dim]")


@app.command()
def version():
    """Show DataSage version and a bit of personality."""
    from . import __version__
    console.print(f"🔮 DataSage {__version__}")
    console.print("Your friendly local AI data detective!")


@app.command()
def info():
    """Show system and model information (the nerdy details)."""
    console.print("[bold]🔮 DataSage System Info[/bold]\n")
    
    # Python and package versions
    console.print(f"Python: {sys.version.split()[0]}")
    
    try:
        import pandas as pd
        console.print(f"Pandas: {pd.__version__}")
    except ImportError:
        console.print("Pandas: [red]Not installed[/red]")
    
    try:
        import transformers
        console.print(f"Transformers: {transformers.__version__}")
    except ImportError:
        console.print("Transformers: [red]Not installed[/red]")
    
    try:
        import torch
        console.print(f"PyTorch: {torch.__version__}")
        console.print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        console.print("PyTorch: [red]Not installed[/red]")
    
    # Model configuration
    console.print("\n[bold]Model Configuration[/bold]")
    try:
        import torch
        generator = LocalLLMGenerator()
        model_info = generator.get_model_info()
        console.print(f"Default model: {model_info['model_name']}")
        
        # Show device detection
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            console.print(f"Will use: GPU ({device_name}) 🚀")
        else:
            console.print(f"Will use: CPU (still works!) 💪")
            
        console.print(f"Temperature: {model_info['generation_config']['temperature']}")
        console.print(f"Max tokens: {model_info['generation_config']['max_new_tokens']}")
    except Exception as e:
        console.print(f"[red]Error loading model info: {e}[/red]")


if __name__ == "__main__":
    app()
