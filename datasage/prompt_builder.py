"""
Prompt building module for DataSage.

This module converts statistical profiles into concise, factual prompts
that stay under token limits for local LLM processing.
"""

from typing import Dict, Any, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts from data profile statistics."""
    
    def __init__(self, max_tokens_per_prompt: int = 800, columns_per_chunk: int = 6):
        """Initialize the prompt builder.
        
        Args:
            max_tokens_per_prompt: Maximum tokens per prompt chunk
            columns_per_chunk: Number of columns to include per prompt
        """
        self.max_tokens_per_prompt = max_tokens_per_prompt
        self.columns_per_chunk = columns_per_chunk
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation).
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def format_dataset_overview(self, dataset_stats: Dict[str, Any]) -> str:
        """Format dataset-level statistics into prompt text.
        
        Args:
            dataset_stats: Dataset statistics from profiler
            
        Returns:
            Formatted dataset overview text
        """
        overview = f"""Dataset Overview:
• {dataset_stats['n_rows']:,} rows and {dataset_stats['n_cols']} columns
• Memory usage: {dataset_stats['memory_usage_mb']:.1f} MB
• Duplicate rows: {dataset_stats['duplicate_rows_pct']:.1f}%"""
        
        return overview
    
    def format_column_stats(self, column: Dict[str, Any]) -> str:
        """Format a single column's statistics into prompt text.
        
        Args:
            column: Column statistics from profiler
            
        Returns:
            Formatted column statistics text
        """
        name = column['name']
        col_type = column['inferred_type']
        non_null_pct = column['non_null_pct']
        n_unique = column['n_unique']
        
        # Base information
        lines = [
            f"• {name} ({col_type}): {non_null_pct:.1f}% non-null, {n_unique:,} unique values"
        ]
        
        # Type-specific details
        if col_type == "numeric":
            if 'min' in column and 'max' in column:
                lines.append(f"  Range: {column['min']:.2f} to {column['max']:.2f}")
            if 'mean' in column:
                lines.append(f"  Mean: {column['mean']:.2f}")
            if 'outliers_iqr_count' in column and column['outliers_iqr_count'] > 0:
                lines.append(f"  Outliers: {column['outliers_iqr_count']} values")
            if 'zeros_pct' in column and column['zeros_pct'] > 5:
                lines.append(f"  Zero values: {column['zeros_pct']:.1f}%")
        
        elif col_type == "categorical":
            if 'top_k_values' in column and column['top_k_values']:
                top_val = column['top_k_values'][0]
                lines.append(f"  Most common: '{top_val['value']}' ({top_val['frequency_pct']:.1f}%)")
            if 'rare_values_count' in column and column['rare_values_count'] > 0:
                lines.append(f"  Rare values: {column['rare_values_count']} categories")
        
        elif col_type == "datetime":
            if 'min_date' in column and 'max_date' in column:
                lines.append(f"  Date range: {column['min_date'][:10]} to {column['max_date'][:10]}")
            if 'coverage_days' in column:
                lines.append(f"  Coverage: {column['coverage_days']} days")
        
        elif col_type == "text":
            if 'avg_length' in column:
                lines.append(f"  Average length: {column['avg_length']:.1f} characters")
            if 'pct_empty_strings' in column and column['pct_empty_strings'] > 5:
                lines.append(f"  Empty strings: {column['pct_empty_strings']:.1f}%")
        
        # Quality issues
        if column.get('quality_issues'):
            issues_text = self._format_quality_issues(column['quality_issues'])
            if issues_text:
                lines.append(f"  Issues: {issues_text}")
        
        return "\n".join(lines)
    
    def _format_quality_issues(self, issues: List[str]) -> str:
        """Format quality issues into readable text.
        
        Args:
            issues: List of quality issue flags
            
        Returns:
            Formatted issues text
        """
        issue_descriptions = {
            "high_missing": "high missing values",
            "likely_id": "appears to be an ID column",
            "extreme_outliers": "extreme outliers detected",
            "skewed": "skewed distribution",
            "many_rare_categories": "many rare categories",
            "mostly_empty": "mostly empty text"
        }
        
        readable_issues = []
        for issue in issues:
            if issue in issue_descriptions:
                readable_issues.append(issue_descriptions[issue])
            else:
                readable_issues.append(issue.replace("_", " "))
        
        return ", ".join(readable_issues)
    
    def create_prompt_template(self) -> str:
        """Create the base prompt template for data analysis.
        
        Returns:
            Prompt template string
        """
        return """You are a data quality expert. Analyse the following dataset statistics and create a clear, non-technical summary in Markdown format.

Focus on:
1. Overall dataset characteristics
2. Data quality issues that need attention
3. Column-by-column insights in plain English

Use this exact structure:

## Dataset Overview
[Brief description of the dataset size, structure, and general characteristics]

## Data Quality Summary
[Bullet points highlighting key quality issues or positive findings]

## Column Profiles
[For each column, provide a short paragraph explaining what the data represents and any quality concerns]

Statistics:
{statistics}

Write in UK English. Be concise but informative. Focus on actionable insights for non-technical readers."""
    
    def chunk_columns(self, columns: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split columns into chunks for processing.
        
        Args:
            columns: List of column statistics
            
        Returns:
            List of column chunks
        """
        chunks = []
        for i in range(0, len(columns), self.columns_per_chunk):
            chunk = columns[i:i + self.columns_per_chunk]
            chunks.append(chunk)
        return chunks
    
    def build_prompts(self, profile: Dict[str, Any]) -> List[str]:
        """Build prompts from a complete data profile.
        
        Args:
            profile: Complete profile from profiler.profile_df()
            
        Returns:
            List of prompts for LLM processing
        """
        dataset_stats = profile.get('dataset', {})
        columns = profile.get('columns', [])
        
        if not columns:
            # Handle empty dataset
            simple_stats = self.format_dataset_overview(dataset_stats)
            template = self.create_prompt_template()
            return [template.format(statistics=simple_stats)]
        
        # Sort columns by issues (problematic columns first)
        def column_priority(col):
            issues = col.get('quality_issues', [])
            return (len(issues), col['name'])
        
        sorted_columns = sorted(columns, key=column_priority, reverse=True)
        
        # Split into chunks
        column_chunks = self.chunk_columns(sorted_columns)
        prompts = []
        
        for i, chunk in enumerate(column_chunks):
            # Format statistics for this chunk
            stats_lines = [self.format_dataset_overview(dataset_stats)]
            stats_lines.append("")  # Empty line
            stats_lines.append("Column Details:")
            
            for column in chunk:
                stats_lines.append(self.format_column_stats(column))
            
            statistics_text = "\n".join(stats_lines)
            
            # Check token count
            template = self.create_prompt_template()
            full_prompt = template.format(statistics=statistics_text)
            
            if self.estimate_tokens(full_prompt) > self.max_tokens_per_prompt:
                logger.warning(
                    f"Prompt chunk {i+1} exceeds token limit "
                    f"({self.estimate_tokens(full_prompt)} > {self.max_tokens_per_prompt})"
                )
            
            prompts.append(full_prompt)
        
        return prompts
    
    def build_summary_prompt(self, profile: Dict[str, Any]) -> str:
        """Build a summary-only prompt for quick overview.
        
        Args:
            profile: Complete profile from profiler.profile_df()
            
        Returns:
            Summary prompt for LLM processing
        """
        dataset_stats = profile.get('dataset', {})
        columns = profile.get('columns', [])
        
        # Count issues across all columns
        all_issues = []
        for column in columns:
            all_issues.extend(column.get('quality_issues', []))
        
        issue_summary = {}
        for issue in all_issues:
            issue_summary[issue] = issue_summary.get(issue, 0) + 1
        
        # Format summary statistics
        stats_lines = [self.format_dataset_overview(dataset_stats)]
        stats_lines.append("")
        
        if issue_summary:
            stats_lines.append("Data Quality Issues:")
            for issue, count in sorted(issue_summary.items()):
                readable_issue = self._format_quality_issues([issue])
                stats_lines.append(f"• {count} columns with {readable_issue}")
        else:
            stats_lines.append("• No significant data quality issues detected")
        
        statistics_text = "\n".join(stats_lines)
        
        # Use a simpler template for summaries optimized for T5
        template = """Analyse this dataset: {rows} rows, {columns} columns. 
{statistics}

Write brief analysis covering: dataset overview, data quality status, recommendations if needed."""
        
        return template.format(
            rows=dataset_stats.get('num_rows', 'unknown'),
            columns=dataset_stats.get('num_columns', 'unknown'),
            statistics=statistics_text
        )
