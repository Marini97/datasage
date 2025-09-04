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
        n_rows = dataset_stats.get('n_rows', 0)
        n_cols = dataset_stats.get('n_cols', 0)
        memory_mb = dataset_stats.get('memory_usage_mb', 0)
        duplicate_pct = dataset_stats.get('duplicate_rows_pct', 0)
        
        overview = f"""Dataset Overview:
• {n_rows:,} rows and {n_cols} columns
• Memory usage: {memory_mb:.1f} MB
• Duplicate rows: {duplicate_pct:.1f}%"""
        
        # Add more context for better analysis
        if n_rows > 0:
            avg_memory_per_row = memory_mb / n_rows * 1024 if n_rows > 0 else 0  # KB per row
            overview += f"\n• Average memory per row: {avg_memory_per_row:.2f} KB"
        
        if duplicate_pct > 5:
            overview += f"\n• NOTE: High duplicate percentage may indicate data quality issues"
        
        return overview
    
    def format_exploration_insights(self, insights: Dict) -> str:
        """Format data exploration insights for LLM context.
        
        Args:
            insights: Dictionary of exploration insights from get_data_exploration_insights
            
        Returns:
            Formatted insights text for LLM prompt
        """
        insights_text = "\n## Data Science Exploration Insights:\n"
        
        # Numerical insights
        if insights.get('numerical_summary') and insights['numerical_summary'].get('columns'):
            num_cols = insights['numerical_summary']['columns']
            insights_text += f"\n• {len(num_cols)} numerical columns with distributions analyzed"
            
            # Skewness information
            if 'skewness' in insights['numerical_summary']:
                high_skew = [col for col in num_cols 
                           if abs(insights['numerical_summary']['skewness'][col]) > 1]
                if high_skew:
                    insights_text += f"\n• Highly skewed distributions: {', '.join(high_skew)}"
        
        # Categorical insights
        if insights.get('categorical_summary'):
            cat_count = len(insights['categorical_summary'])
            insights_text += f"\n• {cat_count} categorical columns analyzed"
            
            # High cardinality warning
            high_card = [col for col, info in insights['categorical_summary'].items()
                        if info['unique_values'] > 50]
            if high_card:
                insights_text += f"\n• High cardinality categoricals: {', '.join(high_card)}"
        
        # Correlation insights
        if insights.get('correlations') is not None:
            high_correlations = []
            corr_matrix = insights['correlations']
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_correlations.append(
                            f"{corr_matrix.columns[i]}-{corr_matrix.columns[j]} ({corr_matrix.iloc[i, j]:.2f})"
                        )
            
            if high_correlations:
                insights_text += f"\n• Strong correlations detected: {', '.join(high_correlations[:3])}"
                if len(high_correlations) > 3:
                    insights_text += f" and {len(high_correlations)-3} more"
        
        # Outlier insights
        if insights.get('outliers'):
            high_outlier_cols = [col for col, count in insights['outliers'].items() if count > 0]
            if high_outlier_cols:
                total_outliers = sum(insights['outliers'].values())
                insights_text += f"\n• {total_outliers} outliers detected across {len(high_outlier_cols)} columns"
        
        # Missing data insights
        if insights.get('missing_patterns'):
            total_missing = insights['missing_patterns']['total_missing']
            if total_missing > 0:
                missing_cols = len(insights['missing_patterns']['missing_by_column'])
                insights_text += f"\n• {total_missing} missing values across {missing_cols} columns"
        
        # Key recommendations
        if insights.get('recommendations'):
            insights_text += f"\n• {len(insights['recommendations'])} data quality recommendations identified"
        
        return insights_text
    
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
    
    def create_few_shot_example(self) -> str:
        """Create a few-shot example to guide the model's output format.
        
        Returns:
            Example analysis to guide model output
        """
        return """
EXAMPLE OUTPUT FORMAT:

## Dataset Overview
This dataset contains 1,500 customer records with 8 columns covering demographics, sales, and engagement metrics. The data represents 18 months of customer activity with a mix of numerical, categorical, and date fields totaling 2.3 MB.

## Data Quality Assessment

**Completeness**: The dataset is 92% complete overall, with phone numbers missing in 15% of records and email addresses missing in 8% of records. Critical fields like customer ID and purchase amount are fully populated.

**Consistency**: Age values show some outliers (3 customers with age > 100), and there are 12 duplicate customer records that need deduplication. Purchase amounts range reasonably from £15 to £2,300.

**Integrity**: Customer IDs follow a consistent format and appear to be properly generated. Date fields are well-formatted and span the expected time period without gaps.

## Key Insights
• High data completeness (92%) indicates good data collection processes
• Phone number gaps may limit communication effectiveness 
• Duplicate records suggest the need for better data entry validation
• Purchase patterns show healthy customer engagement with no suspicious values
• Geographic distribution covers all target regions appropriately

## Recommendations

**Immediate Actions**:
- Remove 12 duplicate customer records based on customer ID matching
- Investigate and clean 3 age outliers (values > 100 years)
- Standardize phone number formats for consistency

**Data Validation**:
- Implement duplicate checking at data entry point
- Add age range validation (reasonable limits: 16-95 years)
- Create email format validation rules

**Process Improvements**:
- Train staff on importance of complete contact information
- Implement required field validation for critical data points
- Set up automated data quality monitoring alerts

---
"""
    
    def create_prompt_template(self) -> str:
        """Create an enhanced prompt template for data analysis inspired by PandasAI.
        
        Returns:
            Enhanced prompt template string
        """
        example = self.create_few_shot_example()
        
        return f"""You are a professional data analyst creating a comprehensive data quality report. Your goal is to provide actionable insights that help users understand their data better.

{example}

Now analyze the following dataset using the same format and style:

ANALYSIS CONTEXT:
{{statistics}}

Create a detailed markdown report following the example format above. Be specific, use actual numbers from the data, and provide practical recommendations.

## Dataset Overview
- Provide a clear summary of the dataset size, structure, and data types
- Mention overall data volume, complexity, and what the data represents
- Give context about the business or analytical value

## Data Quality Assessment
Analyze systematically:
- **Completeness**: Which columns have missing values and how this impacts usability
- **Consistency**: Look for outliers, format inconsistencies, or unusual patterns
- **Integrity**: Check for duplicates, ID column issues, or data entry problems
- **Usability**: Assess how ready the data is for analysis or business use

## Key Insights
List 3-5 most important findings:
- Data quality strengths and opportunities
- Notable patterns or distributions
- Critical issues requiring immediate attention
- Overall assessment of data fitness for purpose

## Recommendations
Provide specific, actionable steps:
- **Immediate Actions**: Quick fixes for urgent quality issues
- **Data Validation**: Rules and checks to implement going forward
- **Process Improvements**: Better data collection, entry, or handling procedures

GUIDELINES:
- Use professional language with specific numbers and percentages
- Focus on practical, actionable insights rather than technical jargon
- Highlight both positive aspects and areas needing improvement
- Structure recommendations by priority and implementation difficulty

Write your analysis now:"""
    
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
        """Build an enhanced summary prompt for quick overview.
        
        Args:
            profile: Complete profile from profiler.profile_df()
            
        Returns:
            Enhanced summary prompt for LLM processing
        """
        dataset_stats = profile.get('dataset', {})
        columns = profile.get('columns', [])
        
        # Enhanced dataset statistics formatting
        stats_lines = [self.format_dataset_overview(dataset_stats)]
        stats_lines.append("")
        
        # Analyze column types and quality issues
        column_types = {}
        quality_issues = {}
        
        for column in columns:
            col_type = column.get('inferred_type', 'unknown')
            column_types[col_type] = column_types.get(col_type, 0) + 1
            
            issues = column.get('quality_issues', [])
            for issue in issues:
                quality_issues[issue] = quality_issues.get(issue, 0) + 1
        
        # Format column type summary
        if column_types:
            stats_lines.append("Column Types:")
            for col_type, count in sorted(column_types.items()):
                stats_lines.append(f"• {count} {col_type} columns")
            stats_lines.append("")
        
        # Format quality issues summary
        if quality_issues:
            stats_lines.append("Data Quality Issues:")
            for issue, count in sorted(quality_issues.items()):
                readable_issue = self._format_quality_issues([issue])
                stats_lines.append(f"• {count} columns with {readable_issue}")
        else:
            stats_lines.append("• No significant data quality issues detected")
        
        stats_lines.append("")
        
        # Add sample column details for context
        stats_lines.append("Sample Column Details:")
        # Show top 3 most problematic columns
        sorted_columns = sorted(columns, key=lambda x: len(x.get('quality_issues', [])), reverse=True)
        for column in sorted_columns[:3]:
            stats_lines.append(self.format_column_stats(column))
        
        statistics_text = "\n".join(stats_lines)
        
        # Enhanced template with better structure and examples
        template = """You are an expert data analyst. Analyze this dataset and create a professional data quality report.

DATASET INFORMATION:
{statistics}

Create a comprehensive markdown report with these sections:

## Dataset Overview
Provide a clear summary of the dataset structure, size, and general characteristics. Mention the data types and overall complexity.

## Data Quality Assessment
Analyze the data quality systematically:
- **Completeness**: Identify missing values and their impact
- **Consistency**: Look for outliers, unusual patterns, or inconsistencies  
- **Integrity**: Check for duplicates, format issues, or data entry problems
- **Usability**: Assess how ready the data is for analysis

## Key Insights
List the 3-5 most important findings about this dataset. Focus on:
- Notable patterns or trends
- Critical quality issues requiring attention
- Strengths and limitations of the data
- Potential analytical value

## Recommendations
Provide specific, actionable steps to improve data quality:
- **Immediate actions**: Quick fixes for urgent issues
- **Data validation**: Rules and checks to implement
- **Process improvements**: Better data collection or handling
- **Analysis readiness**: Steps to prepare for analysis

Be specific, use actual numbers from the data, and write in a professional yet accessible tone."""
        
        return template.format(statistics=statistics_text)
