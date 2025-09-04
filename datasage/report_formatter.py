"""
Report formatting module for DataSage.

This module assembles LLM-generated text chunks into well-structured
Markdown reports with proper sections and formatting.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class ReportFormatter:
    """Formats and assembles LLM outputs into structured reports."""
    
    def __init__(self, use_enhanced_fallback: bool = True):
        """Initialize the report formatter.
        
        Args:
            use_enhanced_fallback: Whether to use enhanced statistical fallback
        """
        self.use_enhanced_fallback = use_enhanced_fallback
    
    def clean_markdown_text(self, text: str) -> str:
        """Clean and normalise markdown text from LLM output.
        
        Args:
            text: Raw text from LLM
            
        Returns:
            Cleaned markdown text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = text.strip()
        
        # Fix common markdown issues
        text = re.sub(r'^#+\s*', lambda m: m.group(0), text, flags=re.MULTILINE)
        
        # Ensure proper spacing after headers
        text = re.sub(r'(^#+.*$)\n([^#\n])', r'\1\n\n\2', text, flags=re.MULTILINE)
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from markdown text.
        
        Args:
            text: Markdown text with sections
            
        Returns:
            Dictionary mapping section names to content
        """
        sections = {}
        current_section = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            # Check if this is a header line
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                level = len(header_match.group(1))
                section_name = header_match.group(2).strip()
                current_section = section_name.lower().replace(' ', '_')
                current_content = []
            else:
                # Add to current section content
                if current_section:
                    current_content.append(line)
        
        # Save final section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def merge_duplicate_sections(self, all_sections: List[Dict[str, str]]) -> Dict[str, str]:
        """Merge duplicate sections from multiple chunks.
        
        Args:
            all_sections: List of section dictionaries from different chunks
            
        Returns:
            Merged sections dictionary
        """
        merged = {}
        
        # Priority order for sections
        section_priorities = {
            'dataset_overview': 1,
            'executive_summary': 1,
            'data_quality_summary': 2,
            'data_quality_assessment': 2,
            'column_profiles': 3,
            'recommendations': 4
        }
        
        for sections in all_sections:
            for section_name, content in sections.items():
                if not content.strip():
                    continue
                
                if section_name not in merged:
                    merged[section_name] = content
                else:
                    # For overview sections, prefer the first (most complete) version
                    if section_name in ['dataset_overview', 'executive_summary']:
                        continue
                    
                    # For other sections, append unique content
                    if section_name == 'column_profiles':
                        # Combine column profiles
                        merged[section_name] += '\n\n' + content
                    elif section_name in ['data_quality_summary', 'data_quality_assessment']:
                        # Merge quality summaries, avoiding duplicates
                        existing_lines = set(merged[section_name].split('\n'))
                        new_lines = [line for line in content.split('\n') 
                                   if line.strip() and line not in existing_lines]
                        if new_lines:
                            merged[section_name] += '\n' + '\n'.join(new_lines)
        
        return merged
    
    def format_column_profiles(self, column_content: str) -> str:
        """Format and organise column profile content.
        
        Args:
            column_content: Raw column profiles content
            
        Returns:
            Formatted column profiles
        """
        if not column_content.strip():
            return ""
        
        # Split into individual column sections
        sections = re.split(r'\n#+\s+', column_content)
        
        # Sort sections - put problematic columns first
        def section_priority(section):
            # Look for issue indicators
            issue_indicators = [
                'issue', 'problem', 'concern', 'missing', 'outlier', 
                'skewed', 'empty', 'rare', 'duplicate'
            ]
            
            section_lower = section.lower()
            issue_count = sum(1 for indicator in issue_indicators 
                            if indicator in section_lower)
            
            # Return negative issue count for descending sort
            return -issue_count
        
        if len(sections) > 1:
            # First section might be header text, handle carefully
            header_section = sections[0]
            column_sections = sections[1:] if len(sections) > 1 else []
            
            # Sort column sections by priority
            sorted_sections = sorted(column_sections, key=section_priority)
            
            # Reconstruct with proper headers
            formatted_sections = []
            for i, section in enumerate(sorted_sections):
                if section.strip():
                    # Add header if not present
                    if not section.strip().startswith('#'):
                        section = f"### {section.split('.')[0].strip()}\n{section}"
                    formatted_sections.append(section.strip())
            
            return '\n\n'.join(formatted_sections)
        
        return column_content
    
    def create_report_header(self, title: str = "Data Quality Report") -> str:
        """Create the report header with title and timestamp.
        
        Args:
            title: Report title
            
        Returns:
            Formatted header
        """
        timestamp = datetime.now().strftime("%d %B %Y at %H:%M")
        
        header = f"""# {title}

*Generated on {timestamp} using DataSage*

---
"""
        return header
    
    def is_poor_quality_output(self, content: str) -> bool:
        """Enhanced check for poor quality LLM output.
        
        Args:
            content: Generated content to check
            
        Returns:
            True if content is poor quality
        """
        if not content or len(content.strip()) < 100:
            return True
        
        content_lower = content.lower().strip()
        
        # Check for common failure patterns
        failure_patterns = [
            "error generating response",
            "unable to generate",
            "failed to analyze",
            "cannot process",
            "insufficient data",
            "analysis failed"
        ]
        
        for pattern in failure_patterns:
            if pattern in content_lower:
                return True
        
        # Check for excessive repetition (same phrase repeated)
        words = content.split()
        if len(words) > 10:
            # Look for patterns that repeat more than 2 times
            for i in range(len(words) - 6):
                phrase = " ".join(words[i:i+4])
                count = content.count(phrase)
                if count > 2:
                    return True
        
        # Check if it's mostly headers without content
        lines = content.strip().split('\n')
        header_count = sum(1 for line in lines if line.strip().startswith('#'))
        content_lines = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        
        if header_count > content_lines and len(lines) > 5:
            return True
        
        # Check for minimal content (just brief phrases)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) < 3:
            return True
        
        # Check for incoherent output (too many unrelated short fragments)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        if avg_sentence_length < 3 and len(sentences) > 2:
            return True
            
        return False

    def assemble_report(
        self, 
        llm_outputs: List[str], 
        profile: Optional[Dict[str, Any]] = None,
        title: str = "Data Quality Report"
    ) -> str:
        """Assemble multiple LLM outputs into a coherent report.
        
        Args:
            llm_outputs: List of LLM-generated text chunks
            profile: Original data profile (optional, for fallback info)
            title: Report title
            
        Returns:
            Complete formatted markdown report
        """
        if not llm_outputs:
            return self._create_fallback_report(profile, title)
        
        # Check for poor quality outputs
        good_outputs = []
        for output in llm_outputs:
            if not self.is_poor_quality_output(output):
                good_outputs.append(output)
        
        # If all outputs are poor quality, use enhanced fallback
        if not good_outputs:
            logger.warning("All LLM outputs were poor quality, using enhanced statistical report")
            return self._create_fallback_report(profile, title)
        
        # Clean all good outputs
        cleaned_outputs = [self.clean_markdown_text(output) for output in good_outputs]
        
        # Extract sections from each output
        all_sections = [self.extract_sections(output) for output in cleaned_outputs]
        
        # Merge sections
        merged_sections = self.merge_duplicate_sections(all_sections)
        
        # If no structured sections were found, include the raw output as AI analysis
        if not merged_sections and cleaned_outputs:
            merged_sections['ai_analysis'] = '\n\n'.join(cleaned_outputs)
        
        # Start building the report
        report_parts = [self.create_report_header(title)]
        
        # Add sections in logical order with enhanced handling
        section_order = [
            ('dataset_overview', 'Dataset Overview'),
            ('executive_summary', 'Executive Summary'),
            ('data_quality_assessment', 'Data Quality Assessment'),
            ('data_quality_summary', 'Data Quality Summary'),
            ('key_findings', 'Key Findings'),
            ('key_insights', 'Key Insights'),
            ('ai_analysis', 'AI Analysis'),
            ('column_profiles', 'Column Profiles'),
            ('column_by_column_analysis', 'Column-by-Column Analysis'),
            ('recommendations', 'Recommendations')
        ]
        
        for section_key, section_title in section_order:
            if section_key in merged_sections:
                content = merged_sections[section_key]
                
                # Special formatting for column profiles
                if section_key == 'column_profiles':
                    content = self.format_column_profiles(content)
                
                if content.strip():
                    # Add section with proper header if not already present
                    if not content.strip().startswith('#'):
                        report_parts.append(f"## {section_title}\n\n{content}")
                    else:
                        # Content already has headers, just add it
                        report_parts.append(content)
        
        # Join all parts
        full_report = '\n\n'.join(report_parts)
        
        # Final cleanup
        full_report = self.clean_markdown_text(full_report)
        
        return full_report
    
    def _create_fallback_report(
        self, 
        profile: Optional[Dict[str, Any]], 
        title: str
    ) -> str:
        """Create a comprehensive statistical report when LLM generation fails.
        
        Args:
            profile: Data profile for fallback content
            title: Report title
            
        Returns:
            High-quality statistical markdown report
        """
        header = self.create_report_header(title)
        
        if not profile:
            return header + """
## Error

Unable to generate data quality report. No profile data available.

Please check your input data and try again.
"""
        
        dataset = profile.get('dataset', {})
        columns = profile.get('columns', [])
        
        # Create comprehensive statistical report
        n_rows = dataset.get('n_rows', 0)
        n_cols = dataset.get('n_cols', 0)
        memory_mb = dataset.get('memory_usage_mb', 0)
        duplicate_pct = dataset.get('duplicate_rows_pct', 0)
        
        # Analyze column types and quality issues
        column_types = {}
        quality_issues = {}
        completeness_issues = []
        outlier_columns = []
        
        for column in columns:
            col_type = column.get('inferred_type', 'unknown')
            column_types[col_type] = column_types.get(col_type, 0) + 1
            
            # Track completeness issues
            non_null_pct = column.get('non_null_pct', 100)
            if non_null_pct < 95:
                completeness_issues.append({
                    'name': column.get('name', 'Unknown'),
                    'non_null_pct': non_null_pct,
                    'type': col_type
                })
            
            # Track outliers
            if 'outliers_iqr_count' in column and column['outliers_iqr_count'] > 0:
                outlier_columns.append({
                    'name': column.get('name', 'Unknown'),
                    'outliers': column['outliers_iqr_count'],
                    'total': n_rows
                })
            
            # Track quality issues
            issues = column.get('quality_issues', [])
            for issue in issues:
                quality_issues[issue] = quality_issues.get(issue, 0) + 1
        
        # Calculate overall completeness
        total_completeness = sum(col.get('non_null_pct', 100) for col in columns) / len(columns) if columns else 100
        
        fallback_content = f"""## Dataset Overview

This dataset contains **{n_rows:,} rows** and **{n_cols} columns**, using approximately **{memory_mb:.1f} MB** of memory. The data includes {', '.join(f"{count} {col_type}" for col_type, count in sorted(column_types.items()))} columns.

**Key Characteristics:**
- Dataset size: {n_rows:,} × {n_cols} 
- Memory efficiency: {memory_mb/n_rows*1024:.2f} KB per row
- Duplicate records: {duplicate_pct:.1f}% of total rows
- Overall data completeness: {total_completeness:.1f}%

## Data Quality Assessment

**Completeness Analysis:**
"""
        
        if completeness_issues:
            fallback_content += f"- {len(completeness_issues)} columns have missing values requiring attention:\n"
            for issue in sorted(completeness_issues, key=lambda x: x['non_null_pct']):
                missing_pct = 100 - issue['non_null_pct']
                fallback_content += f"  - **{issue['name']}** ({issue['type']}): {missing_pct:.1f}% missing values\n"
        else:
            fallback_content += "- Excellent data completeness: all columns are fully populated\n"
        
        fallback_content += "\n**Consistency Analysis:**\n"
        
        if outlier_columns:
            fallback_content += f"- {len(outlier_columns)} numerical columns contain outliers:\n"
            for col in outlier_columns:
                outlier_pct = (col['outliers'] / col['total']) * 100
                fallback_content += f"  - **{col['name']}**: {col['outliers']} outliers ({outlier_pct:.1f}% of values)\n"
        else:
            fallback_content += "- No significant outliers detected in numerical columns\n"
        
        if duplicate_pct > 0:
            fallback_content += f"- {duplicate_pct:.1f}% duplicate rows detected - consider deduplication\n"
        else:
            fallback_content += "- No duplicate rows found\n"
        
        fallback_content += "\n**Data Integrity:**\n"
        
        if quality_issues:
            fallback_content += "- Quality issues identified:\n"
            for issue, count in sorted(quality_issues.items()):
                readable_issue = issue.replace('_', ' ').title()
                fallback_content += f"  - {count} columns with {readable_issue.lower()}\n"
        else:
            fallback_content += "- No major data integrity issues detected\n"
        
        # Key insights section
        fallback_content += f"""

## Key Insights

"""
        
        insights = []
        
        # Data volume insight
        if n_rows > 10000:
            insights.append(f"**Large dataset**: With {n_rows:,} rows, this dataset provides substantial data for robust analysis")
        elif n_rows < 100:
            insights.append(f"**Small dataset**: {n_rows} rows may limit statistical power for some analyses")
        else:
            insights.append(f"**Moderate dataset**: {n_rows:,} rows provide good balance between manageability and analytical power")
        
        # Completeness insight
        if total_completeness > 95:
            insights.append(f"**High data quality**: {total_completeness:.1f}% completeness indicates excellent data collection processes")
        elif total_completeness > 80:
            insights.append(f"**Good data quality**: {total_completeness:.1f}% completeness with some gaps to address")
        else:
            insights.append(f"**Data quality concerns**: {total_completeness:.1f}% completeness indicates significant missing data issues")
        
        # Diversity insight
        if len(column_types) > 3:
            insights.append("**Rich data structure**: Multiple data types enable diverse analytical approaches")
        
        # Outlier insight
        if outlier_columns:
            insights.append(f"**Data validation needed**: {len(outlier_columns)} columns with outliers require investigation")
        
        for insight in insights:
            fallback_content += f"- {insight}\n"
        
        # Column profiles
        fallback_content += f"""

## Column Profiles

"""
        
        # Group columns by type for better organization
        for col_type, cols in self._group_columns_by_type(columns).items():
            if not cols:
                continue
                
            fallback_content += f"**{col_type.title()} Columns ({len(cols)}):**\n\n"
            
            for column in cols[:10]:  # Limit to avoid overly long reports
                name = column.get('name', 'Unknown')
                non_null_pct = column.get('non_null_pct', 0)
                n_unique = column.get('n_unique', 0)
                
                fallback_content += f"### {name}\n\n"
                fallback_content += f"- **Data completeness**: {non_null_pct:.1f}% ({n_unique:,} unique values)\n"
                
                if col_type == "numeric":
                    if 'min' in column and 'max' in column:
                        fallback_content += f"- **Range**: {column['min']:.2f} to {column['max']:.2f}\n"
                    if 'mean' in column:
                        fallback_content += f"- **Average**: {column['mean']:.2f}\n"
                    if 'outliers_iqr_count' in column and column['outliers_iqr_count'] > 0:
                        fallback_content += f"- **Quality note**: {column['outliers_iqr_count']} outliers detected\n"
                
                elif col_type == "categorical":
                    if 'top_k_values' in column and column['top_k_values']:
                        top_val = column['top_k_values'][0]
                        fallback_content += f"- **Most common**: '{top_val['value']}' ({top_val['frequency_pct']:.1f}%)\n"
                    if 'rare_values_count' in column and column['rare_values_count'] > 0:
                        fallback_content += f"- **Distribution**: {column['rare_values_count']} rare categories\n"
                
                elif col_type == "datetime":
                    if 'min_date' in column and 'max_date' in column:
                        fallback_content += f"- **Date range**: {column['min_date'][:10]} to {column['max_date'][:10]}\n"
                    if 'coverage_days' in column:
                        fallback_content += f"- **Coverage**: {column['coverage_days']} days of data\n"
                
                elif col_type == "text":
                    if 'avg_length' in column:
                        fallback_content += f"- **Average length**: {column['avg_length']:.1f} characters\n"
                
                # Add quality issues if any
                issues = column.get('quality_issues', [])
                if issues:
                    readable_issues = [issue.replace('_', ' ') for issue in issues]
                    fallback_content += f"- **Quality concerns**: {', '.join(readable_issues)}\n"
                
                fallback_content += "\n"
            
            if len(cols) > 10:
                fallback_content += f"*... and {len(cols) - 10} more {col_type} columns*\n\n"
        
        # Recommendations
        fallback_content += f"""## Recommendations

**Immediate Actions:**
"""
        
        recommendations = []
        
        if duplicate_pct > 5:
            recommendations.append(f"Remove {duplicate_pct:.1f}% duplicate rows to improve data quality")
        
        if completeness_issues:
            missing_cols = [col['name'] for col in completeness_issues if col['non_null_pct'] < 90]
            if missing_cols:
                recommendations.append(f"Address missing values in critical columns: {', '.join(missing_cols[:3])}")
        
        if outlier_columns:
            outlier_names = [col['name'] for col in outlier_columns[:3]]
            recommendations.append(f"Investigate outliers in: {', '.join(outlier_names)}")
        
        if not recommendations:
            recommendations.append("Data quality is good - proceed with analysis")
        
        for i, rec in enumerate(recommendations, 1):
            fallback_content += f"{i}. {rec}\n"
        
        fallback_content += f"""
**Data Validation Rules:**
- Implement range checks for numerical columns with outliers
- Add format validation for categorical columns
- Set up completeness monitoring for critical fields

**Process Improvements:**
- Consider data entry validation to prevent future quality issues
- Establish regular data quality monitoring and reporting
- Document data collection procedures and quality standards

---

*This report was generated using statistical analysis. While comprehensive, it represents a systematic evaluation of data patterns and quality metrics rather than AI-generated insights.*
"""
        
        return header + fallback_content
    
    def _group_columns_by_type(self, columns: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group columns by their inferred type.
        
        Args:
            columns: List of column dictionaries
            
        Returns:
            Dictionary grouping columns by type
        """
        groups = {}
        for column in columns:
            col_type = column.get('inferred_type', 'unknown')
            if col_type not in groups:
                groups[col_type] = []
            groups[col_type].append(column)
        
        return groups
