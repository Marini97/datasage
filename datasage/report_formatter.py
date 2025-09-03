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
    
    def __init__(self):
        """Initialize the report formatter."""
        pass
    
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
        """Check if LLM output is poor quality (repetitive, too short, etc).
        
        Args:
            content: Generated content to check
            
        Returns:
            True if content is poor quality
        """
        if not content or len(content.strip()) < 50:
            return True
            
        # Check for excessive repetition (same phrase repeated)
        words = content.split()
        if len(words) > 10:
            # Look for patterns that repeat more than 3 times
            for i in range(len(words) - 6):
                phrase = " ".join(words[i:i+3])
                count = content.count(phrase)
                if count > 3:
                    return True
        
        # Check if it's mostly headers without content
        lines = content.strip().split('\n')
        header_count = sum(1 for line in lines if line.strip().startswith('#'))
        if header_count > len(lines) / 2 and len(lines) > 5:
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
        
        # If all outputs are poor quality, use fallback
        if not good_outputs:
            logger.warning("All LLM outputs were poor quality, using fallback report")
            return self._create_fallback_report(profile, title)
        
        # Clean all good outputs
        cleaned_outputs = [self.clean_markdown_text(output) for output in good_outputs]
        
        # Extract sections from each output
        all_sections = [self.extract_sections(output) for output in cleaned_outputs]
        
        # Merge sections
        merged_sections = self.merge_duplicate_sections(all_sections)
        
        # Start building the report
        report_parts = [self.create_report_header(title)]
        
        # Add sections in logical order
        section_order = [
            ('dataset_overview', 'Dataset Overview'),
            ('executive_summary', 'Executive Summary'),
            ('data_quality_summary', 'Data Quality Summary'),
            ('data_quality_assessment', 'Data Quality Assessment'),
            ('column_profiles', 'Column Profiles'),
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
        """Create a basic report when LLM generation fails.
        
        Args:
            profile: Data profile for fallback content
            title: Report title
            
        Returns:
            Basic markdown report
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
        
        # Create basic report from profile data
        fallback_content = f"""## Dataset Overview

This dataset contains {dataset.get('n_rows', 0):,} rows and {dataset.get('n_cols', 0)} columns.

## Data Quality Summary

- **Memory usage**: {dataset.get('memory_usage_mb', 0):.1f} MB
- **Duplicate rows**: {dataset.get('duplicate_rows_pct', 0):.1f}%

## Column Profiles

"""
        
        for column in columns[:10]:  # Limit to first 10 columns
            name = column.get('name', 'Unknown')
            col_type = column.get('inferred_type', 'unknown')
            non_null_pct = column.get('non_null_pct', 0)
            
            fallback_content += f"### {name}\n\n"
            fallback_content += f"- **Type**: {col_type}\n"
            fallback_content += f"- **Data completeness**: {non_null_pct:.1f}%\n\n"
        
        if len(columns) > 10:
            fallback_content += f"*... and {len(columns) - 10} more columns*\n\n"
        
        fallback_content += """## Note

This is a basic report generated due to an issue with the AI analysis. 
For detailed insights, please check the system logs and try again.
"""
        
        return header + fallback_content
