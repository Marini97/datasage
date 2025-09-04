"""
Verification module for DataSage.

This module extracts numeric claims from LLM outputs and cross-checks
them against the original statistics, flagging discrepancies.
"""

import re
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class OutputVerifier:
    """Verifies numeric claims in LLM-generated reports."""
    
    def __init__(self):
        """Initialize the output verifier."""
        self.verification_log = []
    
    def extract_numeric_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract numeric claims from text.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of extracted numeric claims with context
        """
        claims = []
        
        # Patterns for different types of numeric claims
        patterns = [
            # Percentages
            (r'(\d+(?:\.\d+)?)\s*%\s*(missing|null|empty|complete|non-null)', 'percentage'),
            (r'(\d+(?:\.\d+)?)\s*%\s*(duplicate|duplicated)', 'percentage'),
            (r'(\d+(?:\.\d+)?)\s*%\s*(zero|zeros)', 'percentage'),
            
            # Ranges and values
            (r'(?:range|ranges?)\s*:?\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)', 'range'),
            (r'(?:min|minimum)\s*:?\s*(\d+(?:\.\d+)?)', 'min_value'),
            (r'(?:max|maximum)\s*:?\s*(\d+(?:\.\d+)?)', 'max_value'),
            (r'(?:mean|average)\s*:?\s*(\d+(?:\.\d+)?)', 'mean_value'),
            
            # Counts
            (r'(\d+(?:,\d+)*)\s*(?:rows?|records?)', 'row_count'),
            (r'(\d+(?:,\d+)*)\s*(?:columns?)', 'column_count'),
            (r'(\d+(?:,\d+)*)\s*(?:unique|distinct)\s*(?:values?)', 'unique_count'),
            (r'(\d+(?:,\d+)*)\s*(?:outliers?)', 'outlier_count'),
            
            # Memory usage
            (r'(\d+(?:\.\d+)?)\s*(?:MB|mb|megabytes?)', 'memory_mb'),
            
            # Days/dates
            (r'(\d+)\s*days?', 'days_count'),
        ]
        
        for pattern, claim_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim = {
                    'type': claim_type,
                    'text': match.group(0),
                    'position': match.span(),
                    'context': self._get_context(text, match.span(), 50)
                }
                
                if claim_type == 'range':
                    claim['min_value'] = float(match.group(1))
                    claim['max_value'] = float(match.group(2))
                elif claim_type in ['percentage', 'min_value', 'max_value', 'mean_value', 'memory_mb', 'days_count']:
                    claim['value'] = float(match.group(1))
                elif claim_type in ['row_count', 'column_count', 'unique_count', 'outlier_count']:
                    # Remove commas from numbers
                    claim['value'] = int(match.group(1).replace(',', ''))
                
                claims.append(claim)
        
        return claims
    
    def _get_context(self, text: str, position: Tuple[int, int], context_chars: int = 50) -> str:
        """Get surrounding context for a matched claim.
        
        Args:
            text: Full text
            position: (start, end) position of match
            context_chars: Number of characters of context on each side
            
        Returns:
            Context string
        """
        start, end = position
        context_start = max(0, start - context_chars)
        context_end = min(len(text), end + context_chars)
        
        context = text[context_start:context_end]
        # Mark the actual claim within the context
        claim_start = start - context_start
        claim_end = end - context_start
        
        return (
            context[:claim_start] + 
            "**" + context[claim_start:claim_end] + "**" + 
            context[claim_end:]
        )
    
    def verify_claims_against_profile(
        self, 
        claims: List[Dict[str, Any]], 
        profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Verify claims against the original data profile.
        
        Args:
            claims: List of extracted claims
            profile: Original data profile from profiler
            
        Returns:
            List of claims with verification results
        """
        verified_claims = []
        dataset_stats = profile.get('dataset', {})
        columns = profile.get('columns', [])
        
        for claim in claims:
            verified_claim = claim.copy()
            verified_claim['verified'] = False
            verified_claim['actual_value'] = None
            verified_claim['discrepancy'] = None
            
            claim_type = claim['type']
            
            try:
                if claim_type == 'row_count':
                    actual = dataset_stats.get('n_rows', 0)
                    claimed = claim['value']
                    verified_claim['actual_value'] = actual
                    verified_claim['verified'] = abs(actual - claimed) <= max(1, actual * 0.01)  # 1% tolerance
                    
                elif claim_type == 'column_count':
                    actual = dataset_stats.get('n_cols', 0)
                    claimed = claim['value']
                    verified_claim['actual_value'] = actual
                    verified_claim['verified'] = actual == claimed
                    
                elif claim_type == 'memory_mb':
                    actual = dataset_stats.get('memory_usage_mb', 0)
                    claimed = claim['value']
                    verified_claim['actual_value'] = actual
                    verified_claim['verified'] = abs(actual - claimed) <= max(0.1, actual * 0.1)  # 10% tolerance
                    
                elif claim_type == 'percentage':
                    # Need to match with specific columns based on context
                    context_lower = claim['context'].lower()
                    
                    if 'duplicate' in context_lower:
                        actual = dataset_stats.get('duplicate_rows_pct', 0)
                        claimed = claim['value']
                        verified_claim['actual_value'] = actual
                        verified_claim['verified'] = abs(actual - claimed) <= 1.0  # 1% tolerance
                    
                    else:
                        # Try to match with column-level percentages
                        verified_claim = self._verify_column_percentage(
                            verified_claim, columns, context_lower
                        )
                
                # Calculate discrepancy for failed verifications
                if not verified_claim['verified'] and verified_claim['actual_value'] is not None:
                    if claim_type in ['percentage']:
                        discrepancy = abs(verified_claim['actual_value'] - claim['value'])
                        verified_claim['discrepancy'] = f"{discrepancy:.1f} percentage points"
                    else:
                        actual_val = verified_claim['actual_value']
                        claimed_val = claim['value']
                        if actual_val != 0:
                            pct_diff = abs(actual_val - claimed_val) / actual_val * 100
                            verified_claim['discrepancy'] = f"{pct_diff:.1f}% difference"
                        else:
                            verified_claim['discrepancy'] = f"Expected 0, got {claimed_val}"
            
            except Exception as e:
                logger.warning(f"Error verifying claim {claim}: {e}")
                verified_claim['error'] = str(e)
            
            verified_claims.append(verified_claim)
        
        return verified_claims
    
    def _verify_column_percentage(
        self, 
        claim: Dict[str, Any], 
        columns: List[Dict[str, Any]], 
        context: str
    ) -> Dict[str, Any]:
        """Verify percentage claims related to specific columns.
        
        Args:
            claim: Claim to verify
            columns: List of column profiles
            context: Context text around the claim
            
        Returns:
            Updated claim with verification results
        """
        claimed_value = claim['value']
        
        # Look for column names in context
        for column in columns:
            column_name_lower = column['name'].lower()
            if column_name_lower in context:
                if 'missing' in context or 'null' in context:
                    # Check non-null percentage (inverse of missing)
                    actual_non_null = column.get('non_null_pct', 0)
                    if 'non-null' in context or 'complete' in context:
                        actual = actual_non_null
                    else:
                        actual = 100 - actual_non_null  # Missing percentage
                    
                    claim['actual_value'] = actual
                    claim['verified'] = abs(actual - claimed_value) <= 2.0  # 2% tolerance
                    claim['column_name'] = column['name']
                    
                elif 'zero' in context and column.get('inferred_type') == 'numeric':
                    actual = column.get('zeros_pct', 0)
                    claim['actual_value'] = actual
                    claim['verified'] = abs(actual - claimed_value) <= 2.0
                    claim['column_name'] = column['name']
                
                break
        
        return claim
    
    def create_verification_annotations(self, verified_claims: List[Dict[str, Any]]) -> str:
        """Create verification annotations for failed claims.
        
        Args:
            verified_claims: List of verified claims
            
        Returns:
            Markdown text with verification annotations
        """
        failed_claims = [claim for claim in verified_claims if not claim.get('verified', True)]
        
        if not failed_claims:
            return ""
        
        annotations = ["## Verification Notes\n"]
        annotations.append(
            "*The following numeric claims in this report have been automatically cross-checked against the source data:*\n"
        )
        
        for i, claim in enumerate(failed_claims, 1):
            context = claim.get('context', claim.get('text', ''))
            actual_value = claim.get('actual_value')
            discrepancy = claim.get('discrepancy', '')
            
            if actual_value is not None:
                annotation = (
                    f"{i}. **{context}**  \n"
                    f"   ⚠️ *Auto-corrected: actual value is {actual_value:.1f}"
                )
                if discrepancy:
                    annotation += f" ({discrepancy})"
                annotation += "*\n"
            else:
                annotation = (
                    f"{i}. **{context}**  \n"
                    f"   ⚠️ *Could not verify this claim against source data*\n"
                )
            
            annotations.append(annotation)
        
        self.verification_log.extend(failed_claims)
        
        return "\n".join(annotations)
    
    def verify_and_annotate_report(
        self, 
        report_text: str, 
        profile: Dict[str, Any], 
        add_annotations: bool = True
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Verify all claims in a report and optionally add annotations.
        
        Args:
            report_text: Full report text
            profile: Original data profile
            add_annotations: Whether to add verification annotations
            
        Returns:
            Tuple of (annotated_report, verification_results)
        """
        # Extract claims
        claims = self.extract_numeric_claims(report_text)
        
        if not claims:
            return report_text, []
        
        # Verify claims
        verified_claims = self.verify_claims_against_profile(claims, profile)
        
        # Add annotations if requested
        annotated_report = report_text
        if add_annotations:
            annotations = self.create_verification_annotations(verified_claims)
            if annotations:
                annotated_report += "\n\n---\n\n" + annotations
        
        return annotated_report, verified_claims
    
    def get_verification_summary(self) -> Dict[str, Any]:
        """Get a summary of all verifications performed.
        
        Returns:
            Summary statistics about verifications
        """
        if not self.verification_log:
            return {
                'total_claims_checked': 0,
                'failed_verifications': 0,
                'success_rate': 100.0
            }
        
        failed_count = len([claim for claim in self.verification_log 
                          if not claim.get('verified', True)])
        
        return {
            'total_claims_checked': len(self.verification_log),
            'failed_verifications': failed_count,
            'success_rate': (1 - failed_count / len(self.verification_log)) * 100
        }
