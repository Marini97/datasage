"""
Tests for the verifier module.
"""

import pytest
from unittest.mock import Mock

from datasage.verifier import OutputVerifier


class TestOutputVerifier:
    """Test output verification functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.verifier = OutputVerifier()
        
        # Mock profile data for verification
        self.mock_profile = {
            "dataset": {
                "n_rows": 1000,
                "n_cols": 5,
                "memory_usage_mb": 2.5,
                "duplicate_rows_pct": 5.0
            },
            "columns": [
                {
                    "name": "employee_id",
                    "inferred_type": "numeric",
                    "non_null_pct": 100.0,
                    "n_unique": 1000,
                    "quality_issues": []
                },
                {
                    "name": "email",
                    "inferred_type": "text",
                    "non_null_pct": 80.0,  # 20% missing
                    "n_unique": 800,
                    "quality_issues": ["high_missing"]
                },
                {
                    "name": "salary",
                    "inferred_type": "numeric",
                    "non_null_pct": 95.0,
                    "zeros_pct": 2.0,
                    "quality_issues": []
                }
            ]
        }
    
    def test_extract_numeric_claims_percentages(self):
        """Test extraction of percentage claims."""
        text = """
        The dataset has 20% missing values in the email column.
        About 5% duplicate rows were found.
        Salary column shows 2% zero values.
        """
        
        claims = self.verifier.extract_numeric_claims(text)
        
        # Should find percentage claims
        percentage_claims = [c for c in claims if c['type'] == 'percentage']
        assert len(percentage_claims) >= 2
        
        # Check specific claims
        missing_claim = next((c for c in claims if 'missing' in c['context'].lower()), None)
        assert missing_claim is not None
        assert missing_claim['value'] == 20.0
        
        duplicate_claim = next((c for c in claims if 'duplicate' in c['context'].lower()), None)
        assert duplicate_claim is not None
        assert duplicate_claim['value'] == 5.0
    
    def test_extract_numeric_claims_counts(self):
        """Test extraction of count claims."""
        text = """
        This dataset contains 1,000 rows and 5 columns.
        The employee_id column has 1,000 unique values.
        We detected 25 outliers in the salary data.
        """
        
        claims = self.verifier.extract_numeric_claims(text)
        
        # Check row count
        row_claims = [c for c in claims if c['type'] == 'row_count']
        assert len(row_claims) >= 1
        assert row_claims[0]['value'] == 1000
        
        # Check column count
        col_claims = [c for c in claims if c['type'] == 'column_count']
        assert len(col_claims) >= 1
        assert col_claims[0]['value'] == 5
        
        # Check outlier count
        outlier_claims = [c for c in claims if c['type'] == 'outlier_count']
        assert len(outlier_claims) >= 1
        assert outlier_claims[0]['value'] == 25
    
    def test_extract_numeric_claims_ranges(self):
        """Test extraction of range claims."""
        text = """
        Salary values range from 30000 to 120000.
        The minimum age is 25 and maximum is 65.
        Average salary is 65000.
        """
        
        claims = self.verifier.extract_numeric_claims(text)
        
        # Check range claim
        range_claims = [c for c in claims if c['type'] == 'range']
        assert len(range_claims) >= 1
        assert range_claims[0]['min_value'] == 30000.0
        assert range_claims[0]['max_value'] == 120000.0
        
        # Check min/max claims
        min_claims = [c for c in claims if c['type'] == 'min_value']
        max_claims = [c for c in claims if c['type'] == 'max_value']
        mean_claims = [c for c in claims if c['type'] == 'mean_value']
        
        assert len(min_claims) >= 1
        assert len(max_claims) >= 1
        assert len(mean_claims) >= 1
    
    def test_extract_numeric_claims_memory(self):
        """Test extraction of memory usage claims."""
        text = "The dataset uses approximately 2.5 MB of memory."
        
        claims = self.verifier.extract_numeric_claims(text)
        
        memory_claims = [c for c in claims if c['type'] == 'memory_mb']
        assert len(memory_claims) >= 1
        assert memory_claims[0]['value'] == 2.5
    
    def test_get_context(self):
        """Test context extraction around claims."""
        text = "This is a long sentence with a specific 25% missing value claim in the middle of it."
        position = (43, 46)  # Position of "25%"
        
        context = self.verifier._get_context(text, position, 20)
        
        assert "**25%**" in context  # Claim should be highlighted
        assert len(context) <= len(text)  # Context should not exceed original
    
    def test_verify_claims_against_profile_row_count(self):
        """Test verification of row count claims."""
        claims = [
            {
                'type': 'row_count',
                'value': 1000,
                'text': '1,000 rows',
                'context': 'dataset has 1,000 rows'
            }
        ]
        
        verified = self.verifier.verify_claims_against_profile(claims, self.mock_profile)
        
        assert len(verified) == 1
        assert verified[0]['verified'] is True
        assert verified[0]['actual_value'] == 1000
    
    def test_verify_claims_against_profile_incorrect_count(self):
        """Test verification of incorrect count claims."""
        claims = [
            {
                'type': 'row_count',
                'value': 999,  # Incorrect value
                'text': '999 rows',
                'context': 'dataset has 999 rows'
            }
        ]
        
        verified = self.verifier.verify_claims_against_profile(claims, self.mock_profile)
        
        assert len(verified) == 1
        assert verified[0]['verified'] is True  # Within 1% tolerance
        assert verified[0]['actual_value'] == 1000
    
    def test_verify_claims_against_profile_column_percentage(self):
        """Test verification of column-specific percentage claims."""
        claims = [
            {
                'type': 'percentage',
                'value': 20.0,
                'text': '20% missing',
                'context': 'email column has 20% missing values'
            }
        ]
        
        verified = self.verifier.verify_claims_against_profile(claims, self.mock_profile)
        
        assert len(verified) == 1
        assert verified[0]['verified'] is True
        assert verified[0]['actual_value'] == 20.0  # 100 - 80 = 20% missing
        assert verified[0]['column_name'] == 'email'
    
    def test_verify_claims_against_profile_duplicate_percentage(self):
        """Test verification of duplicate percentage claims."""
        claims = [
            {
                'type': 'percentage',
                'value': 5.0,
                'text': '5% duplicate',
                'context': 'found 5% duplicate rows'
            }
        ]
        
        verified = self.verifier.verify_claims_against_profile(claims, self.mock_profile)
        
        assert len(verified) == 1
        assert verified[0]['verified'] is True
        assert verified[0]['actual_value'] == 5.0
    
    def test_verify_claims_with_tolerance(self):
        """Test verification with tolerance for small discrepancies."""
        claims = [
            {
                'type': 'percentage',
                'value': 21.0,  # Slightly off from actual 20%
                'text': '21% missing',
                'context': 'email column has 21% missing values'
            }
        ]
        
        verified = self.verifier.verify_claims_against_profile(claims, self.mock_profile)
        
        assert len(verified) == 1
        assert verified[0]['verified'] is True  # Within 2% tolerance
    
    def test_verify_claims_outside_tolerance(self):
        """Test verification of claims outside tolerance."""
        claims = [
            {
                'type': 'percentage',
                'value': 30.0,  # Too far from actual 20%
                'text': '30% missing',
                'context': 'email column has 30% missing values'
            }
        ]
        
        verified = self.verifier.verify_claims_against_profile(claims, self.mock_profile)
        
        assert len(verified) == 1
        assert verified[0]['verified'] is False
        assert verified[0]['actual_value'] == 20.0
        assert 'discrepancy' in verified[0]
    
    def test_create_verification_annotations(self):
        """Test creation of verification annotations."""
        failed_claims = [
            {
                'verified': False,
                'context': 'email column has 30% missing values',
                'actual_value': 20.0,
                'discrepancy': '10.0 percentage points'
            },
            {
                'verified': False,
                'context': 'dataset has 900 rows',
                'actual_value': 1000,
                'discrepancy': '10.0% difference'
            }
        ]
        
        annotations = self.verifier.create_verification_annotations(failed_claims)
        
        assert "## Verification Notes" in annotations
        assert "⚠️" in annotations
        assert "actual value is 20.0" in annotations
        assert "actual value is 1000" in annotations
    
    def test_create_verification_annotations_no_failures(self):
        """Test annotation creation when no claims failed."""
        verified_claims = [
            {
                'verified': True,
                'actual_value': 20.0
            }
        ]
        
        annotations = self.verifier.create_verification_annotations(verified_claims)
        
        assert annotations == ""  # No annotations for successful verifications
    
    def test_verify_and_annotate_report(self):
        """Test full report verification and annotation."""
        report_text = """
        # Data Quality Report
        
        This dataset contains 1,000 rows and 5 columns.
        The email column has 25% missing values.
        Memory usage is approximately 2.5 MB.
        """
        
        annotated_report, verification_results = self.verifier.verify_and_annotate_report(
            report_text, 
            self.mock_profile, 
            add_annotations=True
        )
        
        # Should have extracted and verified claims
        assert len(verification_results) > 0
        
        # Should have correct claims
        row_claim = next((c for c in verification_results if c['type'] == 'row_count'), None)
        assert row_claim is not None
        assert row_claim['verified'] is True
        
        # Should have annotated failed claims
        email_claim = next((c for c in verification_results if 'email' in c.get('context', '').lower()), None)
        if email_claim and not email_claim['verified']:
            assert "## Verification Notes" in annotated_report
    
    def test_verify_and_annotate_report_no_annotations(self):
        """Test report verification without annotations."""
        report_text = "This dataset contains 1,000 rows."
        
        annotated_report, verification_results = self.verifier.verify_and_annotate_report(
            report_text, 
            self.mock_profile, 
            add_annotations=False
        )
        
        # Should not add annotations even if there are issues
        assert "## Verification Notes" not in annotated_report
        assert annotated_report == report_text
        assert len(verification_results) > 0
    
    def test_get_verification_summary(self):
        """Test verification summary generation."""
        # Simulate some verifications
        self.verifier.verification_log = [
            {'verified': True},
            {'verified': False},
            {'verified': True},
            {'verified': False}
        ]
        
        summary = self.verifier.get_verification_summary()
        
        assert summary['total_claims_checked'] == 4
        assert summary['failed_verifications'] == 2
        assert summary['success_rate'] == 50.0
    
    def test_get_verification_summary_empty(self):
        """Test verification summary with no verifications."""
        summary = self.verifier.get_verification_summary()
        
        assert summary['total_claims_checked'] == 0
        assert summary['failed_verifications'] == 0
        assert summary['success_rate'] == 100.0


class TestVerifierEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extract_claims_no_matches(self):
        """Test claim extraction with no numeric content."""
        verifier = OutputVerifier()
        text = "This text contains no numeric claims whatsoever."
        
        claims = verifier.extract_numeric_claims(text)
        assert len(claims) == 0
    
    def test_verify_claims_malformed_profile(self):
        """Test verification with malformed profile data."""
        verifier = OutputVerifier()
        claims = [{'type': 'row_count', 'value': 100, 'text': '100 rows', 'context': 'has 100 rows'}]
        malformed_profile = {}  # Missing dataset and columns
        
        verified = verifier.verify_claims_against_profile(claims, malformed_profile)
        
        # Should handle gracefully
        assert len(verified) == 1
        assert not verified[0]['verified']
    
    def test_verify_claims_with_exceptions(self):
        """Test verification with data that causes exceptions."""
        verifier = OutputVerifier()
        claims = [
            {
                'type': 'row_count',
                'value': 'invalid',  # Invalid numeric value
                'text': 'invalid rows',
                'context': 'has invalid rows'
            }
        ]
        
        profile = {
            "dataset": {"n_rows": 100, "n_cols": 5, "memory_usage_mb": 1.0, "duplicate_rows_pct": 0.0},
            "columns": []
        }
        
        verified = verifier.verify_claims_against_profile(claims, profile)
        
        # Should handle exceptions gracefully
        assert len(verified) == 1
        assert 'error' in verified[0] or not verified[0]['verified']


if __name__ == "__main__":
    pytest.main([__file__])
