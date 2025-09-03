"""
Tests for the prompt builder module.
"""

import pytest
from unittest.mock import Mock

from datasage.prompt_builder import PromptBuilder


class TestPromptBuilder:
    """Test prompt building functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.builder = PromptBuilder(max_tokens_per_prompt=800, columns_per_chunk=3)
        
        # Mock profile data
        self.mock_profile = {
            "dataset": {
                "n_rows": 1000,
                "n_cols": 5,
                "memory_usage_mb": 2.5,
                "duplicate_rows_pct": 0.0
            },
            "columns": [
                {
                    "name": "id",
                    "inferred_type": "numeric",
                    "non_null_pct": 100.0,
                    "n_unique": 1000,
                    "min": 1.0,
                    "max": 1000.0,
                    "mean": 500.5,
                    "quality_issues": ["likely_id"]
                },
                {
                    "name": "name",
                    "inferred_type": "text",
                    "non_null_pct": 95.0,
                    "n_unique": 980,
                    "avg_length": 12.5,
                    "pct_empty_strings": 0.0,
                    "quality_issues": []
                },
                {
                    "name": "category",
                    "inferred_type": "categorical",
                    "non_null_pct": 100.0,
                    "n_unique": 5,
                    "top_k_values": [
                        {"value": "A", "count": 400, "frequency_pct": 40.0},
                        {"value": "B", "count": 300, "frequency_pct": 30.0}
                    ],
                    "rare_values_count": 1,
                    "quality_issues": []
                },
                {
                    "name": "salary",
                    "inferred_type": "numeric",
                    "non_null_pct": 98.0,
                    "n_unique": 850,
                    "min": 30000.0,
                    "max": 120000.0,
                    "mean": 65000.0,
                    "outliers_iqr_count": 15,
                    "quality_issues": ["extreme_outliers"]
                },
                {
                    "name": "join_date",
                    "inferred_type": "datetime",
                    "non_null_pct": 100.0,
                    "n_unique": 800,
                    "min_date": "2020-01-01T00:00:00",
                    "max_date": "2023-12-31T00:00:00",
                    "coverage_days": 1460,
                    "quality_issues": []
                }
            ]
        }
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "This is a sample text with approximately twenty words for testing token estimation functionality."
        estimated = self.builder.estimate_tokens(text)
        
        # Should be roughly text length / 4
        expected = len(text) // 4
        assert abs(estimated - expected) <= 2  # Allow small variance
    
    def test_format_dataset_overview(self):
        """Test dataset overview formatting."""
        overview = self.builder.format_dataset_overview(self.mock_profile["dataset"])
        
        assert "1,000 rows" in overview
        assert "5 columns" in overview
        assert "2.5 MB" in overview
        assert "0.0%" in overview
    
    def test_format_column_stats_numeric(self):
        """Test numeric column formatting."""
        numeric_col = self.mock_profile["columns"][3]  # salary column
        formatted = self.builder.format_column_stats(numeric_col)
        
        assert "salary (numeric)" in formatted
        assert "98.0% non-null" in formatted
        assert "Range: 30000.00 to 120000.00" in formatted
        assert "Mean: 65000.00" in formatted
        assert "Outliers: 15 values" in formatted
        assert "extreme outliers detected" in formatted
    
    def test_format_column_stats_categorical(self):
        """Test categorical column formatting."""
        cat_col = self.mock_profile["columns"][2]  # category column
        formatted = self.builder.format_column_stats(cat_col)
        
        assert "category (categorical)" in formatted
        assert "100.0% non-null" in formatted
        assert "Most common: 'A' (40.0%)" in formatted
        assert "Rare values: 1 categories" in formatted
    
    def test_format_column_stats_datetime(self):
        """Test datetime column formatting."""
        datetime_col = self.mock_profile["columns"][4]  # join_date column
        formatted = self.builder.format_column_stats(datetime_col)
        
        assert "join_date (datetime)" in formatted
        assert "Date range: 2020-01-01 to 2023-12-31" in formatted
        assert "Coverage: 1460 days" in formatted
    
    def test_format_column_stats_text(self):
        """Test text column formatting."""
        text_col = self.mock_profile["columns"][1]  # name column
        formatted = self.builder.format_column_stats(text_col)
        
        assert "name (text)" in formatted
        assert "95.0% non-null" in formatted
        assert "Average length: 12.5 characters" in formatted
    
    def test_format_quality_issues(self):
        """Test quality issues formatting."""
        issues = ["high_missing", "extreme_outliers", "likely_id"]
        formatted = self.builder._format_quality_issues(issues)
        
        assert "high missing values" in formatted
        assert "extreme outliers detected" in formatted
        assert "appears to be an ID column" in formatted
    
    def test_chunk_columns(self):
        """Test column chunking."""
        columns = self.mock_profile["columns"]
        chunks = self.builder.chunk_columns(columns)
        
        # With 5 columns and chunk size 3, should get 2 chunks
        assert len(chunks) == 2
        assert len(chunks[0]) == 3
        assert len(chunks[1]) == 2
    
    def test_build_prompts_multiple_chunks(self):
        """Test building prompts with multiple chunks."""
        prompts = self.builder.build_prompts(self.mock_profile)
        
        # Should create multiple prompts due to chunking
        assert len(prompts) >= 1
        
        for prompt in prompts:
            assert "Dataset Overview:" in prompt
            assert "Column Details:" in prompt
            assert "## Dataset Overview" in prompt
            assert "## Data Quality Summary" in prompt
            assert "## Column Profiles" in prompt
    
    def test_build_prompts_prioritizes_issues(self):
        """Test that columns with issues are prioritized."""
        prompts = self.builder.build_prompts(self.mock_profile)
        first_prompt = prompts[0]
        
        # Columns with issues should appear first
        # salary has extreme_outliers, id has likely_id
        assert "salary" in first_prompt or "id" in first_prompt
    
    def test_build_summary_prompt(self):
        """Test building summary-only prompt."""
        summary_prompt = self.builder.build_summary_prompt(self.mock_profile)
        
        assert "Executive Summary" in summary_prompt
        assert "Data Quality Assessment" in summary_prompt
        assert "Recommendations" in summary_prompt
        assert "1,000 rows and 5 columns" in summary_prompt
        assert "extreme outliers detected" in summary_prompt
    
    def test_prompt_template_structure(self):
        """Test that prompt template has required structure."""
        template = self.builder.create_prompt_template()
        
        assert "## Dataset Overview" in template
        assert "## Data Quality Summary" in template
        assert "## Column Profiles" in template
        assert "{statistics}" in template
        assert "UK English" in template
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        empty_profile = {
            "dataset": {"n_rows": 0, "n_cols": 0, "memory_usage_mb": 0.0, "duplicate_rows_pct": 0.0},
            "columns": []
        }
        
        prompts = self.builder.build_prompts(empty_profile)
        
        assert len(prompts) == 1
        assert "0 rows and 0 columns" in prompts[0]
    
    def test_token_limit_warning(self, caplog):
        """Test that token limit warnings are logged."""
        # Create a profile that would exceed token limits
        large_profile = self.mock_profile.copy()
        large_profile["columns"] = self.mock_profile["columns"] * 10  # Duplicate columns
        
        builder = PromptBuilder(max_tokens_per_prompt=200, columns_per_chunk=20)  # Very small limit
        
        with caplog.at_level("WARNING"):
            prompts = builder.build_prompts(large_profile)
        
        # Should log warnings about exceeding token limits
        assert any("exceeds token limit" in record.message for record in caplog.records)


class TestPromptBuilderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_malformed_profile_data(self):
        """Test handling of malformed profile data."""
        builder = PromptBuilder()
        
        # Missing dataset section
        malformed_profile = {"columns": []}
        prompts = builder.build_prompts(malformed_profile)
        assert len(prompts) >= 1
        
        # Missing columns section
        malformed_profile = {"dataset": {"n_rows": 10, "n_cols": 0, "memory_usage_mb": 0.1, "duplicate_rows_pct": 0.0}}
        prompts = builder.build_prompts(malformed_profile)
        assert len(prompts) >= 1
    
    def test_single_column_chunking(self):
        """Test chunking with very small chunk size."""
        builder = PromptBuilder(columns_per_chunk=1)
        
        profile = {
            "dataset": {"n_rows": 10, "n_cols": 2, "memory_usage_mb": 0.1, "duplicate_rows_pct": 0.0},
            "columns": [
                {"name": "col1", "inferred_type": "numeric", "non_null_pct": 100.0, "n_unique": 10, "quality_issues": []},
                {"name": "col2", "inferred_type": "text", "non_null_pct": 100.0, "n_unique": 10, "quality_issues": []}
            ]
        }
        
        prompts = builder.build_prompts(profile)
        
        # Should create 2 prompts, one for each column
        assert len(prompts) == 2


if __name__ == "__main__":
    pytest.main([__file__])
