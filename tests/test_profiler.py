"""
Tests for the profiler module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from datasage.profiler import (
    profile_df,
    profile_column,
    infer_column_type,
    compute_numeric_stats,
    compute_categorical_stats,
    compute_datetime_stats,
    compute_text_stats,
    compute_data_quality_flags
)


class TestInferColumnType:
    """Test column type inference."""
    
    def test_numeric_integer(self):
        series = pd.Series([1, 2, 3, 4, 5])
        assert infer_column_type(series) == "numeric"
    
    def test_numeric_float(self):
        series = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        assert infer_column_type(series) == "numeric"
    
    def test_numeric_string_numbers(self):
        series = pd.Series(["1", "2", "3", "4", "5"])
        assert infer_column_type(series) == "numeric"
    
    def test_categorical_low_cardinality(self):
        series = pd.Series(["A", "B", "A", "C", "B", "A"] * 10)
        assert infer_column_type(series) == "categorical"
    
    def test_categorical_small_unique_count(self):
        series = pd.Series(["Red", "Blue", "Green"] * 20)
        assert infer_column_type(series) == "categorical"
    
    def test_text_high_cardinality(self):
        series = pd.Series([f"Text entry {i}" for i in range(100)])
        assert infer_column_type(series) == "text"
    
    def test_datetime_proper(self):
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        series = pd.Series(dates)
        assert infer_column_type(series) == "datetime"
    
    def test_datetime_string_format(self):
        series = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"])
        assert infer_column_type(series) == "datetime"


class TestComputeNumericStats:
    """Test numeric statistics computation."""
    
    def test_basic_numeric_stats(self):
        series = pd.Series([1, 2, 3, 4, 5])
        stats = compute_numeric_stats(series)
        
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert stats["mean"] == 3.0
        assert abs(stats["std"] - 1.58) < 0.1  # Approximate
        assert stats["q1"] == 2.0
        assert stats["q3"] == 4.0
    
    def test_outlier_detection(self):
        # Data with clear outliers
        series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 is an outlier
        stats = compute_numeric_stats(series)
        
        assert stats["outliers_count"] > 0
    
    def test_negative_count(self):
        series = pd.Series([-2, -1, 0, 1, 2])
        stats = compute_numeric_stats(series)
        
        assert stats["negative_count"] == 2
    
    def test_zeros_percentage(self):
        series = pd.Series([0, 0, 1, 2, 3])
        stats = compute_numeric_stats(series)
        
        assert stats["zeros_percentage"] == 40.0  # 2 out of 5
    
    def test_empty_series(self):
        series = pd.Series([], dtype=float)
        stats = compute_numeric_stats(series)
        
        # Should return default values for empty series, not empty dict
        assert stats["mean"] == 0.0
        assert stats["outliers_count"] == 0


class TestComputeCategoricalStats:
    """Test categorical statistics computation."""
    
    def test_top_values(self):
        series = pd.Series(["A", "B", "A", "C", "A", "B"])
        stats = compute_categorical_stats(series)
        
        top_values = stats["top_k_values"]
        assert len(top_values) >= 1
        assert top_values[0]["value"] == "A"  # Most frequent
        assert top_values[0]["count"] == 3
        assert abs(top_values[0]["frequency_pct"] - 50.0) < 1.0
    
    def test_rare_values_count(self):
        # Create data where some values appear < 1% of the time  
        # With 200 items, 1% = 2, so values appearing once are rare
        data = ["Common"] * 198 + ["Rare1", "Rare2"]  # Each rare value appears 0.5% of time
        series = pd.Series(data)
        stats = compute_categorical_stats(series)
        
        assert stats["rare_values_count"] == 2


class TestComputeDatetimeStats:
    """Test datetime statistics computation."""
    
    def test_datetime_range(self):
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(10)]
        series = pd.Series(dates)
        stats = compute_datetime_stats(series)
        
        assert "min_date" in stats
        assert "max_date" in stats
        assert stats["coverage_days"] == 9  # 10 dates = 9 days difference
    
    def test_string_datetime_conversion(self):
        series = pd.Series(["2023-01-01", "2023-01-10"])
        stats = compute_datetime_stats(series)
        
        assert stats["coverage_days"] == 9


class TestComputeTextStats:
    """Test text statistics computation."""
    
    def test_average_length(self):
        series = pd.Series(["Hi", "Hello", "Hello World"])
        stats = compute_text_stats(series)
        
        expected_avg = (2 + 5 + 11) / 3
        assert abs(stats["avg_length"] - expected_avg) < 0.1
    
    def test_empty_strings_percentage(self):
        series = pd.Series(["", "Hello", "", "World"])
        stats = compute_text_stats(series)
        
        assert stats["pct_empty_strings"] == 50.0  # 2 out of 4


class TestComputeDataQualityFlags:
    """Test data quality flag detection."""
    
    def test_high_missing_flag(self):
        # Create series with >20% missing values
        series = pd.Series([1, 2, None, None, None])
        flags = compute_data_quality_flags(series, "numeric", {})
        
        assert "high_missing" in flags
    
    def test_likely_id_flag(self):
        # Create series where each value is unique (ID-like)
        series = pd.Series([f"ID_{i}" for i in range(10)])
        flags = compute_data_quality_flags(series, "text", {})
        
        assert "likely_id" in flags
    
    def test_extreme_outliers_flag(self):
        series = pd.Series([1, 2, 3, 4, 5])
        stats = {"outliers_iqr_count": 10}  # Mock high outlier count
        flags = compute_data_quality_flags(series, "numeric", stats)
        
        assert "extreme_outliers" in flags


class TestProfileColumn:
    """Test complete column profiling."""
    
    def test_numeric_column_profile(self):
        series = pd.Series([1, 2, 3, 4, 5], name="test_col")
        profile = profile_column(series, "test_col")
        
        assert profile["name"] == "test_col"
        assert profile["inferred_type"] == "numeric"
        assert profile["non_null_pct"] == 100.0
        assert profile["n_unique"] == 5
        assert "min" in profile
        assert "max" in profile
        assert isinstance(profile["quality_issues"], list)
    
    def test_categorical_column_profile(self):
        series = pd.Series(["A", "B", "A", "C"] * 5, name="category_col")
        profile = profile_column(series, "category_col")
        
        assert profile["inferred_type"] == "categorical"
        assert "top_k_values" in profile
        assert len(profile["top_k_values"]) > 0


class TestProfileDf:
    """Test complete DataFrame profiling."""
    
    def test_basic_dataframe_profile(self):
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "active": [True, False, True, True, False]
        })
        
        profile = profile_df(df)
        
        # Check dataset-level stats
        assert profile["dataset"]["n_rows"] == 5
        assert profile["dataset"]["n_cols"] == 4
        assert profile["dataset"]["duplicate_rows_pct"] == 0.0
        assert profile["dataset"]["memory_usage_mb"] > 0
        
        # Check columns
        assert len(profile["columns"]) == 4
        column_names = [col["name"] for col in profile["columns"]]
        assert set(column_names) == {"id", "name", "age", "active"}
    
    def test_empty_dataframe_profile(self):
        df = pd.DataFrame()
        profile = profile_df(df)
        
        assert profile["dataset"]["n_rows"] == 0
        assert profile["dataset"]["n_cols"] == 0
        assert profile["columns"] == []
    
    def test_dataframe_with_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 2, 1],
            "b": ["x", "y", "x"]
        })
        
        profile = profile_df(df)
        
        # Should detect duplicate (first and third rows are identical)
        assert profile["dataset"]["duplicate_rows_pct"] > 0
    
    def test_dataframe_with_missing_values(self):
        df = pd.DataFrame({
            "complete": [1, 2, 3, 4, 5],
            "partial": [1, None, 3, None, 5]
        })
        
        profile = profile_df(df)
        
        # Find the partial column profile
        partial_col = next(col for col in profile["columns"] if col["name"] == "partial")
        assert partial_col["non_null_pct"] == 60.0  # 3 out of 5 non-null
        
        # Should flag high missing values
        assert "high_missing" in partial_col["quality_issues"]


if __name__ == "__main__":
    pytest.main([__file__])
