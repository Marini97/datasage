"""
Data profiling module for DataSage.

This module provides functions to analyse pandas DataFrames and compute
comprehensive statistics and data quality metrics.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


def infer_column_type(series: pd.Series) -> str:
    """Infer the semantic type of a pandas Series.
    
    Args:
        series: The pandas Series to analyse
        
    Returns:
        One of: 'numeric', 'categorical', 'datetime', 'text'
    """
    # Check for datetime first
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Try to parse as datetime if string-like
    if series.dtype == "object" and len(series.dropna()) > 0:
        sample = series.dropna().iloc[:min(100, len(series.dropna()))]
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return "datetime"
        except (ValueError, TypeError):
            pass
    
    # Check for numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    
    # Try to convert to numeric
    if series.dtype == "object":
        try:
            pd.to_numeric(series.dropna(), errors="raise")
            return "numeric"
        except (ValueError, TypeError):
            pass
    
    # Check if categorical (low cardinality relative to size)
    if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
        non_null_count = series.notna().sum()
        unique_count = series.nunique()
        
        # Consider categorical if unique values are less than 20% of non-null values
        # or if there are fewer than 50 unique values
        if non_null_count > 0 and (unique_count / non_null_count < 0.2 or unique_count <= 50):
            return "categorical"
    
    # Default to text
    return "text"


def get_sample_values(series: pd.Series, max_samples: int = 3) -> List[str]:
    """Get a sample of non-null values from a series.
    
    Args:
        series: The pandas Series to sample from
        max_samples: Maximum number of samples to return
        
    Returns:
        List of string representations of sample values
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return []
    
    # Get unique values first, then sample
    unique_values = non_null.unique()
    sample_size = min(max_samples, len(unique_values))
    samples = np.random.choice(unique_values, size=sample_size, replace=False)
    
    return [str(val) for val in samples]


def compute_numeric_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute statistics for numeric columns.
    
    Args:
        series: Numeric pandas Series
        
    Returns:
        Dictionary of numeric statistics
    """
    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(series, errors="coerce")
    
    if numeric_series.notna().sum() == 0:
        return {}
    
    stats = {
        "min": float(numeric_series.min()),
        "max": float(numeric_series.max()),
        "mean": float(numeric_series.mean()),
        "std": float(numeric_series.std()),
        "q1": float(numeric_series.quantile(0.25)),
        "q3": float(numeric_series.quantile(0.75)),
    }
    
    # Compute IQR and outliers
    iqr = stats["q3"] - stats["q1"]
    stats["iqr"] = float(iqr)
    
    # Count outliers using 1.5 * IQR rule
    lower_bound = stats["q1"] - 1.5 * iqr
    upper_bound = stats["q3"] + 1.5 * iqr
    outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
    stats["outliers_iqr_count"] = int(len(outliers))
    
    # Count negative values (useful for columns that should be non-negative)
    stats["negative_count"] = int((numeric_series < 0).sum())
    
    # Percentage of zeros
    zero_count = (numeric_series == 0).sum()
    total_count = numeric_series.notna().sum()
    stats["zeros_pct"] = float(zero_count / total_count * 100) if total_count > 0 else 0.0
    
    return stats


def compute_categorical_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute statistics for categorical columns.
    
    Args:
        series: Categorical pandas Series
        
    Returns:
        Dictionary of categorical statistics
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return {}
    
    value_counts = non_null.value_counts()
    total_count = len(non_null)
    
    # Top 5 most frequent values
    top_k = min(5, len(value_counts))
    top_values = []
    for value, count in value_counts.head(top_k).items():
        frequency_pct = count / total_count * 100
        top_values.append({
            "value": str(value),
            "count": int(count),
            "frequency_pct": float(frequency_pct)
        })
    
    # Count rare values (frequency < 1%)
    rare_threshold = total_count * 0.01
    rare_values_count = (value_counts < rare_threshold).sum()
    
    return {
        "top_k_values": top_values,
        "rare_values_count": int(rare_values_count)
    }


def compute_datetime_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute statistics for datetime columns.
    
    Args:
        series: Datetime pandas Series
        
    Returns:
        Dictionary of datetime statistics
    """
    # Try to convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            datetime_series = pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        except (ValueError, TypeError):
            return {}
    else:
        datetime_series = series
    
    non_null = datetime_series.dropna()
    if len(non_null) == 0:
        return {}
    
    min_date = non_null.min()
    max_date = non_null.max()
    
    # Calculate coverage in days
    coverage_days = (max_date - min_date).days
    
    return {
        "min_date": min_date.isoformat(),
        "max_date": max_date.isoformat(),
        "coverage_days": int(coverage_days)
    }


def compute_text_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute statistics for text columns.
    
    Args:
        series: Text pandas Series
        
    Returns:
        Dictionary of text statistics
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return {}
    
    # Convert to string
    text_series = non_null.astype(str)
    
    # Average length
    lengths = text_series.str.len()
    avg_length = float(lengths.mean())
    
    # Percentage of empty strings
    empty_strings = (text_series == "").sum()
    total_count = len(text_series)
    pct_empty_strings = float(empty_strings / total_count * 100) if total_count > 0 else 0.0
    
    return {
        "avg_length": avg_length,
        "pct_empty_strings": pct_empty_strings
    }


def compute_data_quality_flags(
    series: pd.Series, 
    column_type: str, 
    stats: Dict[str, Any]
) -> List[str]:
    """Compute data quality issue flags for a column.
    
    Args:
        series: The pandas Series
        column_type: The inferred column type
        stats: The computed statistics
        
    Returns:
        List of data quality issue flags
    """
    issues = []
    
    # High missing values (>20%)
    non_null_pct = (series.notna().sum() / len(series)) * 100
    if non_null_pct < 80:
        issues.append("high_missing")
    
    # Likely ID column
    if series.nunique() / len(series) > 0.95:
        issues.append("likely_id")
    
    # Type-specific issues
    if column_type == "numeric" and stats:
        # Extreme outliers
        if stats.get("outliers_iqr_count", 0) > len(series) * 0.05:  # >5% outliers
            issues.append("extreme_outliers")
        
        # Skewed distribution
        q1, q3 = stats.get("q1", 0), stats.get("q3", 0)
        mean, median = stats.get("mean", 0), (q1 + q3) / 2
        if abs(mean - median) > abs(q3 - q1):  # Mean far from median
            issues.append("skewed")
    
    elif column_type == "categorical" and stats:
        # Many rare categories
        if stats.get("rare_values_count", 0) > 10:
            issues.append("many_rare_categories")
    
    elif column_type == "text" and stats:
        # Mostly empty strings
        if stats.get("pct_empty_strings", 0) > 50:
            issues.append("mostly_empty")
    
    return issues


def profile_column(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """Profile a single column of a DataFrame.
    
    Args:
        series: The pandas Series to profile
        column_name: Name of the column
        
    Returns:
        Dictionary containing column profile
    """
    # Basic info
    total_count = len(series)
    non_null_count = series.notna().sum()
    non_null_pct = (non_null_count / total_count * 100) if total_count > 0 else 0.0
    
    # Infer type
    column_type = infer_column_type(series)
    
    # Base profile
    profile = {
        "name": column_name,
        "inferred_type": column_type,
        "non_null_pct": float(non_null_pct),
        "n_unique": int(series.nunique()),
        "sample_values": get_sample_values(series)
    }
    
    # Type-specific statistics
    if column_type == "numeric":
        profile.update(compute_numeric_stats(series))
    elif column_type == "categorical":
        profile.update(compute_categorical_stats(series))
    elif column_type == "datetime":
        profile.update(compute_datetime_stats(series))
    elif column_type == "text":
        profile.update(compute_text_stats(series))
    
    # Data quality flags
    profile["quality_issues"] = compute_data_quality_flags(series, column_type, profile)
    
    return profile


def profile_df(df: pd.DataFrame) -> Dict[str, Any]:
    """Profile a pandas DataFrame comprehensively.
    
    Args:
        df: The pandas DataFrame to profile
        
    Returns:
        Dictionary containing the complete data profile
    """
    if df.empty:
        return {
            "dataset": {
                "n_rows": 0,
                "n_cols": 0,
                "memory_usage_mb": 0.0,
                "duplicate_rows_pct": 0.0
            },
            "columns": []
        }
    
    # Dataset-level statistics
    n_rows, n_cols = df.shape
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    duplicate_rows_pct = (duplicate_count / n_rows * 100) if n_rows > 0 else 0.0
    
    dataset_profile = {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "memory_usage_mb": float(memory_usage_mb),
        "duplicate_rows_pct": float(duplicate_rows_pct)
    }
    
    # Profile each column
    column_profiles = []
    for column_name in df.columns:
        column_profile = profile_column(df[column_name], column_name)
        column_profiles.append(column_profile)
    
    return {
        "dataset": dataset_profile,
        "columns": column_profiles
    }
