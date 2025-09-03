"""
DataSage: A fully local LLM-assisted data quality profiler.

This package provides tools to profile CSV files and pandas DataFrames,
generating comprehensive data quality reports using local language models.
"""

__version__ = "0.1.0"
__author__ = "DataSage Team"

# Import core functions - these will be available when dependencies are installed
try:
    from .profiler import profile_df
    from .model import LocalLLMGenerator
    from .report_formatter import ReportFormatter
    
    __all__ = ["profile_df", "LocalLLMGenerator", "ReportFormatter"]
except ImportError:
    # Dependencies not installed yet - this is OK during development
    __all__ = []
