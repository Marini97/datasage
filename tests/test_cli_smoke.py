"""
Smoke tests for the CLI module.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock
import pandas as pd

from datasage.cli import app
from typer.testing import CliRunner


class TestCLISmoke:
    """Smoke tests for CLI functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        
        # Create a temporary CSV file for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv_path = Path(self.temp_dir) / "test_data.csv"
        
        # Create sample data
        test_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, None, 45],
            'salary': [50000, 60000, 70000, 65000, 80000],
            'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales']
        })
        test_data.to_csv(self.test_csv_path, index=False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        if self.test_csv_path.exists():
            os.remove(self.test_csv_path)
        os.rmdir(self.temp_dir)
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_profile_command_basic(self, mock_llm_class):
        """Test basic profile command execution."""
        # Mock the LLM generator to avoid model loading
        mock_llm = Mock()
        mock_llm.generate_markdown.return_value = """
        ## Dataset Overview
        Small test dataset with 5 rows and 5 columns.
        
        ## Data Quality Summary
        - One column has missing values
        - No duplicates detected
        
        ## Column Profiles
        ### id
        Sequential identifier column.
        
        ### name
        Text names, all present.
        """
        mock_llm_class.return_value = mock_llm
        
        output_path = Path(self.temp_dir) / "test_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path)
        ])
        
        # Should exit successfully
        assert result.exit_code == 0
        
        # Should create output file
        assert output_path.exists()
        
        # Output file should contain expected content
        with open(output_path, 'r') as f:
            content = f.read()
            assert "Data Quality Report" in content
            assert "Dataset Overview" in content
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_profile_command_summary_only(self, mock_llm_class):
        """Test profile command with summary-only flag."""
        mock_llm = Mock()
        mock_llm.generate_markdown.return_value = """
        ## Executive Summary
        Small test dataset.
        
        ## Data Quality Assessment
        Generally good quality.
        
        ## Recommendations
        Address missing values in age column.
        """
        mock_llm_class.return_value = mock_llm
        
        output_path = Path(self.temp_dir) / "summary_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path),
            "--summary-only"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_profile_command_specific_columns(self, mock_llm_class):
        """Test profile command with specific columns."""
        mock_llm = Mock()
        mock_llm.generate_markdown.return_value = "## Dataset Overview\nFiltered dataset analysis."
        mock_llm_class.return_value = mock_llm
        
        output_path = Path(self.temp_dir) / "columns_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path),
            "--columns", "name,age,salary"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
    
    def test_profile_command_missing_file(self):
        """Test profile command with non-existent file."""
        result = self.runner.invoke(app, [
            "profile", 
            "nonexistent_file.csv"
        ])
        
        # Should fail with appropriate error
        assert result.exit_code != 0
    
    def test_profile_command_invalid_columns(self):
        """Test profile command with invalid column names."""
        output_path = Path(self.temp_dir) / "invalid_columns_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path),
            "--columns", "nonexistent_column"
        ])
        
        # Should fail with error about missing columns
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "not found" in str(result.exception).lower()
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_profile_command_debug_mode(self, mock_llm_class):
        """Test profile command with debug flag."""
        mock_llm = Mock()
        mock_llm.generate_markdown.return_value = "Debug test output"
        mock_llm_class.return_value = mock_llm
        
        output_path = Path(self.temp_dir) / "debug_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path),
            "--debug"
        ])
        
        assert result.exit_code == 0
        # Debug mode should still produce a valid report
        assert output_path.exists()
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_profile_command_no_verify(self, mock_llm_class):
        """Test profile command with verification disabled."""
        mock_llm = Mock()
        mock_llm.generate_markdown.return_value = "Test output with 100% accuracy"
        mock_llm_class.return_value = mock_llm
        
        output_path = Path(self.temp_dir) / "no_verify_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path),
            "--no-verify"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Should not contain verification notes
        with open(output_path, 'r') as f:
            content = f.read()
            assert "Verification Notes" not in content
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_profile_command_max_rows(self, mock_llm_class):
        """Test profile command with max rows limit."""
        mock_llm = Mock()
        mock_llm.generate_markdown.return_value = "Sampled dataset analysis"
        mock_llm_class.return_value = mock_llm
        
        output_path = Path(self.temp_dir) / "max_rows_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path),
            "--max-rows", "3"
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
    
    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "DataSage" in result.stdout
    
    @patch('datasage.cli.pd')
    @patch('datasage.cli.torch')
    @patch('datasage.cli.transformers')
    def test_info_command(self, mock_transformers, mock_torch, mock_pd):
        """Test info command."""
        # Mock package versions
        mock_pd.__version__ = "1.5.0"
        mock_torch.__version__ = "1.12.0"
        mock_torch.cuda.is_available.return_value = False
        mock_transformers.__version__ = "4.20.0"
        
        result = self.runner.invoke(app, ["info"])
        
        assert result.exit_code == 0
        assert "System Information" in result.stdout
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_profile_command_llm_error(self, mock_llm_class):
        """Test profile command when LLM fails to load."""
        # Mock LLM to raise an error
        mock_llm_class.side_effect = RuntimeError("Model loading failed")
        
        output_path = Path(self.temp_dir) / "error_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(self.test_csv_path),
            "-o", str(output_path)
        ])
        
        # Should exit with error
        assert result.exit_code == 1
        assert "Error generating LLM responses" in result.stdout
    
    def test_profile_command_default_output_path(self):
        """Test profile command with default output path."""
        with patch('datasage.cli.LocalLLMGenerator') as mock_llm_class:
            mock_llm = Mock()
            mock_llm.generate_markdown.return_value = "Default output test"
            mock_llm_class.return_value = mock_llm
            
            result = self.runner.invoke(app, [
                "profile", 
                str(self.test_csv_path)
            ])
            
            assert result.exit_code == 0
            
            # Should create report with default name
            expected_output = self.test_csv_path.parent / f"{self.test_csv_path.stem}_report.md"
            assert expected_output.exists()
            
            # Clean up
            if expected_output.exists():
                os.remove(expected_output)
    
    def test_profile_command_keyboard_interrupt(self):
        """Test profile command handling of keyboard interrupt."""
        with patch('datasage.cli.pd.read_csv') as mock_read_csv:
            # Mock to raise KeyboardInterrupt
            mock_read_csv.side_effect = KeyboardInterrupt()
            
            result = self.runner.invoke(app, [
                "profile", 
                str(self.test_csv_path)
            ])
            
            assert result.exit_code == 1
            assert "cancelled" in result.stdout.lower()


class TestCLIIntegration:
    """Integration tests that test the full pipeline with mocked LLM."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('datasage.cli.LocalLLMGenerator')
    def test_full_pipeline_with_real_csv(self, mock_llm_class):
        """Test the full pipeline with realistic CSV data."""
        # Create more realistic test data
        departments = ['Engineering', 'Sales', 'Marketing', 'HR', 'Finance']
        test_data = pd.DataFrame({
            'employee_id': range(1, 101),
            'first_name': [f'Person{i}' for i in range(1, 101)],
            'last_name': [f'Last{i}' for i in range(1, 101)],
            'email': [f'person{i}@company.com' if i % 10 != 0 else None for i in range(1, 101)],
            'age': [25 + (i % 40) for i in range(1, 101)],
            'salary': [40000 + (i * 1000) + (100 if i > 95 else 0) for i in range(1, 101)],  # Some outliers
            'department': [departments[i % 5] for i in range(100)],
            'hire_date': pd.date_range('2020-01-01', periods=100, freq='3D')
        })
        
        csv_path = Path(self.temp_dir) / "realistic_data.csv"
        test_data.to_csv(csv_path, index=False)
        
        # Mock LLM to return realistic output
        mock_llm = Mock()
        mock_llm.generate_markdown.return_value = """
        ## Dataset Overview
        This employee dataset contains 100 records with 8 columns representing staff information.
        
        ## Data Quality Summary
        - 10% missing email addresses need attention
        - Salary outliers detected for senior employees
        - Complete hiring date records spanning 3 years
        
        ## Column Profiles
        
        ### employee_id
        Sequential employee identifier from 1 to 100, functioning as primary key.
        
        ### email
        Email addresses with 10% missing values requiring follow-up.
        """
        mock_llm_class.return_value = mock_llm
        
        output_path = Path(self.temp_dir) / "realistic_report.md"
        
        result = self.runner.invoke(app, [
            "profile", 
            str(csv_path),
            "-o", str(output_path)
        ])
        
        assert result.exit_code == 0
        assert output_path.exists()
        
        # Verify report content
        with open(output_path, 'r') as f:
            content = f.read()
            assert "Data Quality Report" in content
            assert "100 records" in content or "employee" in content.lower()
            assert "Verification Notes" in content or "verified" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__])
