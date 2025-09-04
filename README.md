# DataSage 🧙‍♂️

> *"Your AI companion for understanding messy datasets - now with interactive chat and comprehensive data exploration!"*

DataSage is a professional-grade data quality profiler powered by local AI that helps you understand your datasets through comprehensive analysis, interactive visualizations, and intelligent insights - all while keeping your data completely private.

## ✨ What Makes DataSage Special

- 🔒 **Privacy First** - Your data never leaves your machine
- 🤖 **Multi-Model AI** - OpenAI API support + local models with intelligent fallbacks
- 💬 **Interactive Chat** - Ask questions about your data and get AI-powered answers
- 📊 **Professional Visualizations** - Comprehensive data exploration with charts and plots
- ⚡ **Smart Fallbacks** - Always provides valuable insights, even when AI struggles
- 🎯 **Data Scientist Tools** - Distribution analysis, correlation matrices, outlier detection
- 🌐 **Modern Web Interface** - Beautiful Streamlit UI with real-time analysis

## 🚀 Quick Start

### Installation
```bash
# Clone and set up
git clone https://github.com/Marini97/datasage.git
cd datasage
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch the Web Interface (Recommended)
```bash
python -m datasage.cli web
# Opens at http://localhost:8501
```

### Command Line Usage
```bash
# Analyze a dataset
python -m datasage.cli profile data.csv

# With enhanced AI (if OpenAI API available)
export OPENAI_API_KEY="your-key-here"
python -m datasage.cli profile data.csv
```

## 🎯 Three Powerful Interfaces

### 1. 📊 Dataset Explorer
- **Interactive file upload** with drag-and-drop support
- **Sample datasets** (Tips, Titanic, Iris, etc.) for testing
- **Comprehensive statistics** and data quality metrics
- **Visual exploration** with distribution plots, correlation matrices
- **Missing data analysis** with pattern visualization
- **Outlier detection** with interactive charts

### 2. � AI Report Generation  
- **Multi-tier AI system** for highest quality reports
- **Professional analysis** with actionable recommendations
- **Quality assessment** across completeness, consistency, integrity
- **Statistical fallback** ensures always useful output
- **Downloadable reports** in Markdown format

### 3. 💬 Interactive Chat
- **Ask anything** about your dataset
- **Real-time AI responses** with full dataset context
- **Conversation history** to track your analysis
- **Quick insight buttons** for common questions
- **Enhanced with exploration data** for accurate answers

## 🔍 Data Science Features

### Professional Analysis Tools
- **Distribution Analysis**: Histograms, KDE plots, skewness detection
- **Correlation Matrix**: Heatmaps with strength indicators  
- **Missing Data Patterns**: Visual missing data analysis
- **Outlier Detection**: IQR-based detection with box plots
- **Data Quality Metrics**: Comprehensive quality assessment
- **Statistical Summaries**: Professional-grade statistics

### AI-Powered Insights
```python
# Example AI responses:
"Your dataset shows strong positive correlation (0.89) between 
total_bill and tip amounts, suggesting tip percentage increases 
with bill size. The 3 outliers in the tip column (>$5.00) may 
represent special occasions or data entry errors..."
```

## 🤖 Advanced AI Architecture

### Multi-Model Support
1. **OpenAI API** (highest quality) - GPT-3.5/4 when API key available
2. **Local FLAN-T5** - Fast, private, runs on CPU/GPU
3. **Statistical Fallback** - Professional reports when AI unavailable

### Enhanced Prompting
- **Few-shot examples** for consistent output quality
- **Statistical context** from comprehensive data exploration
- **Structured prompts** with data scientist insights
- **Quality validation** with automatic fallbacks

## 📋 Example Output

```markdown
# Data Quality Report: Restaurant Tips Dataset

## Dataset Overview
- **244 rows × 7 columns** using 0.13 MB memory
- **Mix of numerical and categorical data**
- **6 outliers detected** across 2 columns
- **High correlation (0.89)** between total_bill and tip

## Key Insights
- Strong tipping patterns correlate with bill amounts
- Weekend meals show 15% higher average tips
- 3 potential data entry errors in tip column
- No missing values detected (excellent completeness)

## Recommendations
1. Investigate outliers in tip column (>$5.00)
2. Consider tip percentage analysis for better insights
3. Weekend vs weekday analysis shows business opportunities
```

## ⚙️ Configuration

### Basic Configuration
```bash
# Use sample datasets
python -m datasage.cli web

# Debug mode for troubleshooting
python -m datasage.cli profile data.csv --debug
```

### Enhanced AI Setup
```bash
# For best AI quality, set OpenAI API key
export OPENAI_API_KEY="sk-..."
export OPENAI_API_BASE="https://api.openai.com/v1"  # Optional

# Then run normally - will automatically use OpenAI
python -m datasage.cli profile data.csv
```

### Custom Model Configuration  
```bash
export DATASAGE_MODEL="microsoft/DialoGPT-small"  # Custom local model
export DATASAGE_MAX_TOKENS=500
export DATASAGE_TEMPERATURE=0.1
```

## 🛠️ Development

```bash
# Development setup
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Test web interface
python test_web.py

# Code formatting
black datasage/
ruff check datasage/
```

## 📊 What's Included

### Core Modules
- `profiler.py` - Comprehensive statistical analysis
- `model.py` - Multi-model LLM support with fallbacks  
- `enhanced_generator.py` - OpenAI + local model orchestration
- `prompt_builder.py` - Advanced prompt engineering
- `report_formatter.py` - Professional report assembly
- `web_app.py` - Interactive Streamlit interface
- `cli.py` - Command line interface

### Key Features
- ✅ **Multi-format support** (CSV with more formats planned)
- ✅ **Interactive data exploration** with professional visualizations
- ✅ **AI chat interface** for dataset Q&A
- ✅ **Quality assessment** with actionable recommendations
- ✅ **Outlier detection** and missing data analysis
- ✅ **Correlation analysis** with statistical significance
- ✅ **Memory usage optimization** for large datasets

## 🎯 Use Cases

**Data Scientists & Analysts:**
- Quick dataset understanding and quality assessment
- Professional-grade statistical analysis and visualization
- Interactive exploration before modeling

**Business Users:**
- Understand data quality issues in business datasets  
- Get AI-powered insights without technical expertise
- Generate reports for stakeholders

**Students & Researchers:**
- Learn data analysis through interactive exploration
- Understand statistical concepts through visualization
- Practice data quality assessment

## 🔮 Recent Enhancements

### v0.2.0 Features
- ✨ **Interactive Chat Interface** - Ask questions about your data
- � **Comprehensive Data Exploration** - Professional visualizations
- 🤖 **Enhanced AI Integration** - OpenAI API + improved local models
- 🎯 **Quality Improvements** - Better prompts and statistical fallbacks
- 🌐 **Improved UI** - Reorganized interface with better UX

## 🗺️ Roadmap

**Coming Soon:**
- � Advanced visualization options (scatter plots, time series)
- 📁 Excel, Parquet, and JSON file support  
- 🔗 Integration with pandas profiling and Great Expectations
- 📊 Export visualizations and reports to PDF
- 🤝 Collaboration features for team data analysis

**Future Vision:**
- 🧠 Advanced ML-powered data insights
- 🔄 Real-time data monitoring and alerting
- 📱 Mobile-responsive interface
- 🌍 Multi-language support

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

We welcome contributions! Check out our issues or submit your own ideas.

---

*Built with 🧙‍♂️ magic and ☕ coffee. Your data stays private, your insights go far!*
