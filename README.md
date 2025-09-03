# DataSage 🔮

> *"What if your data could actually talk to you? Let's find out!"*

A fun experiment in making AI help you understand your messy datasets! This started as a side project where I wanted to learn how to build local LLM tools that don't send your data to some random cloud service.

## What's this all about? 🤔

Ever looked at a CSV file and thought "I have no idea what's going on here"? Same! DataSage takes your data, runs it through some stats, then asks a local AI to explain what's interesting (or broken) about it.

**The good stuff:**
- 🔒 **Privacy focused** - Your data stays on your machine (because I'm paranoid too)
- 📊 **Actually useful profiling** - Finds the weird stuff in your columns
- 🤖 **AI that runs locally** - No API keys, no monthly bills, no judgment
- ✅ **Fact-checking included** - Because AIs sometimes make stuff up
- ⚡ **Works on regular laptops** - No fancy GPU required (but we'll use it if you have one!)
- 🎯 **Honest about failures** - When the AI gets confused, we fall back to boring but accurate reports

## Wait, why would I use this? 🧐

- You've got a CSV and want to know if it's any good
- You're tired of manually checking for missing values and outliers
- You want to impress people with fancy AI-generated reports
- You're learning about data quality (like me!)
- You don't trust cloud services with your data

## Quickstart (the fun way) 🚀

```bash
# First, get set up
git clone <this-repo>
cd datasage
python -m venv .venv
source .venv/bin/activate  # Windows folks: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# OR if you prefer the modern way: pip install -e .

# Now the magic happens
python -m datasage.cli profile your_messy_data.csv

# Want just the highlights?
python -m datasage.cli profile data.csv --summary-only

# Curious about what's happening under the hood?
python -m datasage.cli profile data.csv --debug
```

## What you'll get 📊

DataSage spits out a nice Markdown report that actually makes sense:

- **Dataset Overview**: "You've got X rows and Y columns, here's what's interesting..."
- **Quality Issues**: "Heads up, column Z is acting weird"
- **Column-by-column breakdown**: What each column is about and any red flags
- **Recommendations**: Actual actionable advice (when the AI cooperates)

## The technical bits (if you're into that) 🤓

- **Model**: google/flan-t5-base (about 250MB, runs locally)
- **Speed**: A few seconds for typical datasets
- **Memory**: 1-2GB while running (not bad!)
- **GPU**: Will use it if you have one, works fine without
- **Honesty**: Tells you when the AI output doesn't look right

## Example Output

```markdown
# Data Quality Report

Generated on 3 September 2025

## Dataset Overview
This dataset contains 1,000 customer records with 5 columns. The data uses approximately 0.2MB of memory.

## Data Quality Summary
- ✅ No duplicate rows detected
- ⚠️ Missing values found in 2 columns (email: 15%, phone: 8%)
- ⚠️ Potential outliers detected in salary column

## Column Profiles

### customer_id (Numeric)
Unique identifier column with values ranging from 1 to 1,000. No missing values or duplicates.

### salary (Numeric)
Annual salary data with mean £45,230. **Data quality issue**: 12 extreme outliers detected above £150,000 may need investigation.
```

## How It Works

```
CSV/DataFrame → Statistical Profile → LLM Prompts → Markdown Report
      ↓                ↓                  ↓              ↓
   pandas           JSON stats      Local Model    Verified Output
```

1. **Statistical Profiling**: Compute comprehensive statistics for each column
2. **Prompt Generation**: Convert stats into concise, factual prompts (<800 tokens)
3. **Local LLM**: Generate plain-English explanations using deterministic settings
4. **Verification**: Cross-check numeric claims and flag discrepancies
5. **Report Assembly**: Combine chunks into a structured Markdown document

## Configuration

DataSage uses local models by default. Configure via environment variables:

```bash
export DATASAGE_MODEL=google/flan-t5-base  # Default
export DATASAGE_MAX_TOKENS=300
export DATASAGE_TEMPERATURE=0  # Deterministic
```

Supported models: Any HuggingFace text2text-generation model that runs on CPU.

## Limitations

- Very wide datasets (>50 columns) are summarised rather than detailed
- Large files (>100MB) may benefit from sampling (future feature)
- Currently supports CSV input; Excel and other formats planned
- No integration with data validation frameworks yet (see Extensions)

## Development

```bash
# Set up development environment
pip install -r requirements.txt  # Install dependencies
pip install -e .                  # Install in development mode
# OR: pip install -e ".[dev]"     # If using pyproject.toml dev extras

# Run tests
pytest

# Format code
black datasage/
ruff check datasage/

# Type checking
mypy datasage/
```

## Current status (the honest bit) 🤷‍♂️

**What works great:**
- CSV profiling is rock solid
- Quality detection is pretty good
- Fallback reports are actually useful
- GPU auto-detection (though I don't have one to test!)
- The whole CLI experience feels nice

**What's... a work in progress:**
- The AI sometimes gets stuck in repetition loops (hence the fallback system)
- Probably needs a better model for more coherent text generation
- Only handles CSV files for now
- The AI's output quality can be unpredictable

**The bottom line:** Even when the AI has an off day, you still get useful data quality insights. And when it works well, it's pretty magical!

## What I learned building this 🎓

1. Local LLMs are cool but can be finicky
2. Fallback systems are absolutely essential  
3. Auto device detection is surprisingly tricky
4. Users appreciate honest error messages
5. Sometimes the simple template-based reports are just as useful as AI-generated ones

## Future ideas (maybe?) 💭

- Try different/better models for text generation
- Support for Excel, JSON, Parquet files  
- Web interface (because CLIs aren't for everyone)
- Better prompt engineering to reduce repetition
- Maybe some visualizations?

## Extensions (Roadmap)

- 📊 Streamlit web interface
- 📋 Great Expectations integration
- 📁 Multi-file batch processing
- 🎯 Custom rule definitions
- 📈 Temporal data quality tracking

## Licence

MIT - see [LICENSE](LICENSE) for details.

---

*Built with ☕ and curiosity. Your data stays local, the insights travel far!*
