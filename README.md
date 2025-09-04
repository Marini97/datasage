# DataSage — LLM Data Quality Profiler

> Understand messy datasets fast, with privacy-first analysis, clear visualisations, and optional AI-generated summaries.

---

## Why DataSage?

- **Privacy-first:** your data stays on your machine by default (local model).  
- **Actionable insights:** concise, explainable summaries of quality issues.  
- **Two ways to use:** a friendly web UI *and* a simple CLI.  
- **Batteries included:** distributions, correlations, missingness patterns, and outlier detection with professional-looking charts.  
- **Optional AI:** plug in an API key to upgrade summaries; otherwise DataSage still produces useful, deterministic reports.

---

## What it does

Given a CSV (or a `pandas` DataFrame), DataSage:

1. Profiles the dataset (shape, memory, types, uniques, missingness, outliers, correlations).
2. Builds a short, plain-English report (local LLM by default; optionally OpenAI).
3. Lets you **explore interactively** (histograms, heatmaps, missingness, outliers) or script it from the CLI.
4. Exports a clean Markdown report you can paste into a notebook, issue, or slide.

---

## Quick start

### 1) Install

```bash
git clone https://github.com/Marini97/datasage.git
cd datasage
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the web app (recommended)

```bash
python -m datasage.cli web
# Then open http://localhost:8501
```

### 3) Or run from the CLI

```bash
# Basic profiling (local, no API keys needed)
python -m datasage.cli profile path/to/data.csv

# Save a Markdown report
python -m datasage.cli profile path/to/data.csv -o report.md
```

---

## Optional: enhanced AI summaries

If you want higher-quality narrative summaries (on top of the local model / statistical fallback), export your key:

```bash
export OPENAI_API_KEY="sk-..."
# (Optional) custom base: export OPENAI_API_BASE="https://api.openai.com/v1"
```

Then re-run the same commands; DataSage will automatically use the remote model when available.

> Prefer local only? Do nothing — DataSage runs with a local model or a purely statistical fallback.

---

## Configuration

Environment variables (all optional):

```bash
# Local model name (Hugging Face)
export DATASAGE_MODEL="google/flan-t5-base"

# Generation controls (keep low for determinism)
export DATASAGE_MAX_TOKENS=300
export DATASAGE_TEMPERATURE=0.0
```

CLI flags (selected):

```bash
python -m datasage.cli profile data.csv   -o report.md   --summary-only   --debug
```

---

## Features at a glance

- **Dataset overview:** rows/columns, memory footprint, per-type breakdown.
- **Per-column stats:** type, non-null %, uniques, sample values.
- **Numerical:** min/max/mean/std, quartiles & IQR, zero/negative counts, outliers.
- **Categorical:** top-k values with frequencies, rare categories.
- **Datetime:** min/max, coverage period.
- **Text:** average length, empty-string rate.
- **Quality flags:** high missingness, likely IDs, skew, duplicates, many rares.
- **Visuals:** histograms/KDE, correlation heatmaps, missingness patterns, boxplots.
- **Reports:** Markdown with Overview, Quality Summary, Column Profiles.
- **Chat (web app):** ask questions about your data; responses are grounded in the profile.

---

## Example

```text
# Data Quality Report

## Dataset Overview
- 244 rows × 7 columns (≈0.13 MB)
- Mixed numeric and categorical types
- 6 outliers across 2 columns
- High correlation (0.89) between total_bill and tip

## Key Insights
- Tipping patterns scale with bill amount
- No missing values detected
- 3 potential data entry errors in “tip”
```

*(Numbers above are illustrative; your dataset will differ.)*

---

## Project structure

```
.
├─ datasage/                # app + CLI + profiling + prompts + reporting
├─ examples/                # sample CSVs / demo assets
├─ tests/                   # unit tests
├─ .github/workflows/       # CI configuration
├─ README.md
├─ requirements.txt
├─ pyproject.toml
└─ LICENSE                  # MIT
```

**Core modules (in `datasage/`):**

- `cli.py` — entry point for CLI & web app
- `profiler.py` — dataframe → stats & quality signals
- `prompt_builder.py` — compact prompts from stats
- `model.py` — local LLM loader + generation helpers
- `enhanced_generator.py` — optional OpenAI + fallbacks
- `report_formatter.py` — stitch chunks into Markdown
- `web_app.py` — Streamlit interface

---

## Architecture

```
CSV → pandas → profiler ─┐
                         ├─→ prompt_builder → { local LLM | OpenAI | statistical fallback } → report_formatter → Markdown
Profiles & visuals ──────┘
                                 │
                                 └─────────────────────────────→ Streamlit UI (explore + chat)
```

- **Deterministic by default:** temperature `0.0`, bounded tokens.  
- **Graceful degradation:** if remote LLMs are absent/unavailable, DataSage still produces a clear, useful report.  
- **No data leaves the machine** unless you opt into a remote API.

---

## Development

```bash
# Dev install
pip install -r requirements.txt
pip install -e .

# Tests
pytest tests/

# Lint/format (if configured in pyproject)
ruff check datasage/
black datasage/
```

---

## Licence

MIT — see `LICENSE`.

---

*Built with 🧙‍♂️ magic and ☕ coffee. Your data stays private, your insights go far!*
