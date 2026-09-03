# Session 7 — LLM Deep Dive, Prompting, Challenges & Evaluation
**Date:** 7 September 2026

## Topics
- LLM Deep Dive — history, model access, fine-tuning vs prompting
- Prompting Techniques — zero/one/few-shot, CoT, role prompting, temperature
- LLM Challenges & Risks — hallucinations, data privacy, jailbreaking, sycophancy, environmental impact
- LLM Evaluation — classical metrics, benchmarks, leaderboards, LLM-as-judge, custom eval pipelines

## Folder Structure
```
material/Session 7/
├── notebooks/
│   ├── 01_prompting_techniques.ipynb   # Hands-on prompting exercises
│   └── 02_llm_evaluation.ipynb         # LLM evaluation techniques
├── getting-started/                    # Promptfoo example (Ollama, no API key)
│   ├── README.md
│   └── promptfooconfig.yaml
├── llm_evals.md                        # Full LLM evaluation reference
├── pre-reading.md                      # Student pre-reading guide (~15-20 min)
└── README.md                           # This file
```

## Pre-Reading
See [pre-reading.md](./pre-reading.md) for the student pre-reading guide. Estimated time: 15-20 minutes.

## Prerequisites

- Python 3.12 and the course `.venv`
- [Ollama](https://ollama.com) installed and running with the `llama3.2` model:

  ```bash
  ollama pull llama3.2
  ```

- No API key is needed for the core notebook activities
- The optional OpenAI Evals extension requires `OPENAI_API_KEY`; when applicable, also set `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID`
- The optional Promptfoo activity requires Node.js because Promptfoo is an external CLI tool
- Keep `.env` out of version control and never commit tokens or other secrets

## Running the Notebooks

### Setup
1. Activate the virtual environment from the repository root.

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the repository in VS Code.

4. In VS Code, navigate to `material/Session 7/notebooks/`

5. Open `01_prompting_techniques.ipynb` and select `.venv (Python 3.12)` as the notebook kernel.

6. Confirm Ollama is running before executing cells that call the local model. Set `OLLAMA_MODEL` or `OLLAMA_BASE_URL` only when you need to override the defaults.

## Notebooks

**01_prompting_techniques.ipynb**
Hands-on prompting exercises covering zero-shot, one-shot, and few-shot prompting; chain-of-thought; role prompting; temperature settings; and live demonstrations of hallucination, sycophancy, and non-determinism. Includes a Grammar Correction Bot implementation.

**02_llm_evaluation.ipynb**
LLM evaluation techniques including classical metrics (BLEU, ROUGE, BERTScore, perplexity), Promptfoo setup, DIY evaluation loops, and LLM-as-judge approaches. The OpenAI Evals section is optional.

## Reference Material
- [llm_evals.md](./llm_evals.md) — Full technical reference for LLM evaluation (metrics, benchmarks, leaderboards, LLM-as-judge, custom eval pipelines)
- [getting-started/](./getting-started/) — Promptfoo example using Ollama (no API key needed)

## Key References
- [Prompting Guide](https://www.promptingguide.ai)
- Wei et al. (2022) — Chain-of-Thought Prompting
- Kojima et al. (2022) — Zero-shot CoT ("Let's think step by step")
- [Artificial Analysis Leaderboard](https://artificialanalysis.ai) — model cost/speed/quality comparison
