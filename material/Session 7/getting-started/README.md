# getting-started (Getting Started Example)

This is a simple example that demonstrates the basic functionality of promptfoo. It tests a translation prompt using a locally running [Ollama](https://ollama.com) model — no cloud API keys required.

For full documentation, see the [promptfoo Getting Started guide](https://promptfoo.dev/docs/getting-started).

## Create your own eval project

To create a new promptfoo project from scratch using this same template, run:

```bash
npx promptfoo@latest init --example getting-started
cd getting-started
```

This generates a `promptfooconfig.yaml` with sample prompts, providers, and test cases ready to run. To use a local Ollama model instead of a cloud API, update the `providers` section in `promptfooconfig.yaml` to point to your Ollama model (as shown in this example).

## Prerequisites

- [Node.js](https://nodejs.org/en/download) installed
- [Ollama](https://ollama.com) installed and running locally with the `gpt-oss:latest` model pulled:

```bash
ollama pull gpt-oss:latest
```

## Setup

1. Run Promptfoo through `npx` so a global installation is not required:

```bash
npx promptfoo@latest --version
```

2. Run the evaluation:

```bash
npx promptfoo@latest eval
```

3. Open the web UI to review results:

```bash
npx promptfoo@latest view
```

## What's happening?

This example:

- Tests a translation prompt against a locally running Ollama model (`gpt-oss:latest`)
- Uses two test cases with different target languages and inputs
- Checks outputs with simple string assertions and an LLM-based rubric grader

The configuration in `promptfooconfig.yaml` shows:

- How to define prompts with variables using `{{variable_name}}`
- How to configure a local Ollama provider instead of a cloud API
- How to set up test cases with assertions
- How to use `llm-rubric` for qualitative output grading

> **Caution:** `llm-rubric` expects its grader model to return structured JSON. Smaller open-source models may follow that format loosely—for example, by adding prose, echoing the rubric, or returning inconsistent fields—which can produce `Could not extract JSON from llm-rubric response` even when the answer being evaluated is correct. This example therefore uses `gpt-oss:latest`. If you experiment with a smaller model as the model under test, keep a more reliable model such as `gpt-oss:latest` as `defaultTest.options.provider` for grading.
