# Session 7 Pre-Reading — LLMs, Prompting & Evaluation

**Date:** 7 September 2026

## What this session covers

In Session 7, we'll take a hands-on deep dive into Large Language Models. You'll learn how to craft effective prompts, understand why LLMs sometimes make things up, and discover what it actually means to evaluate whether an LLM is good at something. We'll build intuition around the tradeoffs between cost, speed, and quality, and you'll get practice prompting models yourself.

## 5 concepts to have in mind before class

**1. What is prompt engineering?**

Prompt engineering is the art of writing instructions to an LLM to get the output you want. It's different from fine-tuning because you're not changing the model — you're just finding the right way to ask it. Think of it as learning to ask the right questions rather than teaching the model new skills. You'll spend a lot of time in class experimenting with phrasing, examples, and structure.

**2. Zero-shot vs few-shot prompting**

Zero-shot means you ask an LLM to do something with no examples — just your instruction. Few-shot means you give it one or more examples of the task first, then ask it to do the same for new data. Few-shot often works better because the examples show the model what you mean, but sometimes zero-shot is fine and simpler.

**3. Hallucinations — what they are and why they happen**

A hallucination is when an LLM makes up information that isn't true — it might invent facts, cite fake sources, or confidently state things it shouldn't know. This happens because LLMs predict text one word at a time based on patterns in their training data. They have no way to check if something is real; they just guess the next plausible word. Understanding this helps you know when and why to distrust model outputs.

**4. What does "evaluating an LLM" mean?**

Evaluation means deciding whether an LLM is good at a specific task. It's harder than it sounds because there's often no single "right answer" (especially for creative or open-ended tasks), it's expensive to check every output manually, and different users care about different things. We'll explore different evaluation approaches and when to use each one.

**5. Temperature — what it controls and when you'd change it**

Temperature is a setting that controls how random or predictable an LLM's output is. Low temperature (close to 0) makes the model more focused and repetitive — good when you want consistent, factual answers. High temperature (closer to 1 or 2) makes it more creative and varied — good when you want diverse ideas. You'll adjust this depending on your use case.

## 3 things to read before class (~15 min total)

1. **Prompting Guide Introduction** — https://www.promptingguide.ai/introduction (~5 min reading)
   Read this for a quick overview of basic prompting techniques and why they matter.

2. **OpenAI: Why language models hallucinate** — https://openai.com/index/why-language-models-hallucinate/ (~5 min reading)
   This explains the mechanics behind hallucinations and will help you think critically about LLM outputs.

3. **Artificial Analysis leaderboard** — https://artificialanalysis.ai (~5 min browsing)
   Browse this to see how different LLMs compare on cost, speed, and quality. Notice the tradeoffs — faster models often cost less but are less capable.

## What to bring to class

- **Ollama with `llama3.2` available locally** — install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and start Ollama before class. No API key is required for the core activities.
- **An idea for something you'd like to prompt an LLM with** — this could be anything: summarising a document, generating code, brainstorming ideas, or answering questions. You'll test it during the session.
- **Curiosity about evaluation** — think about how you would measure whether an LLM did a good job at your chosen task. There's often no obvious answer, and that's the interesting part.
