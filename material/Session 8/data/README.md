# Session 8 Dataset

This folder contains the small PDF corpus used in Session 8 to teach retrieval-augmented generation (RAG).

## Why These Documents

The selected documents are generic handbook and benefits PDFs. They were chosen because they support realistic retrieval questions such as:

- policy lookups
- benefits coverage questions
- comparison across plans
- cases where the model should admit that the answer is not in the documents

These are better for Session 8 than course notes because they create clearer retrieval scenarios.

## Structure

```text
material/Session 8/data/
├── README.md
├── questions.json
└── pdfs/
```

## Teaching Purpose

Students use these documents to learn:

- why PDFs need preprocessing before LLM use
- how text extraction affects downstream quality
- how chunking changes retrieval behavior
- how embeddings and FAISS support a minimal RAG pipeline

## Evaluation questions

`questions.json` contains seven questions, expected answer signals, and source-document hints. These guide manual evaluation; they are not a complete gold-answer dataset. Include questions with missing information when checking whether the app avoids unsupported answers.

`images/dogs.jpg` is the local-image example used in notebook 01.
