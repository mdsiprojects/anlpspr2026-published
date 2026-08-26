# What is a BERT model

## BERT (Bidirectional Encoder Representations from Transformers)

![Image](https://heidloff.net/assets/img/2023/02/transformers.png)

<!-- ![Image](https://www.researchgate.net/publication/366996120/figure/fig1/AS%3A11431281112266680%401673371302310/BERT-Bidirectional-Encoder-Representations-from-Transformers-architecture-38-the.ppm)



![Image](https://www.scaler.com/topics/images/next-sentence-prediction-example.webp) -->

**BERT** is a deep learning model used in **natural language processing (NLP)**. It was introduced by Google Research in 2018 and significantly improved how machines understand human language.

### Core idea

Traditional language models read text **left-to-right** or **right-to-left**.
BERT reads **both directions simultaneously**.

This **bidirectional context** allows the model to interpret the meaning of a word based on the **entire sentence**, not just the preceding words.

Example:

Sentence:

> *“The bank raised interest rates.”*

BERT evaluates *bank* using context from both sides, helping distinguish meanings like **financial bank** vs **river bank**.

---

## Architecture (high level)

BERT is built on the **encoder portion of the Transformer architecture** described in Attention Is All You Need.

Key components:

1. **Token embeddings**

   * Converts words/subwords into vectors.

2. **Transformer encoder layers**

   * Multiple stacked layers (12 or 24 typically).
   * Each layer includes:

     * **Self-attention**
     * **Feed-forward neural networks**

3. **Self-attention mechanism**

   * Allows each word to attend to every other word in the sentence.

Typical versions:

* **BERT Base**

  * 12 layers, 110M parameters
* **BERT Large**

  * 24 layers, 340M parameters

---

## Pre-training objectives

BERT learns language using two tasks.
![Image](https://www.researchgate.net/publication/371908062/figure/fig2/AS%3A11431281188006278%401694484015063/Masked-language-model-and-next-sentence-prediction.png)

### 1. Masked Language Modeling (MLM)

Random words are hidden.

Example:

```
The cat sat on the [MASK].
```

The model predicts the masked word (**mat**).

This forces learning **deep contextual relationships**.

---

### 2. Next Sentence Prediction (NSP)

The model learns whether one sentence logically follows another.

Example:

```
Sentence A: I went to the store.
Sentence B: I bought milk.
```

Label: **IsNext**

---

## What BERT is used for

After pretraining, BERT is **fine-tuned** for specific tasks.

Common applications:

| Task                     | Example                                      |
| ------------------------ | -------------------------------------------- |
| Sentiment analysis       | Classifying positive vs negative reviews     |
| Question answering       | Extracting answers from documents            |
| Named entity recognition | Identifying people, locations, organizations |
| Text classification      | Spam detection, topic classification         |
| Search ranking           | Improving search results                     |

BERT powers many systems in products from Google including **search understanding**.

---

## Why BERT was important

Strategically, BERT shifted NLP from **task-specific architectures** to **pretrained language models** that can be reused.

Key benefits:

* Strong contextual understanding
* Transfer learning for NLP
* Significant accuracy improvements across benchmarks

It influenced later models like:

* RoBERTa
* ALBERT
* DistilBERT

---


BERT is a **Transformer-based language model that understands text bidirectionally and can be fine-tuned for many NLP tasks.**

---
