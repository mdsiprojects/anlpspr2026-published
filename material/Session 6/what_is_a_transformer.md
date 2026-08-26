# What is a Transformer
## Transformer (Neural Network Architecture)

<!-- ![Image](https://images.openai.com/static-rsc-3/WDuHb64OVwEt0Dbfie7hNwtoCGvOKHFlgQCPeYn5XL78wA6oR2HOoPJP_2-FVdzTrlBXi7-DRySEQO2LP6p8F9BpETqN0Dl_i9A27o8jZOM?purpose=fullsize\&v=1)

![Image](https://images.openai.com/static-rsc-3/2A-bo_3udV14ftO57tK2oC5w--jHCK1ZPeAUv3s3MF6xwhKozmXGdss2ELz7GNzPXhN9c_CRmMRbEWfxgAXamcXhnFVdLlDCaRy2Nuh7Kjs?purpose=fullsize\&v=1) -->



![Image](https://miro.medium.com/1%2Atb9TT-mwFn1WPzkkbjoMCQ.png)

A **Transformer** is a deep learning architecture designed to process **sequences of data**, especially text. It is widely used in **natural language processing (NLP)** and underpins many modern language models.

The architecture was introduced in the 2017 paper Attention Is All You Need by researchers at Google Brain.

In operational terms, a Transformer enables a model to **analyze relationships between all words in a sequence simultaneously**, rather than sequentially.

---

## Core concept: Attention

The key innovation is **self-attention**.

Self-attention allows each word in a sentence to **look at every other word** and determine which ones are relevant.

Example sentence:

```
The animal didn't cross the street because it was tired.
```

To understand **“it”**, the model attends to **“animal”**.

This mechanism allows the system to capture **long-range dependencies** in text efficiently.

---

## Main components of a Transformer

A typical Transformer contains two major modules:

### 1. Encoder

Processes the input text and produces contextual representations.

Each encoder layer contains:

* **Multi-head self-attention**
* **Feed-forward neural network**
* **Residual connections**
* **Layer normalization**

Models like BERT use **only the encoder stack**.

---

### 2. Decoder

Generates output sequences one token at a time.

Each decoder layer includes:

* **Masked self-attention**
* **Encoder–decoder attention**
* **Feed-forward network**

Models such as GPT use **only the decoder stack**.

---

## Key mechanisms inside a Transformer

![Image](https://miro.medium.com/1%2AEV2BdvxKSUDN1Ii1Pbv3pg.png)

### Multi-Head Attention

Instead of computing attention once, the model performs it **multiple times in parallel**, allowing it to learn different relationships.

### Positional Encoding

Transformers process tokens simultaneously, so they add **position information** to represent word order.

### Feed-Forward Layers

After attention, each token representation passes through a small neural network to refine features.

---

## Why Transformers replaced older models

Before Transformers, sequence models relied on:

* Recurrent Neural Network
* Long Short-Term Memory

These processed text **token by token**, which limited parallelization and struggled with long dependencies.

Transformers provide:

* **Parallel computation**
* **Better long-context understanding**
* **Higher scalability**

---

## Where Transformers are used

They are now the backbone of most modern AI systems:

| Application          | Example              |
| -------------------- | -------------------- |
| Language models      | GPT-4                |
| Search understanding | BERT                 |
| Machine translation  | Google Translate     |
| Code generation      | AI coding assistants |
| Vision models        | Vision Transformers  |
