# Vector Similarity: A Reference Guide
*Cosine Similarity, Dot Product, and Unit Normalisation*

**Session 8 companion pre-read · approximately 5 minutes.** Read alongside the [APIs and RAG pre-reading guide](pre-reading.md).

---

## 1. Cosine Similarity

Cosine similarity is the cosine of the angle between two non-zero vectors, returning a value between -1 and 1:

$$\cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \times \|\mathbf{B}\|}$$

Expanded:

$$\cos(\theta) = \frac{\sum A_i \times B_i}{\sqrt{\sum A_i^2} \times \sqrt{\sum B_i^2}}$$

| Part | Meaning |
|------|---------|
| Numerator (A · B) | Dot product — multiply corresponding elements and sum them |
| Denominator (‖A‖ × ‖B‖) | Product of each vector's L2 norm (Euclidean length) |

### Range

| Score | Meaning |
|-------|---------|
| 1 | Identical direction |
| 0 | Orthogonal directions |
| -1 | Opposite direction |

> **For text embeddings:** cosine similarity ignores vector magnitude. A suitable embedding model aims to give related texts similar directions, but document length alone does not determine a vector's length or direction. These scores describe geometry, not guaranteed semantic equivalence or the probability that an answer is correct. Evaluate relevance thresholds on your own data.

---

## 2. Dot Product Similarity

Dot product similarity is the numerator of cosine similarity, without normalisation:

$$\text{sim}(\mathbf{A}, \mathbf{B}) = \mathbf{A} \cdot \mathbf{B} = \sum A_i \times B_i$$

The relationship between the two:

$$\text{dot product} = \cos(\theta) \times \|\mathbf{A}\| \times \|\mathbf{B}\|$$

Dot product captures both **direction** (angle) and **magnitude**, whereas cosine similarity captures direction only.

| | Cosine Similarity | Dot Product |
|---|---|---|
| Normalised vectors | Same result | Same result |
| Unnormalised vectors | Ignores magnitude | Affected by magnitude |
| Best use case | Most text embeddings | When magnitude carries meaning |

> **Practical note:** when both vectors have L2 norm 1, their dot product equals their cosine similarity. Check the model's output normalisation and recommended metric. FAISS stores the vectors you provide; it does not automatically normalise them.

---

## 3. Vector Notation: What ‖A‖ Means

The double bar notation `‖A‖` means the **magnitude** (length) of vector A, also called the **norm**:

$$\|\mathbf{A}\| = \sqrt{A_1^2 + A_2^2 + A_3^2 + \ldots + A_n^2}$$

This is simply Pythagoras extended to n dimensions — the length of the hypotenuse given all sides. For example:

$$\mathbf{A} = [3, 4] \quad \Rightarrow \quad \|\mathbf{A}\| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$$

So when you see `‖A‖ = 1`, it means the vector has length exactly 1 — which is the definition of unit normalised.

---

## 4. Unit Normalisation (L2 Normalisation)

### In Plain Terms

Think of every embedding vector as an arrow pointing in some direction in space.

Without normalisation, arrows can have different lengths. That length depends on the embedding model and input; it is not a measure of the number of words.

**Unit normalised** means you take every arrow — no matter how long or short it was originally — and shrink or stretch it so it is exactly length 1. The direction stays exactly the same. You haven't changed what it means or what it's pointing at. You've just made all arrows the same length.

> **Real world analogy:** think of compass directions. It doesn't matter if you drew the arrow 1cm or 10cm on paper — north is still north. Unit normalising is like saying "all arrows are exactly 1cm; we only care about the direction they point."

### The Formula

To L2-normalise a non-zero vector, divide each element by its magnitude:

$$\hat{\mathbf{A}} = \frac{\mathbf{A}}{\|\mathbf{A}\|}$$

Example: `A = [3, 4]`

```
‖A‖ = 5  →  Â = [3/5, 4/5] = [0.6, 0.8]

Verify: √(0.6² + 0.8²) = √(0.36 + 0.64) = √1.0 = 1  ✓
```

In Python:

```python
import numpy as np

a = np.array([3.0, 4.0])
norm = np.linalg.norm(a)
if norm == 0:
    raise ValueError("Cannot normalise a zero vector")
a_normalized = a / norm  # [0.6, 0.8]

print(np.linalg.norm(a_normalized))   # 1.0
```

### L2 Normalisation vs Unit Normalisation

**In this guide, unit normalisation means L2 normalisation.** More generally, a unit vector has norm 1 under a specified norm, so name the norm when the distinction matters.

The "L2" refers to the L2 norm, which is the Euclidean length formula used above. It is called L2 because it squares the elements (power of 2) before summing. There are other norms — L1 (sum of absolute values), L∞ (maximum value) — but L2 is by far the most common in machine learning and embeddings. When someone says "unit normalised" in an ML context, they almost always mean L2 normalised.

| Term | Meaning |
|------|---------|
| Unit normalised | Vector has length exactly 1 |
| L2 normalised | Same thing — normalised using the L2 (Euclidean) norm |
| L2 norm | The √ΣAᵢ² formula for vector length |

### Why Embedding Models Use It

L2 normalisation removes information about vector length so comparisons depend on direction. Use it when that matches the embedding model's training and recommended metric; some models use unnormalised dot product deliberately.

The side benefit is that the maths gets simpler and faster: dot product and cosine similarity become identical, which matters at scale when doing millions of similarity lookups.

---

## 5. OpenAI Embeddings

OpenAI documents its embeddings as L2-normalised. Their dot product can therefore be used for cosine similarity, and Euclidean distance produces the same neighbour ordering. The session uses `text-embedding-3-small`. See the [OpenAI embeddings guide](https://developers.openai.com/api/docs/guides/embeddings).

| Model | Dimensions | Unit Normalised |
|-------|-----------|-----------------|
| text-embedding-3-small | 1536 (configurable) | Yes |
| text-embedding-3-large | 3072 (configurable) | Yes |
| text-embedding-ada-002 | 1536 | Yes |

> **Gotcha:** if you manually truncate dimensions, the resulting shorter vector is no longer unit-normalised and must be re-normalised manually. If you use the API's built-in `dimensions` parameter, re-normalisation is handled automatically.

## 6. How This Connects to FAISS in Session 8

The RAG example uses `faiss.IndexFlatL2`, which reports **squared Euclidean distance**. Smaller distances indicate closer vectors. This is a distance, not a cosine similarity score.

For L2-normalised vectors:

$$\|\hat{\mathbf{A}}-\hat{\mathbf{B}}\|^2=2-2\cos(\theta)$$

For example, cosine similarity 0.8 corresponds to squared L2 distance 0.4. Both metrics rank normalised vectors in the same order, but their numerical values and preferred directions differ. With unnormalised vectors, that equivalence does not hold. See [FAISS metrics and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

**Check your understanding:** if both vectors are unit length and point in the same direction, their dot product is 1 and their squared L2 distance is 0. A zero vector cannot be L2-normalised and has undefined cosine similarity.
