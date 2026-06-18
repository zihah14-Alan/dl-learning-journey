# Lecture 01 — Overview

> Course: *PyTorch Tutorial* by Hongpu Liu (河北工业大学, SLAM Research Group)
> My notes for Lecture 1. This lecture is pure background — no coding yet.
> Goal: understand **what** machine learning / deep learning is and **where** PyTorch fits.

---

## 1. From human intelligence to machine learning

**Human intelligence**, at its core, is: *receive external information → infer / reason → produce a decision.*
The lecture's everyday examples: deciding "what to eat for dinner" or "what to wear" — the brain takes in information and infers an answer.

**Prediction** is the process of turning raw perceptual information into an **abstract concept**. Two examples:
- Seeing a photo of a cat → mapping it to the abstract concept "cat".
- Seeing a handwritten digit → mapping it to the abstract number it represents.

**Machine learning** replaces the *brain* that used to do this inference with an **algorithm**: we let an algorithm perform the reasoning instead of a human.

---

## 2. Supervised learning

The most common paradigm in machine learning is **supervised learning**.

Definition: we take a **labeled dataset** (a set of data where we already know the correct answer / label for each item), build a **model**, and **train** that model on this data. The trained model becomes our algorithm.

So the three key words for this lecture:
- **Model** — the function we are trying to fit.
- **Training** — the process of using labeled data to adjust the model.
- **Supervised** — "supervised" because every training example carries its known answer (label).

---

## 3. ML vs. classic algorithm design — a different way of thinking

In a traditional **algorithms course**, a human designs the computational procedure by hand, using one of a few classic strategies:

| Strategy | Idea | Example |
|---|---|---|
| Exhaustive search (穷举法) | List all possibilities, check which satisfies the condition | — |
| Greedy (贪心法) | At each step, pick the locally best choice | **Gradient descent uses this idea** |
| Divide and conquer (分治法) | Split the problem in two, solve recursively | Quicksort |
| Dynamic programming (动态规划) | Remember solved sub-problems to avoid recomputation | — |

> Note to self: the lecturer points out that **gradient descent is essentially a greedy strategy** — at each step it moves in the locally best (steepest-descent) direction. This connects directly to the next lecture.

The key difference: in machine learning, the algorithm is **not hand-designed**. Instead, it is **discovered from the dataset** — we let the data tell us what the mapping should be.

---

## 4. The AI → ML → Representation learning → Deep learning hierarchy

These are **nested** subsets (from the *Deep Learning* book by Goodfellow, Bengio, Courville):

```
AI  ⊃  Machine Learning  ⊃  Representation Learning  ⊃  Deep Learning
```

- **AI** — the broadest field (e.g. knowledge bases, rule-based systems).
- **Machine learning** — e.g. logistic regression.
- **Representation learning** — learning the features themselves (e.g. shallow autoencoders).
- **Deep learning** — e.g. MLPs / multi-layer neural networks.

So deep learning is actually a **very small subset** of the whole AI field.

### What is representation learning?
"Representation" here means **feature extraction**. Raw data can be huge, with each sample carrying a great many features. Representation learning tries to find a **better way to represent** the information in a data sample — typically by extracting a compact feature vector from complex, unstructured input.

In representation learning, **feature extraction is a separate, standalone step**: an algorithm turns complex unstructured input into a vector, and that vector is then fed into a "mapping from features" stage.

### The curse of dimensionality (维度诅咒)
More features → more dimensions → **exponentially more data needed** to cover the space. This is mostly an **engineering / cost problem**: higher dimension means a larger data requirement, which means more money.

So we want to **reduce dimensionality**.

Example: an N-dimensional vector (N×1) represents one sample point in N-dimensional space. To map it into 3-dimensional space (3×1), we need a **3×N matrix** that projects N dimensions down to 3, *while preserving the important metric/structural information* of the original high-dimensional space. This projection from high to low dimension is the essence of representation learning.

### Manifold (流形)
Analogy from the lecture: the universe is 3-D, but a galaxy is roughly a flat plane. If the data points in a high-dimensional space actually lie on a relatively **smooth, differentiable surface** (a manifold), then we can map that surface down to a lower-dimensional space without losing much. That low-dimensional representation of an intrinsically lower-dimensional surface embedded in high-dimensional space is exactly what representation learning aims for — **dimensionality reduction**.

---

## 5. How to develop a learning system — four generations

The lecture builds this up layer by layer (PPT pages 14–18). Each row adds more "learned" (vs. hand-designed) components — shaded/learned parts grow from right to left:

| Approach | Input → ... → Output | What's hand-designed |
|---|---|---|
| **Rule-based systems** | Input → **hand-designed program** → Output | Everything (the whole program) |
| **Classic machine learning** | Input → **hand-designed features** → mapping → Output | The features |
| **Representation learning** | Input → **features (learned)** → mapping → Output | Nothing (features are learned), but features & mapping trained separately |
| **Deep learning** | Input → **simple features → additional layers of more abstract features** → mapping → Output | Nothing — fully learned, end-to-end |

### Deep learning input = raw / simple features
In deep learning we feed in **raw features**:
- An image → convert pixel values into a tensor, feed it in.
- Audio → feed in the waveform sequence.
- A row from a relational table → feed in the whole record.

(Some preprocessing is still needed — e.g. discrete values may need basic transformation/encoding.) Then we stack **additional layers** to extract features, feed those into the learner, and output. The "learner" is essentially a **multi-layer neural network**.

### Key difference from traditional representation learning → End-to-end
- **Traditional**: feature extraction and the learner are trained **separately**. Features have no labels (so feature learning is *unsupervised*), while the learner uses labels (*supervised*) — hence they're split.
- **Deep learning**: the whole pipeline is trained **as one unified process**. This is called **end-to-end** training.

---

## 6. Why deep learning took off — the new challenge

Limits that pushed the field toward deep learning (PPT page 21):
- Limit of **hand-designed features** (you can only engineer so much by hand).
- **SVMs do not scale** well to big datasets.
- More and more applications need to handle **unstructured data** (images, audio, text).

The ImageNet (ILSVRC) error-rate chart tells the story: as networks went **deeper**, top error dropped dramatically — from ~28% (shallow, 2010) down to **3.57% with 152-layer ResNet (2015)**. Depth won.

---

## 7. Brief history of neural networks

From neuroscience → mathematics & engineering:
- **Hubel & Wiesel (1959)** — studied receptive fields in the cat's visual cortex; biological inspiration for how vision processes edges/orientations.
- **Single neural unit → Perceptron** — a neuron takes inputs `x_i`, weights them `w_i`, sums, and applies a function: `y = f(Σ w_i · x_i)`.
- **Connected neural units → Artificial Neural Network** — stack and connect many perceptrons into layers to form an ANN.

### Milestone CNN architectures
```
[1998] LeNet-5  →  [2012] AlexNet  →  [2014] GoogLeNet & VGG  →  [2015] ResNet
```
(LeNet-5, LeCun 1998: the classic CNN for digit recognition — Input 32×32 → conv/subsample stages → fully connected → 10 outputs.)

---

## 8. ⭐ Back Propagation — the most important algorithm

> This is the heart of the lecture. BP is, in essence, **computing derivatives** (gradients) efficiently.

### Why not just write the analytic derivative?
As the network gets deeper, writing one closed-form analytic expression for the derivative becomes **intractable**. So instead we use a **computational graph**.

### Computational graph + atomic operations
In the computational graph we only perform **atomic operations** — the simplest basic operations that can't be sensibly broken down further (e.g. a single `+` or `*`). Each node is one such operation.

### Forward pass
The **forward pass** feeds input data along the arrows, computing step by step until we reach the final output. During the forward pass we can *also* compute each **local partial derivative** at every node.

### Worked example (PPT pages 28–30)
Graph:
```
        e = c * d
       /         \
   c = a + b     d = b + 1
     /    \         |
    a      b -------+
```
Given inputs `a = 2`, `b = 1`:

**Forward pass (compute values):**
- `c = a + b = 2 + 1 = 3`
- `d = b + 1 = 1 + 1 = 2`
- `e = c * d = 3 * 2 = 6`

**Local partial derivatives at each node:**
- `∂c/∂a = 1`,  `∂c/∂b = 1`   (since `c = a + b`)
- `∂d/∂b = 1`                  (since `d = b + 1`)
- `∂e/∂c = d = 2`,  `∂e/∂d = c = 3`   (since `e = c * d`)

**Backward pass (chain rule — multiply along paths, sum over multiple paths):**

`∂e/∂a` — only one path (a → c → e):
$$\frac{\partial e}{\partial a} = \frac{\partial e}{\partial c}\cdot\frac{\partial c}{\partial a} = 2 \times 1 = 2$$

`∂e/∂b` — **two paths** (b → c → e *and* b → d → e), so we **sum** them:
$$\frac{\partial e}{\partial b} = \frac{\partial e}{\partial c}\cdot\frac{\partial c}{\partial b} + \frac{\partial e}{\partial d}\cdot\frac{\partial d}{\partial b} = (2 \times 1) + (3 \times 1) = 5$$

### The key rules to remember
1. **Multiply** partial derivatives **along** a path.
2. When a variable reaches the output through **multiple paths**, **sum** the contributions of all those paths.
3. The forward pass computes values *and* local gradients; the backward pass combines them via the chain rule.

> This is exactly what PyTorch's autograd does automatically — it builds the computational graph during the forward pass and applies these chain-rule rules in the backward pass. (Connects to `loss.backward()` later in the course.)

---

## 9. Good news — deep learning is approachable

(PPT page 34)
- Deep learning is **not too difficult**: basic **algebra + probability + Python** is enough.
- Can get going in **less than a year** of study.
- You **don't have to start from scratch** — frameworks provide:
  - efficient, convenient **GPU** usage,
  - many ready-made neural-network **components** (building blocks).
- Popular frameworks: Theano, TensorFlow (Google), Caffe/Caffe2, Torch/**PyTorch** (Facebook).

> Takeaway from the lecturer: when building deep networks, the important skill is **knowing how to use the basic building blocks, then assembling them for your task** — not reinventing everything.

---

## 10. What is PyTorch / why PyTorch

**What:** PyTorch is a Python package with two high-level features:
1. **Tensor computation** (like NumPy) but with strong **GPU acceleration**.
2. **Deep neural networks** built on a **tape-based autograd** system.

**Why:**
- **Dynamic graph** — the graph is created on the fly → more flexible, easy to debug, intuitive/cleaner code.
- **More "neural-networkic"** — you write code the way the network works; AutoGrad handles forward/backward automatically.

> Version note: this course uses **PyTorch 0.4 (2018)**; I'm on **2.11**. Core APIs (tensors, autograd, training loop, `nn.Module`) are essentially unchanged, so the code still runs. A few old constructs are deprecated (e.g. `Variable` — just use tensors directly now). The lecturer's own advice applies: **versions move fast, so read the docs** — that habit matters more than memorizing one version's syntax.