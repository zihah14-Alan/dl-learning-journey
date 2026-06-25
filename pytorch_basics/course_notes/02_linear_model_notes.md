# Lecture 02 — Linear Model

> 刘二大人《PyTorch 深度学习实践》 · Lecture 2
> Building intuition for **what a model is**, **what "loss" means**, and how to find the best weight by **brute-force search** — the setup for gradient descent in Lecture 3.

---

## 1. Problem setup

The running example: a student who studies `x` hours scores `y` points on the final exam.

| x (hours) | y (points) |
|-----------|------------|
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | **?** |

The question: **what score should we predict for 4 hours of study?**

We can't answer this by lookup — `x = 4` isn't in the data. We need a *model* that captures the relationship and generalizes to unseen inputs.

### The supervised-learning pipeline

A standard machine-learning workflow has four stages:

1. **Prepare the dataset** — collect `(x, y)` pairs.
2. **Choose / design the model** — pick a function family (here, a linear model).
3. **Training** — find the parameters that best fit the data.
4. **Inference** — use the trained model to predict on new inputs (e.g. `x = 4`).

The rows with known answers (`x = 1, 2, 3`) form the **training set**; the row we want to predict (`x = 4`) is the **test set**. Because every training example comes paired with its correct label `y`, this is **supervised learning**.

---

## 2. Data and generalization

A few realities about data that matter even before we touch a model:

- **No dataset captures the true distribution exactly.** Even a large dataset is only a finite sample, so it never perfectly represents the real-world distribution it was drawn from.
- **Realistic data is often hard to obtain.**

**Cat-vs-dog example.** Suppose we train an image classifier to tell cats from dogs, using photos collected from the web. Those photos tend to be clean, well-lit, nicely posed — even "beautified." Once the model is deployed and real users upload casual photos (odd angles, bad lighting, partial views), the model can fail, because the deployment data doesn't look like the training data.

### Overfitting

**Overfitting** = the model performs well on the training set but poorly on the test set. This usually means the model has memorized the **noise** in the training data — random quirks specific to those particular samples — instead of the underlying pattern. Noise is exactly what we *don't* want it to learn.

### Generalization

What we actually want is **generalization**: the ability to make correct predictions on inputs the model has never seen.

To estimate generalization *before* deployment, we split the labeled data into two parts:

- a **training set** — used to fit the parameters, and
- a **development / validation set** (dev set) — held out and used only to evaluate how well the model generalizes.

The dev set is **not** used to update the parameters; it acts as a stand-in for "unseen" data.

---

## 3. Model design

> Goal: find the most suitable model for the data. We start with the simplest reasonable choice and can swap in something more complex later.

The data looks roughly linear, so we start with a **linear model**:

$$\hat{y} = x \cdot \omega + b$$

To keep this first example as simple as possible, the lecture drops the bias term `b`:

$$\hat{y} = x \cdot \omega$$

Notation:

- The **hat** on ŷ marks it as a *prediction* (the model's output), distinguishing it from the true label `y` in the data.
- ω (omega) is the **weight** — the single parameter we need to learn.

---

## 4. Finding the best weight

How do we pick a good ω?

1. **Start with a random guess** for ω.
2. **Measure how far off the model is.** For each sample, the error is the gap between prediction and truth:

$$\hat{y}^{(1)} - y^{(1)}, \quad \hat{y}^{(2)} - y^{(2)}, \quad \hat{y}^{(3)} - y^{(3)}$$

3. We want these gaps to be **small**. If the model fit the data perfectly, every gap would be `0`.

To turn "the gaps are small" into a single number we can minimize, we **square** each gap (so positive and negative errors don't cancel, and large errors are penalized more). That gives us the **loss**.

> Both the **sum of squared errors** and the **sum of absolute errors** are reasonable ways to measure error. This lecture uses squared error, which leads to **MSE** (Mean Square Error); absolute error would lead to **MAE**. Squared error is the standard choice for linear regression because it is smooth and differentiable everywhere — which matters once we start using gradients (Lecture 3).

### Loss vs. Cost

These two terms are easy to confuse — keep them straight:

- **Loss** — the error on a **single sample**:

$$\text{loss} = (\hat{y} - y)^2 = (x \cdot \omega - y)^2$$

- **Cost** — the **average loss over the whole training set** (this average squared error is the MSE):

$$\text{cost} = \frac{1}{N} \sum_{n=1}^{N} \left( \hat{y}_n - y_n \right)^2$$

The ideal case is **zero loss / zero cost** — a perfect fit. In practice we just try to make the cost as small as possible.

### Worked example

**Computing the per-sample loss for ω = 3:**

| x | y | ŷ = 3x | loss = (ŷ − y)² |
|---|---|--------|------------------|
| 1 | 2 | 3 | 1 |
| 2 | 4 | 6 | 4 |
| 3 | 6 | 9 | 9 |

MSE = (1 + 4 + 9) / 3 = 14/3 ≈ **4.7**

**Cost (MSE) across candidate weights:**

| ω | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **MSE** | 18.7 | 4.7 | **0** | 4.7 | 18.7 |

The cost is lowest at **ω = 2**, which matches the obvious pattern in the data (`y = 2x`).

---

## 5. Brute-force search (exhaustive search)

How do we actually search over weights? The simplest method is **brute force**: try many candidate values and keep the one with the lowest cost.

The recipe:

1. Pick a range that probably contains the best weight — here, testing suggests the minimum lies somewhere in `[0, 4]`.
2. Step through that range (`0.0, 0.1, 0.2, …, 4.0`) and compute the cost at each value.
3. Plot cost against ω.

The result is a **bowl-shaped (parabolic) curve**, and its lowest point is the best weight. For this problem the curve bottoms out at ω = 2.

> **Why brute force doesn't scale.** With one parameter, sweeping a 1-D range is trivial. With two parameters (ω and `b`) we sweep a 2-D grid — still manageable. But a real neural network has *millions* of parameters, and the number of grid points grows **exponentially** with the number of parameters (the *curse of dimensionality*). Brute force becomes hopeless almost immediately. This is exactly the problem **gradient descent** (Lecture 3) solves: instead of checking every point, it follows the slope of the bowl downhill to the minimum.

---

## 6. Code — plotting the cost curve

> This is the lecture's own code for the single-parameter model `ŷ = x · ω`. No PyTorch yet — just NumPy and matplotlib.

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 2. Define the model: y_hat = x * w
def forward(x):
    return x * w          # `w` is read from the global scope (set by the loop below)

# 3. Define the loss for a single sample: (y_hat - y)^2
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []      # candidate weights
mse_list = []    # cost (MSE) for each weight

# 4. Brute-force sweep over w in [0.0, 4.0], step 0.1
for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum / 3)
    w_list.append(w)
    mse_list.append(l_sum / 3)     # cost = mean of the per-sample losses

# 5. Plot cost vs. weight
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
```

A few things worth noticing:

- **`forward()` and `loss()` read `w` as a global.** `w` is never passed in as an argument — the functions pick it up from the loop variable in the outer (global) scope. This is fine for a tiny script, but it is generally fragile; later lectures move parameters inside the model object instead.
- **Why `np.arange(0.0, 4.1, 0.1)` and not `4.0`?** `np.arange` *excludes* the stop value, so stopping at `4.1` is what actually includes `4.0`.
- **Floating-point quirk.** With a float step you'll see values like `w= 0.30000000000000004` in the printout — ordinary floating-point rounding, harmless here.
- **Cost is the sum of per-sample losses divided by N.** `l_sum` accumulates loss over the three samples; dividing by `3` turns it into the mean (MSE).

---

## 7. Exercise

Extend the model to include the bias term:

$$\hat{y} = x \cdot \omega + b$$

Now there are **two** parameters, so brute force means a **double loop** over ω and `b`. Plotting cost against two parameters gives a 3-D **surface** (a bowl in 3-D) instead of a 2-D curve.

Tips from the lecture:

- Read up on how to draw a 3-D graph in matplotlib.
- `np.meshgrid()` is the standard tool for building the 2-D grid of `(w, b)` values; combine it with **vectorized** NumPy operations — compute the whole grid at once instead of looping element by element.

*(This is the assignment to implement in `02_linear_model.ipynb`.)*

---

## Key takeaways

- A **model** is a parameterized function (`ŷ = x · ω`); **training** = finding the parameters that minimize a **cost**.
- **Loss** is per-sample; **cost** (MSE) is the average loss over the dataset.
- We want **low cost** *and* good **generalization** — not just memorizing the training set.
- **Brute-force search** works for 1–2 parameters but collapses under the curse of dimensionality → this motivates **gradient descent** (Lecture 3).
