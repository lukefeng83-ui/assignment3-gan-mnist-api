# Assignment 3 Part 3: Theory Questions

## Question 1

**Given an input feature map of size 8 × 8, a transposed convolution with kernel size 4 × 4, stride 2, padding 1, and output padding 1, what will the output size be? Show your calculations.**

The formula for one spatial dimension of a transposed convolution is:

```text
O = (I - 1) × S - 2P + K + OP
```

Where:

```text
O = output size
I = input size
S = stride
P = padding
K = kernel size
OP = output padding
```

Given:

```text
I = 8
S = 2
P = 1
K = 4
OP = 1
```

Substitute into the formula:

```text
O = (8 - 1) × 2 - 2(1) + 4 + 1
O = 7 × 2 - 2 + 4 + 1
O = 14 - 2 + 4 + 1
O = 17
```

Therefore, the output size is:

```text
17 × 17
```

---

## Question 2

**How does the output size of a transposed convolution change if you increase the stride from 2 to 3, keeping everything else fixed?**

The transposed convolution output size formula is:

```text
O = (I - 1) × S - 2P + K + OP
```

The stride `S` is multiplied by `(I - 1)`. Therefore, increasing the stride increases the output size.

Using the same values from Question 1 but changing stride from 2 to 3:

```text
I = 8
S = 3
P = 1
K = 4
OP = 1
```

```text
O = (8 - 1) × 3 - 2(1) + 4 + 1
O = 7 × 3 - 2 + 4 + 1
O = 21 - 2 + 4 + 1
O = 24
```

With stride 2, the output was:

```text
17 × 17
```

With stride 3, the output becomes:

```text
24 × 24
```

So increasing the stride from 2 to 3 increases the output size from 17 to 24 in each spatial dimension.

---

## Question 3

**Write the formula to compute output size for a 2D transposed convolution given input size I, kernel size K, stride S, padding P, and output padding OP.**

For one spatial dimension, the output size formula is:

```text
O = (I - 1) × S - 2P + K + OP
```

For a 2D feature map, apply the formula separately to height and width:

```text
O_h = (I_h - 1) × S_h - 2P_h + K_h + OP_h
O_w = (I_w - 1) × S_w - 2P_w + K_w + OP_w
```

If height and width use the same parameters, then both dimensions have the same output size.

---

## Question 4

**If a Conv2DTranspose layer is used to upsample a feature map from 16 × 16 to 32 × 32, what kernel size and stride could be used, assuming no padding? Give one possible configuration.**

Assuming no padding and no output padding:

```text
P = 0
OP = 0
I = 16
O = 32
```

Formula:

```text
O = (I - 1) × S - 2P + K + OP
```

Because `P = 0` and `OP = 0`:

```text
O = (I - 1) × S + K
```

One possible configuration is:

```text
S = 2
K = 2
```

Calculation:

```text
O = (16 - 1) × 2 + 2
O = 15 × 2 + 2
O = 30 + 2
O = 32
```

Therefore, one valid configuration is:

```text
kernel_size = 2
stride = 2
padding = 0
output_padding = 0
```

---

## Question 5

**Given a mini-batch of 4 values: [6, 8, 10, 6], compute the normalized output using BatchNorm without γ and β.**

BatchNorm without γ and β normalizes values to zero mean and unit variance:

```text
x_normalized = (x - mean) / sqrt(variance + epsilon)
```

For this calculation, ignoring epsilon for simplicity:

```text
x_normalized = (x - mean) / standard deviation
```

Given mini-batch:

```text
[6, 8, 10, 6]
```

### Step 1: Compute the mean

```text
mean = (6 + 8 + 10 + 6) / 4
mean = 30 / 4
mean = 7.5
```

### Step 2: Compute the variance

```text
variance = [(6 - 7.5)^2 + (8 - 7.5)^2 + (10 - 7.5)^2 + (6 - 7.5)^2] / 4
```

```text
variance = [(-1.5)^2 + (0.5)^2 + (2.5)^2 + (-1.5)^2] / 4
variance = [2.25 + 0.25 + 6.25 + 2.25] / 4
variance = 11 / 4
variance = 2.75
```

### Step 3: Compute the standard deviation

```text
standard deviation = sqrt(2.75)
standard deviation ≈ 1.658
```

### Step 4: Normalize each value

For 6:

```text
(6 - 7.5) / 1.658 = -1.5 / 1.658 ≈ -0.905
```

For 8:

```text
(8 - 7.5) / 1.658 = 0.5 / 1.658 ≈ 0.302
```

For 10:

```text
(10 - 7.5) / 1.658 = 2.5 / 1.658 ≈ 1.508
```

For 6:

```text
(6 - 7.5) / 1.658 = -1.5 / 1.658 ≈ -0.905
```

Therefore, the normalized output is approximately:

```text
[-0.905, 0.302, 1.508, -0.905]
```

---

## Question 6

**What is the key mathematical difference between ReLU and LeakyReLU? Give their formulas.**

The key difference is how they handle negative input values.

### ReLU

ReLU sets all negative values to zero.

Formula:

```text
ReLU(x) = max(0, x)
```

This means:

```text
ReLU(x) = x, if x > 0
ReLU(x) = 0, if x <= 0
```

### LeakyReLU

LeakyReLU allows a small non-zero gradient for negative values.

Formula:

```text
LeakyReLU(x) = x, if x > 0
LeakyReLU(x) = αx, if x <= 0
```

Where α is a small positive constant. In this assignment:

```text
α = 0.2
```

So:

```text
LeakyReLU(x) = x, if x > 0
LeakyReLU(x) = 0.2x, if x <= 0
```

---

## Question 7

**Why might LeakyReLU be preferred over ReLU in deep networks?**

LeakyReLU may be preferred because it reduces the risk of the dying ReLU problem.

With ReLU, negative inputs are set to zero. If a neuron keeps receiving negative inputs, its output can become zero and its gradient can also become zero. This means the neuron may stop learning.

LeakyReLU avoids this by allowing a small negative output when the input is negative:

```text
LeakyReLU(x) = αx for x <= 0
```

Because the gradient is not completely zero for negative inputs, the neuron can still update during backpropagation.

This can make training more stable in deep networks, including GAN discriminators.
