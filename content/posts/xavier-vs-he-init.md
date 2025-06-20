---
title: "Xavier vs He Init"
date: 2025-06-20T11:01:04-04:00
draft: false
toc: false
images:
tags:
  - untagged
---

## 🧠 The Problem

When training neural nets, bad weight initialization leads to exploding/vanishing gradients.

## 📐 Xavier Initialization

- Designed for tanh / sigmoid
- Keeps variance consistent across layers

**Formula:**

$$ W \sim \mathcal{U}\left(-\frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in} + n_{out}}} \right) $$

## ⚡ He Initialization

- Designed for ReLU
- Keeps forward activation variance high enough

**Formula:**

$$ W \sim \mathcal{N}\left(0, \frac{2}{n_{in}} \right) $$

## 🧪 PyTorch Example

```python
import torch.nn as nn

# Xavier
nn.Linear(256, 128)
nn.init.xavier_uniform_(layer.weight)

# He
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
```