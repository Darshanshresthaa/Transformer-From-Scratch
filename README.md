#  Mini GPT (Character-Level Transformer) — From Scratch

## 📌 Project Overview

This project is a **character-level GPT (Generative Pretrained Transformer)** built from scratch using PyTorch.
It learns patterns from raw text data and generates new text one character at a time.

The model implements core Transformer concepts including:

* Self-Attention
* Multi-Head Attention
* Feedforward Networks
* Residual Connections
* Layer Normalization

---

##  Features

* Character-level text generation
* Custom tokenizer (no external libraries)
* Full Transformer decoder architecture
* Autoregressive text generation
* Training from raw `.txt` dataset

---



##  Requirements

Install dependencies:

```bash
pip install torch
```

---

##  Dataset

* Input file: `input.txt`
* Can be any text (e.g., books, articles, Shakespeare, etc.)
* Model learns **character-level patterns**

---

##  Data Processing

### 1. Text Normalization

```python
text = text.lower()
```

### 2. Vocabulary Creation

```python
char = sorted(list(set(text)))
```

### 3. Encoding & Decoding

* `char_to_int`: character → integer
* `int_to_char`: integer → character

```python
encoder("hello") → [1, 5, 7, 7, 9]
decoder([...]) → "hello"
```

---

##  Model Architecture

###  Hyperparameters

```python
embedding_dim = 256
block_size = 64
n_layer = 5
n_head = 8
batch_size = 12
epochs = 10000
```

---

###  1. Self-Attention Head

* Computes Query, Key, Value
* Applies scaled dot-product attention
* Uses masking to prevent future token access

```python
score = (Q @ K) / sqrt(d)
```

---

###  2. Multi-Head Attention

* Multiple attention heads run in parallel
* Outputs are concatenated and projected

---

###  3. Feed Forward Network

```python
Linear → GELU → Linear
```

* Adds non-linearity and deeper representation learning

---

###  4. Decoder Block

Each block contains:

* LayerNorm + Multi-Head Attention + Residual
* LayerNorm + FeedForward + Residual

---

###  5. Full GPT Model

Components:

* Token Embedding
* Positional Embedding
* Stacked Decoder Blocks
* Final LayerNorm
* Output Linear Layer (logits)

---

##  Training

### Training Loop

```python
for epoch in range(epochs):
    x, y = data_prepare()
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Loss Function

* Cross Entropy Loss

### Optimizer

* AdamOptimizer 



##  Data Preparation

```python
data_prepare()
```

* Randomly samples sequences
* Input (`x`) = sequence of characters
* Target (`y`) = next character sequence

Example:

```
x: "hello worl"
y: "ello world"
```

---

## Text Generation

```python
model.generate(context, max_new_tokens=200)
```

Steps:

1. Take initial prompt
2. Predict next character
3. Append prediction
4. Repeat

---

##  Training Behavior

Typical loss progression:

```
Start: ~3.8 (random)
Mid:   ~2.0 (learning patterns)
End:   ~1.3–1.5 (good learning stage)
```

---

##  Output Example

```
Input:  "the king"
Output: "the kinghase! if this breast of blinn..."
```

* Model learns structure and style
* Still improving coherence with more training

---

##  Limitations

* Character-level → slower learning than word-level
* Small dataset → limited vocabulary understanding
* No dropout → possible instability
* Basic sampling → noisy output

---

##  Possible Improvements

* Add Dropout (regularization)
* Increase batch size
* Train for more epochs
* Use temperature sampling
* Add top-k / top-p sampling
* Use larger dataset
* Save & load checkpoints

---

##  Saving Model

```python
torch.save(model.state_dict(), "model.pt")
```

Load later:

```python
model.load_state_dict(torch.load("model.pt"))
```

---

##  Learning Outcomes

By building this project, you understand:

* Transformer architecture internals
* Self-attention mechanism
* GPT-style autoregressive generation
* Training deep learning models from scratch

---

##  Conclusion

This project is a **complete minimal GPT implementation** that demonstrates how modern language models work at a fundamental level.

It serves as a strong foundation for:

* NLP research
* LLM development
* Advanced AI systems

---

##  Author

Darshan Shrestha

---

##  If you found this useful

