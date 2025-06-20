---
title: "word2vec from scratch: Intuition to Implementation"
date: 2025-06-16T18:12:56-04:00
draft: false
toc: true
images:
tags:
  - word2vec
  - embeddings
  - nlp
---
Have you ever thought about how we teach machines which words are similar — and which ones aren’t? It's wild to realize that our phones had smart autocomplete features *decades* before modern AI tools like ChatGPT became mainstream. 

<div align="center">
  <img src="/images/word2vec_1.jpg" alt="Skip-gram architecture" width="600"/>
  <p>In an <code>n</code>-dimensional space, words “cling” to their own axes—Word2Vec learns this by pushing true neighbors together and pulling random words apart (contrastive learning).</p>
</div>


Word embeddings laid the foundation for modern NLP models. Before transformer-style architectures took over (with their dynamic embeddings), Word2Vec was the go-to method for capturing the semantic meaning of words through dense (~ 100-300 dim) vectors.

In this post, I'm walking through the **intuitions, architecture, methods, and math behind Word2Vec** — and eventually implementing it from scratch using PyTorch.

---

## 💡 Why Word2Vec?

The core idea behind Word2Vec is that **words that appear in similar contexts should have similar representations** and should appear closer together in higher dimensional vector spaces. This principle — often summed up as *“You shall know a word by the company it keeps”* — forms the foundation of modern embedding models. Instead of manually designing features, Word2Vec lets the model *learn* them by predicting words based on context.

There are two main ways to solve this problem:

- **Skip-gram** — Predict surrounding context words given the current word.
- **CBOW (Continuous Bag of Words)** — Predict the current word given its surrounding context.

In both cases, the model learns to assign each word a dense vector in a high-dimensional space — an **embedding** — such that semantically similar words are close together.

At the heart of the model is one hashmap and two embedding matrices:  
- **LookupTable** - Maps words from your vocabulary to a number between `0` and `len(vocab)-1` -> each one of those numbers represents a row in the embedding matrices

- **E_in** — Used to represent the input word  
- **E_out** — Used to compute predictions over context words

Here's an easy way to visualize a matrix like **E_in** or **E_out**.

Both are of shape *(vocab_size, embedding_dim)*. During training, we take the dot product of the input word’s vector from `E_in` with all rows of `E_out`, yielding similarity scores. A softmax is typically applied to convert these into probabilities, and the model learns by maximizing the probability of true context words. After training, we often discard `E_out` and treat `E_in` as our final word embedding matrix.

TLDR of this section: 
We are learning two embedding matrices, **E_in** and **E_out** which will contain dense vectors that have information about the semantics of our corpus. We also have a couple different ways to learn these matrices (skip gram, CBOW) which approach the problem in different ways.

---

## 🔍 Methods to approach word embeddings

The simplest way in training Word2Vec is just having two matrices *E_in* and *E_out* and initialize them with Xavier initialization. We use **two separate matrices** because the role of a word as a center word (input) is distinct from its role as a context word (output). Keeping them separate allows the model to learn directional context — e.g., “dog” and “bark” might be close in one direction but not the other.

Formally, `E_in, E_out ∈ ℝ^{|V| × d}`, where \|V\| is the vocabulary size and *d* is the embedding dimension. Xavier initialization sets each weight to values drawn from a distribution whose variance keeps activations well-scaled, preventing gradients from exploding or vanishing—even in this mostly linear architecture.

For a quick recap / introduction to Xavier style initialization, I recommend reading this other small [post](https://www.geeksforgeeks.org/deep-learning/xavier-initialization/). 

0) Vanilla Training Method (Warning: this will fry your GPU 😭)

    The the most straightfowrard way to train Word2Vec is to use the full softmax across the whole vocabulary. Here's how is works:

    - For each center word, grab it's embedding vector from *E_in*. Recall that each row in *E_in* is its own embedding.
    - Compute dot products between this vector and **every** row in *E_out*, giving you a tensor of shape (vocab_size,). 
    - Apply a softmax over all vocabulary words to get probabilities of what are close. You have the option to sample top k here or just pick the argmax. 
    - The model learns by maximizing probability of the true context word -> shows us that we must use *cross-entropy loss*.
    - Backpropagate and update weights in both embedding matrices (*E_in* and *E_out*)

    Pretty straightforward right? Here's the catch: 

    Every forward pass, we are doing a softmax on your entire vocabulary - and your vocabulary might be around 50000+ words in real world datasets. You are essentially doing thousands of unnecessary calculations per word pair, burning your compute and making training brutally slow, though the core weight updates are only from a small handful of words that you have.
    
    **Bottom line:**
    This method is great for understanding the core math/backprop and what exactly is going on in Word2Vec but falls short in reality (unless you have time and money to burn). 

1) Skip Gram with Negative Sampling (Industry standard 🔥)

    Essentially we are going to do what we did in the previous example but now we are going to address the bottleneck - the O(V) computation cost per forward pass. But other than that we are going to follow the same general structure. 

    In essance, what SGwNS says is that: 
    "Computing for all 50000+ words in your vocabulary is too expensive. Let's just compare the center word to 1 good word (ground truth word) and take a random sample of "bad" words to push them away."

    So you go from softmaxxing 50k values -> 1 positive dot product (your center word <-> the word you want (ground truth)) & 5-20 negative dot products (your center word <-> unrelated words)

    We can set `k` = `1 (positive dot product word)` + `5-20 (negative dot product words)`. So now we pay O(k) instead of O(V) - huge gains. (k < < V)

    And this is exactly how we beat the bottleneck: contrastive learning. 

    **Bottom line:**
    We are using a smart trick to push away random words that are not meaningful to our current word, hopefully resulting in results similar to the vanilla method. (Spoiler: this trick does work really well and that's why it is the industry standard) 

2) Continuous Bag of Words (also cool but slightly lower performance than SGwNS 🙉)

    This essentially works in the opposite direction as Skip Gram: 

    - Instead of predicting cnontext from the center word, you predict the center word from it's surrounding context words (kind of like what modern day LLMs do, but on 2015 technology)
    - You take all the context word embeddings, average (or sum) them, and feed the result into the model.
    - The model then tries to guess the center word using either of the methods we talked above:
      - full softmax (like the vanilla version)
      - negative sampling (like SGwNS)

    **Bottom line:**
    Since we are comibing multiple context words into one average (or sum), CBOW trains faster than Skip-Gram and is more stable for smaller datasets. However, the main tradeoff is the fine-grained semantic realtionships because of that averaging. Most of the times, this method performs slightly worse than SGwNS and hence insn't that big in practice.

Next, let's talk about loss functions for each method.

---

## 🧠 Training Objective / Criterion

Let's split up the training objective by section (0, 1, or 2) and then we can ask two more sub-questions within each section:

- What should our loss function look like to actually pull together words that are similar (or push away words that are not similar)?
- What is going to get backpropagated once we actually have a scalar loss value?

--

0) Vanilla Method

Here, we want the model to assign **high probability to the correct context word**. We don't care about other words right now - we just want:
- Low loss when the predicted context word is the *actual* neighbor
- High loss when the predicition is far off (e.g., `coffee` showing up near `bird`)

Since this is a **multi-class classification** task (predict 1 word out of vocab_size), the natural loss to use is **cross-entropy**, applied on top of a **softmax**.


🧮 The Goal:

> Maximize the probability of the true context word $w_o$, given the center word $w_c$:
> $$
> P(w_o \mid w_c) = \frac{e^{u_o^\top v_c}}{\sum_{w=1}^{V} e^{u_w^\top v_c}}
> $$

Math symbols scare me too so I'll break down what that it saying. The numerator is how similar the correct word is to the center word. The denominator is how similar every word in the vocab is to the center word. So if the correct context word is super close in high dimensional space, we have a larger numerator (good!). If a bunch of random words are also closeby our denominator increases (bad!).

We want to maximize this probability so that means making the true word similarity as big as possible while keeping all other dot products in the denominator small. Since this is a probability, *cross-entropy loss* is the star. 

It's defined as:

$$
L = -\log\ \Bigl(
  \frac{\exp(u_o^T v_c)}
       {\sum_{w=1}^{V} \exp(u_w^T v_c)}
\Bigr)
$$

This basically measures "how surprised are we by the correct answer?", exactly what we should be asking when we predict a good/bad word. We now have a scalar loss that we can backpropagate through our network (loss.backward())

During backpropagation, only the embeddings of the center word and the true context word actually get updated. All other words are just there to normalize the softmax, they aren't contributing anything else. This is why is method is so inefficient - we are doing a HUGE amount of computation just to update 2 weights (crunching 50k numbers to update 2 weights).

Let's fix this **O(V) nightmare** in the next section with negative sampling. However, we would need to change our training objective now. 

--

1) Skip Gram with Negative Sampling

Let's address this bottleneck now. Instead of predicting one word out of 50k, we can turn it into a binary classification task.

> Is this a *true context word* -> label = 1

> Is this a *random irrelevant word* -> label = 0

For every center word, we train on: 
- One positive pair (center, actual context word)
- `k` negative pairs (center, randomly sampled "noise" words)

And we are going to use the sigmoid function here instead of softmax so that each pair is treated independently. (This used to confuse me a lot too at first, but I just don't have the space to write about it more in this post ☹️).

**Probability Objective:**

We want to maximize the probability that a real word pair appears together:

$$
P(\text{real} \mid w_c, w_o) = \sigma(u_o^\top v_c)
$$

And for a negative (fake) sample, we want:

$$
P(\text{fake} \mid w_c, w_i) = \sigma(-u_i^\top v_c)
$$

**Loss Function:** 

$$
L = -\log(\sigma(u_o^\top v_c)) - \sum_{i=1}^{k} \log(\sigma(-u_i^\top v_c))
$$


This is a really cool loss function because it has two jobs. One for pulling the true word closer together and one for pushing the fake words away. We can see that the first term is calculating the negative log of the similarity between the center word and the context word. This gives us a loss that is lower when they are highly similar - exactly what we wanted. 

The second term is where we pull in embeddings for k negative sampled words that are not in the context. We want their similarity with the center word to be low - that's why we negate the dot product, apply sigmoid (which favors small similarity), and penalize high values. In a nutshell this pushes away random/fake context words, minimizing the chance that the model falsely thinks they belong together.

These two terms combined teach the model "pull what matters, push what doesn't".

- $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function  
- $u_o$ is the embedding of the true context word (positive)  
- $u_i$ are the embeddings of $k$ negative words  

--

2) Continous Bag of Words (CBOW)

You probably get the gist of what we do now: understand our probability objective and based on that, make a loss objective.

Here, we have to be ready to flip our heads in the opposite way. We are going to predict the center word now, given a set of context words.

The probability objective:

$$
P(w_c \mid w_{c - m}, \dots, w_{c + m}) = \frac{e^{u_c^\top \bar{v}}}{\sum_{w=1}^{V} e^{u_w^\top \bar{v}}}
$$

Our loss function (with negative sampling) is:

$$
L = -\log(\sigma(u_c^\top \bar{v})) - \sum_{i=1}^{k} \log(\sigma(-u_i^\top \bar{v}))
$$

-	$\sigma(x) = \frac{1}{1 + e^{-x}}$ — the sigmoid function, which squashes scores into a range between 0 and 1. We use it to convert similarity scores into probabilities.
-	$u_c$ — the output embedding of the center word (the word we’re trying to predict). Think of this like a “target” vector.
-	$\bar{v}$ — the average of the input (context) embeddings surrounding the center word. This is what we feed in to predict the center.
-	$u_i$ — the output embeddings of $k$ negative samples (randomly chosen words that are not the true center). We want the model to push these words away from the context representation.

---

## 🛠️ Coding it out
I'll be putting in snippets here but to view the entire repository, please go over to [github]().

*coming tmr*

---

## 🦾 Tradeoffs Recap
I'll be putting in snippets here but to view the entire repository, please go over to [github]().

*coming tmr*

---

## ⚙️ TL;DR / Summary

Word2Vec may be old school now, but it's still one of the most elegant and interpretable ideas in the NLP world. It's also a great way to build up your ML intuition, understand the baby steps towards contrastive learning, and get comfortable with training loops.

If you enjoyed reading this feel free to star this [repo]() on github 🙂