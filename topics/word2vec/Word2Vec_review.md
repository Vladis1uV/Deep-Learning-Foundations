# Word2Vec

### Task

The main task that Word2Vec models are designed to solve is learning vector representations of words such that words with similar meanings or contexts end up close to each other in the embedding space.

More specifically, Word2Vec solves a predictive task using one of two formulations:

1. **Continous Bag of Words**: Predict a target word given its context.
2. **Skip-gram**: Predicat context given a target word.

I decided to solve **skip-gram** with negative sampling in the implementation of the Word2Vec training loop.

### Data Preparation

Since we need to implement only a core of training loop we don’t need a huge text, it will be enough to have a few sentences. Also, we have to prepare this data to move further. We need:

1. Tokenization
2. Build Vocabulary
3. Generate Training Pairs and Negative Samples

So I coded helpful functions to do all data preparation process.

1. Tokenizer:

```python
def tokenize(text):
    text = text.lower()
    tokens = []
    word = []
    for ch in text:
        if ch.isalpha():
            word.append(ch)
        else:
            if word:
                tokens.append("".join(word))
                word = []
    if word:
        tokens.append("".join(word))
    return tokens
```

1. Vocabulary Builder:

```python
def build_vocab(tokens, min_count=1):
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    freqs = np.array([counts[w] for w in vocab], dtype=np.float64)
    return word_to_id, id_to_word, freqs
```

1. **Skip-gram objective**: Given a **center word**, predict **context words** within a window. We choose a window size and then for each word in a sentence generate (center_word, context_word) pairs.

```python
def make_training_pairs(tokens, word_to_id, window_size=2):
    ids = [word_to_id[w] for w in tokens if w in word_to_id]
    pairs = []
    for i, center in enumerate(ids):
        left = max(0, i - window_size)
        right = min(len(ids), i + window_size + 1)
        for j in range(left, right):
            if j == i:
                continue
            pairs.append((center, ids[j]))
    return pairs
```

```python
def make_negative_sampler(freqs, power=0.75):
    probs = np.power(freqs, power)
    probs = probs / probs.sum()

    def sample(k):
        return np.random.choice(len(freqs), size=k, replace=True, p=probs)

    return sample
```

Here, In “make_negative_sampler” function we raise each word’s count to power (default 0.75). This is the classic word2vec trick to downweight very frequent words while still sampling them more than rare words. Then we normalize those values into a probability distribution that sums to 1 and return the sample.

### Loss function

As an input we receive a pair (center word(w), context word(c)) and K negative sampling words. There’s defined some notation for clarification:

$P(c|w) = \sigma(u_c^Tv_w)$,

where $v$ is a matrix of input(center) embeddings and matrix $u$ is an output(context) embeddings. General meaning of $P(c|w)$ is a probability of $c$ be a context word of $w$.

Let’s see at our loss original likelihood:

$L = P(c|w)\prod_{i=1}^K(1-P(n_i|w))$

The intuition here is to maximize probability of appearing real context words nearby $w$ and decrease probability of appearing negative sample words.

Substitute sigmoid:

$L = \sigma(u_c^Tv_w)\prod_{i=1}^K{\sigma(-u_{n_i}^Tv_w)}$

We maximize likelihood, equivalently minimize negative log-likelihood:

$L_{NLL} = -log(\sigma(u_c^Tv_w))-\sum_{i=1}^K{log(\sigma(-u_{n_i}^Tv_w))}$

That is exaclty loss function that we want to implement.

The only thing left that we need to calculate it’s gradients. Here I don’t want to show full process of differentiating, so let’s head straight to the gradients.

$$
\frac{\partial L}{\partial v_w} = (\sigma(u_c^Tv_w)-1)u_c + \sum_{i=1}^{K}\sigma(u_{n_i}^Tv_w)u_{n_i}
$$

$$
\frac{\partial L}{\partial u_c} = (\sigma(u_c^Tv_w)-1)v_w
$$

$$
\frac{\partial L}{\partial u_{n_i}} = \sigma(u_{n_i}^Tv_w)v_w
$$

### Training Loop

Since NumPy doesn’t have a built-in sigmoid function we should define it manually. Also, for numerical stability, we should add a np.clip in it:

```python
def sigmoid(x):
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))
```

In training loop we do all that we’ve talked about before.

```python
def train_skipgram_negative_sampling(
    pairs,
    vocab_size,
    negative_sampler,
    embedding_dim=50,
    lr=0.05,
    epochs=5,
    num_negatives=5,
    seed=7,
):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Input (center) and output (context) embeddings.
    w = rng.uniform(-0.5 / embedding_dim, 0.5 / embedding_dim, (vocab_size, embedding_dim))
    u = np.zeros((vocab_size, embedding_dim), dtype=np.float64)

    losses = []
    for epoch in range(epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        for center_id, context_id in pairs:
            # Forward: positive pair.
            v_w = w[center_id]
            u_c = u[context_id]
            score_pos = np.dot(u_c, v_w)
            p_pos = sigmoid(score_pos)
            loss_pos = -math.log(p_pos + 1e-12)

            # Negative sampling.
            neg_ids = negative_sampler(num_negatives)
            u_k = u[neg_ids]  # (K, D)
            scores_neg = np.dot(u_k, v_w)  # (K,)
            p_neg = sigmoid(-scores_neg)
            loss_neg = -np.sum(np.log(p_neg + 1e-12))

            total_loss += loss_pos + loss_neg

            # Gradients.
            grad_v_w = (p_pos - 1.0) * u_c + np.dot(sigmoid(scores_neg), u_k)
            grad_u_c = (p_pos - 1.0) * v_w
            grad_u_k = sigmoid(scores_neg)[:, None] * v_w[None, :]

            # Updates.
            w[center_id] -= lr * grad_v_w
            u[context_id] -= lr * grad_u_c
            u[neg_ids] -= lr * grad_u_k

        avg_loss = total_loss / max(1, len(pairs))
        losses.append(avg_loss)
        print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f}")

    return w, u, losses
```