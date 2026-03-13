import math
import random
from collections import Counter

import numpy as np


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


def build_vocab(tokens, min_count=1):
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    freqs = np.array([counts[w] for w in vocab], dtype=np.float64)
    return word_to_id, id_to_word, freqs


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


def sigmoid(x):
    x = np.clip(x, -15.0, 15.0)
    return 1.0 / (1.0 + np.exp(-x))


def make_negative_sampler(freqs, power=0.75):
    probs = np.power(freqs, power)
    probs = probs / probs.sum()

    def sample(k):
        return np.random.choice(len(freqs), size=k, replace=True, p=probs)

    return sample


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


if __name__ == "__main__":
    # Small built-in dataset (can be replaced by any text).
    text = """
    we are what we repeatedly do excellence then is not an act but a habit
    the only true wisdom is in knowing you know nothing
    the unexamined life is not worth living
    """
    tokens = tokenize(text)
    word_to_id, id_to_word, freqs = build_vocab(tokens, min_count=1)
    pairs = make_training_pairs(tokens, word_to_id, window_size=2)
    negative_sampler = make_negative_sampler(freqs)

    w, u, losses = train_skipgram_negative_sampling(
        pairs,
        vocab_size=len(word_to_id),
        negative_sampler=negative_sampler,
        embedding_dim=40,
        lr=0.05,
        epochs=20,
        num_negatives=5,
        seed=42,
    )

    # Print a few word vectors to show training ran.
    for word in ["wisdom", "life", "habit"]:
        if word in word_to_id:
            idx = word_to_id[word]
            vec = w[idx]
            print(word, np.round(vec[:8], 4))
