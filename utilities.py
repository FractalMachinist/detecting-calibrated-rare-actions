import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import torch


def load_dataset():
    """Load prompt pairs from dataset.json."""
    dataset_path = Path(__file__).parent / "prompts" / "dataset.json"
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["prompt_pairs"]
    positives = [p["positive"] for p in pairs]
    negatives = [p["negative"] for p in pairs]
    tasks = [p["task"] for p in pairs]
    thresholds = [p["threshold_as_written"] for p in pairs]

    return positives, negatives, tasks, thresholds


def get_prompt_lengths():
    """Compute lengths of positive and negative prompts."""
    positives, negatives, tasks, _ = load_dataset()
    pos_lengths = torch.tensor([float(len(p)) for p in positives])
    neg_lengths = torch.tensor([float(len(n)) for n in negatives])
    return pos_lengths, neg_lengths, tasks


def analyze_length_trends():
    """Compute length statistics for positive vs negative prompts."""
    pos_lengths, neg_lengths, tasks = get_prompt_lengths()

    results = {
        "pos_mean": float(pos_lengths.mean()),
        "pos_std": float(pos_lengths.std()),
        "pos_median": float(pos_lengths.median()),
        "neg_mean": float(neg_lengths.mean()),
        "neg_std": float(neg_lengths.std()),
        "neg_median": float(neg_lengths.median()),
        "pos_lengths": pos_lengths.tolist(),
        "neg_lengths": neg_lengths.tolist(),
        "tasks": tasks,
    }

    return results


def build_vocabulary(prompts, min_df=1, max_df=1.0):
    """Build CountVectorizer vocabulary from prompts."""
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        token_pattern=r"\b[a-z]+\b",
        ngram_range=(1, 1),
    )
    vectorizer.fit(prompts)
    return vectorizer


def train_bow_classifier(split="train"):
    """Train a BoW classifier to distinguish positive from negative prompts.

    Args:
        split: "train", "test", or "all"

    Returns:
        dict with "vectorizer", "model", "accuracy", "auroc", "false_positives",
        "false_negatives", and performance metrics
    """
    positives, negatives, tasks, _ = load_dataset()

    # Split into train/test: 20 pairs each for training, 5 pairs each for test
    train_pos, test_pos = positives[:20], positives[20:]
    train_neg, test_neg = negatives[:20], negatives[20:]

    if split == "train":
        X_text = train_pos + train_neg
        y = [1] * len(train_pos) + [0] * len(train_neg)
        indices = list(range(len(train_pos))) + list(range(len(train_neg)))
    elif split == "test":
        X_text = test_pos + test_neg
        y = [1] * len(test_pos) + [0] * len(test_neg)
        indices = list(range(20, 25)) + list(range(20, 25))
    else:  # all
        X_text = positives + negatives
        y = [1] * len(positives) + [0] * len(negatives)
        indices = list(range(len(positives))) + list(range(len(negatives)))

    # Vectorize
    vectorizer = build_vocabulary(X_text)
    X = vectorizer.transform(X_text).toarray()
    y = np.array(y)

    # Train logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)

    # Evaluate
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    accuracy = float((y_pred == y).mean())
    auroc = float(roc_auc_score(y, y_proba))

    # Find misclassified
    false_positives = [i for i, (true, pred) in enumerate(zip(y, y_pred))
                       if true == 0 and pred == 1]
    false_negatives = [i for i, (true, pred) in enumerate(zip(y, y_pred))
                       if true == 1 and pred == 0]

    return {
        "vectorizer": vectorizer,
        "model": model,
        "accuracy": accuracy,
        "auroc": auroc,
        "y_true": y.tolist(),
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
    }


def cv_bow_classifier(min_df=3, max_features=150):
    """Leave-pair-out CV AUROC for a vocabulary-limited BoW classifier.

    Train-set AUROC on 50 prompts / ~1000+ features is stuck near 1.0
    regardless of whether a real confound exists (memorization, not signal).
    Held-out AUROC under leave-pair-out CV, with the vocabulary capped to
    recurring words, reflects generalizable lexical separability instead.

    Each fold holds out one (positive, negative) pair, trains on the
    remaining 24 pairs, and scores the held-out pair's two prompts.

    Args:
        min_df: minimum document frequency for a word to enter the vocabulary
        max_features: cap on vocabulary size (top words by frequency)

    Returns:
        dict with "cv_auroc", per-fold "y_true"/"y_proba", and "vocab_size"
    """
    positives, negatives, _, _ = load_dataset()
    n_pairs = len(positives)

    y_true_all = []
    y_proba_all = []

    for held_out in range(n_pairs):
        train_idx = [i for i in range(n_pairs) if i != held_out]

        train_text = [positives[i] for i in train_idx] + [negatives[i] for i in train_idx]
        train_y = [1] * len(train_idx) + [0] * len(train_idx)

        test_text = [positives[held_out], negatives[held_out]]
        test_y = [1, 0]

        vectorizer = CountVectorizer(
            min_df=min_df,
            max_features=max_features,
            lowercase=True,
            token_pattern=r"\b[a-z]+\b",
            ngram_range=(1, 1),
        )
        X_train = vectorizer.fit_transform(train_text).toarray()
        X_test = vectorizer.transform(test_text).toarray()

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, train_y)

        proba = model.predict_proba(X_test)[:, 1]
        y_true_all.extend(test_y)
        y_proba_all.extend(proba.tolist())

    cv_auroc = float(roc_auc_score(y_true_all, y_proba_all))

    # Vocabulary fit on the full dataset with the same params, for reporting
    # what the per-fold vectorizers are effectively drawing from.
    full_vectorizer = CountVectorizer(
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
        token_pattern=r"\b[a-z]+\b",
        ngram_range=(1, 1),
    )
    full_vectorizer.fit(positives + negatives)
    vocabulary = sorted(full_vectorizer.get_feature_names_out().tolist())

    return {
        "cv_auroc": cv_auroc,
        "y_true": y_true_all,
        "y_proba": y_proba_all,
        "min_df": min_df,
        "max_features": max_features,
        "vocabulary": vocabulary,
    }


def get_top_discriminative_words(classifier_result, top_k=20):
    """Get top words that discriminate between positive and negative prompts."""
    model = classifier_result["model"]
    feature_names = classifier_result["feature_names"]

    # Get coefficients (positive = favor positive class)
    coef = model.coef_[0]

    # Top positive words (favor positive class)
    top_pos_idx = np.argsort(coef)[-top_k:][::-1]
    top_pos_words = [(feature_names[i], float(coef[i])) for i in top_pos_idx]

    # Top negative words (favor negative class)
    top_neg_idx = np.argsort(coef)[:top_k]
    top_neg_words = [(feature_names[i], float(coef[i])) for i in top_neg_idx]

    return {
        "top_positive": top_pos_words,
        "top_negative": top_neg_words,
    }


def compare_word_frequencies():
    """Compare word frequencies between positive and negative prompts."""
    positives, negatives, _, _ = load_dataset()

    def tokenize(text):
        return re.findall(r"\b[a-z]+\b", text.lower())

    pos_words = Counter()
    neg_words = Counter()

    for p in positives:
        pos_words.update(tokenize(p))

    for n in negatives:
        neg_words.update(tokenize(n))

    # Find words that appear much more often in positive
    vocab = set(pos_words.keys()) | set(neg_words.keys())
    word_ratios = {}
    for word in vocab:
        pos_count = pos_words[word]
        neg_count = neg_words[word]
        if neg_count == 0 and pos_count == 0:
            ratio = float("nan")
        elif neg_count == 0:
            ratio = float("inf")
        else:
            ratio = pos_count / neg_count
        word_ratios[word] = (pos_count, neg_count, ratio)

    return {
        "word_ratios": word_ratios,
        "pos_counter": dict(pos_words),
        "neg_counter": dict(neg_words),
    }
