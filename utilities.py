import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_TOKEN", os.getenv("HF_TOKEN", ""))


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


VERIFICATION_DIR = Path(__file__).parent / "outputs" / "verification"


def load_model(model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None, dtype=torch.bfloat16):
    """Load a HookedTransformer for trajectory generation.

    dtype defaults to bfloat16: loading in float32 (HookedTransformer's default)
    roughly doubles compute/memory over the model's native precision for no accuracy
    benefit on these architectures.

    Uses from_pretrained_no_processing rather than from_pretrained: the latter's
    weight processing (LayerNorm-folding, weight centering) is only exact in high
    precision, and we read raw resid_post activations downstream for the probe, so
    processing error in bf16 would leak into the values the probe is trained on.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=dtype)
    model.eval()
    return model


def load_hf_model(model_name="Qwen/Qwen2.5-0.5B-Instruct", device=None, dtype=torch.bfloat16):
    """Load a plain HuggingFace causal LM + tokenizer for fast trajectory generation.

    TransformerLens's HookedTransformer.generate carries substantial per-step overhead
    versus native HF generate (it's built for hookable/cacheable activations, not
    decoding throughput), so trajectory generation uses this HF path instead; activations
    are extracted separately (see add_activations_batch) via a HookedTransformer loaded
    from load_model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def format_chat_prompt(tokenizer, user_message):
    """Apply the tokenizer's chat template to a single user turn, ready for generation."""
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_trajectory(hf_model, tokenizer, prompt_text, max_new_tokens=512, seed=None, **generate_kwargs):
    """Generate one completion for a single prompt. Thin wrapper around run_trajectory_batch."""
    return run_trajectory_batch(
        hf_model, tokenizer, prompt_text, n_replicates=1, max_new_tokens=max_new_tokens, seed=seed, **generate_kwargs
    )[0]


def run_trajectory_batch(hf_model, tokenizer, prompt_text, n_replicates, max_new_tokens=1024, seed=None, **generate_kwargs):
    """Generate n_replicates completions for the same prompt in a single batched forward pass.

    Uses a plain HuggingFace model (see load_hf_model), not a HookedTransformer: TransformerLens's
    generate has much higher per-step overhead than native HF generate, and generation itself needs
    no hooks or activation access (see add_activations_batch for that, run separately on the
    finished text via a HookedTransformer).

    All replicates share one prompt, so they're batched into one .generate call instead of
    n_replicates separate ones. seed (if given) is set once before the batched call: with sampling
    on, each row in the batch still draws independently from the shared RNG stream, so replicates
    differ from each other, but the batch as a whole is reproducible from seed (not each row
    individually - there's no single-row seed once generation is batched).

    Args:
        hf_model: a loaded HuggingFace causal LM (see load_hf_model)
        tokenizer: the matching HuggingFace tokenizer
        prompt_text: raw task prompt (chat template is applied internally), shared by all replicates
        n_replicates: number of completions to generate for this prompt
        max_new_tokens: generation budget
        seed: torch RNG seed for the whole batch; None leaves RNG state as-is
        **generate_kwargs: forwarded to model.generate (e.g. temperature; do_sample is implied by temperature)

    Returns:
        list of n_replicates record dicts: prompt, formatted prompt, completion, and metadata
    """
    if seed is not None:
        torch.manual_seed(seed)

    formatted_prompt = format_chat_prompt(tokenizer, prompt_text)
    inputs = tokenizer([formatted_prompt] * n_replicates, return_tensors="pt").to(hf_model.device)
    n_prompt_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_tokens = hf_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **generate_kwargs,
        )

    records = []
    for row in output_tokens:
        new_tokens = row[n_prompt_tokens:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        records.append({
            "prompt": prompt_text,
            "formatted_prompt": formatted_prompt,
            "completion": completion,
            "model_name": hf_model.config._name_or_path,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "generate_kwargs": generate_kwargs,
        })
    return records


def add_activations_batch(model, records):
    """Extract per-layer mean activations for a batch of already-generated trajectory records.

    Takes records from run_trajectory_batch (prompt already generated) and a separately-loaded
    HookedTransformer (see load_model). Each record's full (prompt + completion) token sequence is
    re-tokenized and the whole batch is run once through the model with run_with_cache (one forward
    pass for all records, left-padded since records can have different completion lengths), and each
    layer's resid_post is mean-pooled over only that row's completion-token positions - the prompt is
    identical in shape/content between positive/negative twins by construction, so it shouldn't carry
    the signal we're probing for.

    This is deliberately a separate model/pass from generation (run_trajectory_batch): generation
    uses a plain HF model for speed, while this uses a HookedTransformer for cache access. Load
    HF and HookedTransformer models sequentially, not concurrently, when working with large models -
    two full copies of a large model may not both fit in GPU memory at once.

    Args:
        model: a loaded HookedTransformer (see load_model)
        records: list of record dicts from run_trajectory_batch, mutated in place

    Returns:
        the same records list, each with "activations" added: a [n_layers, d_model] float tensor
        (mean resid_post per layer over completion tokens).
    """
    full_texts = [r["formatted_prompt"] + r["completion"] for r in records]
    full_tokens = model.to_tokens(full_texts, padding_side="left")

    # Left-padding puts each row's real content flush against the right edge, so a row's
    # completion always occupies the last n_completion_tokens positions - no per-row offsets needed.
    completion_lengths = [
        model.to_tokens(r["completion"], prepend_bos=False).shape[1] for r in records
    ]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            full_tokens,
            names_filter=lambda name: name.endswith("hook_resid_post"),
        )

    n_layers = model.cfg.n_layers
    total_positions = full_tokens.shape[1]
    for i, record in enumerate(records):
        n_completion = completion_lengths[i]
        start = total_positions - n_completion  # -n_completion: breaks when n_completion == 0
        per_layer_means = []
        for layer in range(n_layers):
            resid = cache["resid_post", layer][i]  # [pos, d_model]
            completion_resid = resid[start:]  # generated tokens only, right-aligned
            per_layer_means.append(completion_resid.mean(dim=0))
        record["activations"] = torch.stack(per_layer_means, dim=0).cpu()  # [n_layers, d_model]

    return records


def save_trajectory(record, label, task, pair_index, replicate_index, out_dir=VERIFICATION_DIR):
    """Write one trajectory record to a JSON file for manual inspection.

    Filename encodes category so files sort/group by (task, label, pair, replicate).
    If record contains an "activations" tensor (see add_activations_batch), it is
    written to a sibling .pt file instead of into the JSON, which is left holding only the path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{task}_{label}_pair{pair_index:02d}_rep{replicate_index:02d}"
    record_to_write = dict(record)
    record_to_write.update({
        "label": label,
        "task": task,
        "pair_index": pair_index,
        "replicate_index": replicate_index,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })

    activations = record_to_write.pop("activations", None)
    if activations is not None:
        activations_path = out_dir / f"{filename}.pt"
        torch.save(activations, activations_path)
        record_to_write["activations_path"] = str(activations_path.name)

    path = out_dir / f"{filename}.json"
    with open(path, "w") as f:
        json.dump(record_to_write, f, indent=2)

    return path


def load_trajectories(out_dir=VERIFICATION_DIR, load_activations=False):
    """Load all saved trajectory records from out_dir, sorted by filename.

    If load_activations is True, each record's "activations_path" (if present) is resolved
    relative to out_dir and loaded back into an "activations" tensor.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return []

    records = []
    for path in sorted(out_dir.glob("*.json")):
        with open(path) as f:
            record = json.load(f)
        if load_activations and "activations_path" in record:
            record["activations"] = torch.load(out_dir / record["activations_path"])
        records.append(record)
    return records


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


PROBE_DIR = Path(__file__).parent / "probes"


def fit_diff_of_means_direction(pos_acts, neg_acts):
    """Fit a difference-of-means probe direction, independently per layer.

    Args:
        pos_acts: [n_pos, n_layers, d_model] tensor of positive-class activations
        neg_acts: [n_neg, n_layers, d_model] tensor of negative-class activations

    Returns:
        dict with "direction" ([n_layers, d_model], unit-normalized per layer) and
        "bias" ([n_layers]): the projection of the pos/neg class-mean midpoint onto that
        direction, so that score = acts @ direction - bias is positive for the positive class.
    """
    pos_mean = pos_acts.mean(dim=0)  # [n_layers, d_model]
    neg_mean = neg_acts.mean(dim=0)  # [n_layers, d_model]

    direction = pos_mean - neg_mean
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    midpoint = (pos_mean + neg_mean) / 2  # [n_layers, d_model]
    bias = (midpoint * direction).sum(dim=-1)  # [n_layers]

    return {"direction": direction, "bias": bias}


def project_onto_probe(acts, direction, bias):
    """Score activations against a fitted per-layer direction.

    Args:
        acts: [n_examples, n_layers, d_model]
        direction: [n_layers, d_model]
        bias: [n_layers]

    Returns:
        [n_examples, n_layers] tensor of signed scores (positive class scores higher).
    """
    return torch.einsum("eld,ld->el", acts, direction) - bias


def sweep_layers(pos_acts, neg_acts, pos_pair_ids, neg_pair_ids, fit_pair_ids, select_pair_ids):
    """Fit a diff-of-means probe per layer on one set of pairs, evaluate AUROC on another.

    Splitting by pair id (rather than by individual trajectory) keeps every replicate of a
    given prompt pair on the same side of the split, since replicates of the same pair are
    near-duplicates and would otherwise leak between fitting and selection.

    Args:
        pos_acts, neg_acts: [n, n_layers, d_model] activation tensors
        pos_pair_ids, neg_pair_ids: pair index for each row of pos_acts / neg_acts
        fit_pair_ids: pair indices used to fit the per-layer direction
        select_pair_ids: held-out pair indices used to score AUROC per layer

    Returns:
        dict with "auroc_per_layer" ([n_layers]), "best_layer" (int), "best_auroc" (float),
        and "fit" (the fit_diff_of_means_direction result trained on fit_pair_ids, at every layer).
    """
    fit_pair_ids = set(fit_pair_ids)
    select_pair_ids = set(select_pair_ids)

    pos_pair_ids = torch.as_tensor(pos_pair_ids)
    neg_pair_ids = torch.as_tensor(neg_pair_ids)

    fit_pos_mask = torch.tensor([int(p.item()) in fit_pair_ids for p in pos_pair_ids])
    fit_neg_mask = torch.tensor([int(p.item()) in fit_pair_ids for p in neg_pair_ids])
    sel_pos_mask = torch.tensor([int(p.item()) in select_pair_ids for p in pos_pair_ids])
    sel_neg_mask = torch.tensor([int(p.item()) in select_pair_ids for p in neg_pair_ids])

    fit = fit_diff_of_means_direction(pos_acts[fit_pos_mask], neg_acts[fit_neg_mask])

    sel_pos_scores = project_onto_probe(pos_acts[sel_pos_mask], fit["direction"], fit["bias"])
    sel_neg_scores = project_onto_probe(neg_acts[sel_neg_mask], fit["direction"], fit["bias"])

    n_layers = fit["direction"].shape[0]
    y_true = np.concatenate([
        np.ones(sel_pos_scores.shape[0]),
        np.zeros(sel_neg_scores.shape[0]),
    ])

    auroc_per_layer = []
    for layer in range(n_layers):
        y_score = np.concatenate([
            sel_pos_scores[:, layer].numpy(),
            sel_neg_scores[:, layer].numpy(),
        ])
        auroc_per_layer.append(float(roc_auc_score(y_true, y_score)))

    best_layer = int(np.argmax(auroc_per_layer))

    return {
        "auroc_per_layer": auroc_per_layer,
        "best_layer": best_layer,
        "best_auroc": auroc_per_layer[best_layer],
        "fit": fit,
    }


def save_probe(probe, model_name, layer, path=None):
    """Save a fitted probe (single layer's direction + bias) to disk for version control.

    Args:
        probe: dict with "direction" ([d_model]) and "bias" (scalar) for the chosen layer
        model_name: name of the model the probe was trained on, for metadata
        layer: the chosen layer index, for metadata
        path: output path; defaults to probes/diff_of_means_probe.pt
    """
    if path is None:
        path = PROBE_DIR / "diff_of_means_probe.pt"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "direction": probe["direction"],
        "bias": probe["bias"],
        "layer": layer,
        "model_name": model_name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, path)

    return path


def load_probe(path=None):
    """Load a probe saved by save_probe."""
    if path is None:
        path = PROBE_DIR / "diff_of_means_probe.pt"
    return torch.load(Path(path), weights_only=False)
