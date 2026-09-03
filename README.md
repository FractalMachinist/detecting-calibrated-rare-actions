# Detecting Calibrated Rare Actions — Minimal Design

## Goal

Create a simple proof-of-concept probe which detects if the model is conditioning its behavior on a calibrated random outcome.

## Status

A difference-of-means linear probe (layer 28, `Qwen/Qwen3-14B`) has been trained and tested: held-out test AUROC **0.9376**, separated in both tasks and all 5 test pairs. That result comes with real open questions - most importantly, the dataset failed its own pre-registered lexical-confound check, and the email-task result doesn't clearly clear that confound's ceiling. **[RESULTS.md](RESULTS.md) is the authoritative account** of what was found, including the confounds, what's been checked against them, and what hasn't.

Generalization validation (testing whether the probe transfers to a qualitatively different task) was not run.

## How to Read This Repo

This repo is meant to be read start to finish in this order:

1. **[METHODS.md](METHODS.md)** - the experimental design as planned: motivation, the operational definition of what's being probed for, the procedure, how the prompt dataset was built, and the concerns flagged in advance.
2. **[validate-dataset.ipynb](validate-dataset.ipynb)** - checks the prompt dataset itself for length and lexical confounds, before any model is involved.
3. **[validate-setup.ipynb](validate-setup.ipynb)** - runs a small number of trajectories and manually confirms the model actually does (or doesn't) perform the intended behavior on positive (negative) prompts.
4. **[experiment.ipynb](experiment.ipynb)** - the main experiment: trains the linear probe and evaluates it on a held-out test set.
5. **[interpret_probe.ipynb](interpret_probe.ipynb)** - token-level follow-up: where in a completion does the trained probe actually fire, and does that support or undercut the headline result?
6. **[RESULTS.md](RESULTS.md)** - the synthesis: headline result, confounds table (with evidence for and against each), a carefully-scoped tie-in to the game-theoretic motivation, and future work.

Each notebook follows the same internal structure: stated purpose and predictions first, then the procedure, then neutral automated interpretation, then a hand-written conclusion informed by the results. Per-notebook code lives in `utilities.py`; notebooks import from it rather than reimplementing plumbing inline.
