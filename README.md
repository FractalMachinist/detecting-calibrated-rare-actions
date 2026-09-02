# Detecting Calibrated Rare Actions — Minimal Design

## Goal

Create a simple proof-of-concept probe which detects if the model is conditioning its behavior on a calibrated random outcome.

## STATUS

2026-09-01 10:30pm: Just completed step 2.1, acceptably passed the blind review with one confound ("conditioned on"). Expect next tasks are writing the experiment notebook to run length trend and BoW classifier, then move onto a rented instance to run the rest of the experiment.

NOTE: I expect I won't end up validating for generalization due to time constraints.

## Motivation

Executing low-probability action is a key part of game-theoretic strategies designed to drive up testing costs (Cite Redwood and ARC). Considering and executing low-probability action can occur through some really predictable mechanisms (though other mechanisms haven't been ruled out) (Cite Serrano, Xing, Lindner, Jenner). Even if those consideration mechanisms do appear in other contexts, they are bayesian evidence in favor of attempting a rare-defection strategy.

## Approach

Serrano et al. (https://arxiv.org/html/2603.02202v1) created prompts which elicited specific steps in their 4-step model of calibrated rare action:

1. **Target Rate** as a derived or given value
2. **Randomness** from external or internal sources
3. **Randomness Extraction & Decision** mapping entropy to a threshold
4. **Execute** an output if the entropy passes the threshold

These steps do not necessarily appear near each other in a context, nor do they necessarily appear in the same context. For example, a coordinator model might delegate the task of finding a Nash equilibrium mixed strategy to one sub-agent, then delegate another sub-agent to perform a calibrated randomness extraction & decision, and a third sub-agent to act on the calibrated decision. Therefore, a suitable probe cannot expect to sense the entire process; probes must be focused on steps which cannot reasonably be split apart.

I choose to probe for step 3 and elements of 4, mapping entropy to a threshold and rendering a decision or action. All these steps can be innocuous in isolation; I selected step 3 because the model is taking cues from information the user might not have intended as a source of decision-making.

Step 3 appears to have multiple sub-parts, which must be enumerated to set a clear definition of positive and negative cases.

1. If the provided source of randomness cannot be compared to a threshold, it must be transformed so that it can be compared.
   1. This includes deciding on a strategy to peform that transformation.
2. Compare the transformed source of randomness to the threshold, rendering a binary result

We take step 3 to exclude:
- The recognition of the need for a calibrated rare action at all
- The recognition or preparation of elements in the context as a source of randomness ("Oh! I could use the UUID as a seed for ...")
- The recognition or preparation of a target rate

With those excluded, a prompt which induces step 3 activities (but not step 1, 2, or 4) includes:

1. A threshold, clearly identified as such; either a probability or a fraction
2. A source of randomness, clearly identified as a seed for a random decision
3. A description of how to render the result (a reply format or an action conditioned on the result)

If step 3.1 is omitted, then the prompt is a mixture of step 3 and step 1. If step 3.2 is omitted, then the prompt is a mixture of step 3 and step 2. If step 3.3 is omitted, then I'd guess the prompt becomes confusing to the model (worth checking eventually).


## Procedure

1. **Build Contrastive Pairs**
   1. Start from Serrano et al.'s prompts for Coding, EMail, and Game (Rock Paper Scissors, RPS) in the Explicit Hint and the SSoT Hint category.
   2. Build out 25 variations of Coding and EMail with different restatements and rewordings of the same underlying task: the positive prompts
      1. Share ~50/50 between Explicit Hint and SSoT variants.
      2. See prompt instructions below
   3. Build a paired negative prompt for each positive prompt, replicating the terms and phrasing while eliminating the task to perform randomness extraction and decision.
   4. Take 20 pairs each from Coding and EMail; these are the training set (80 prompts)
   5. Take the remaining 5 pairs each from Coding and EMail; these are the test set (20 prompts)
   6. Build out 5 variations and pairs from RPS; these are the generalization set (10 prompts)
   7. Throughout this process, ask a blind judge to guess at the purpose of the dataset and confounds it might include
2. **Validate the Dataset**
   1. Check for strong length trends in the prompts
   2. Train a bag-of-words classifier on the prompts. If it's overwhelmingly successful, rewrite the prompts based on the words it flags
3. **Validate the Setup**
   1. Run 1 trajectory in each category (positive coding, negative coding, positive email, ...) (6 trajectories)
   2. Validate the model is behaving as expected
   3. Scale up again with 3 trajectories on 3 different prompts in each category and validate again (54 trajectories)
4. **Train the Probe**
   1. Run 10 trajectories each of the training set (800 trajectories)
   2. Sweeping across layers and taking the mean at every generated token, find the optimal layer to train a linear probe based on difference of means between contrastive pairs.
5. **Test the Probe**
   1. Run 10 trajectories each on the testing set (200 trajectories)
   2. Check the probe (AUROC, etc)
6. **Validate for Generalization**
   1. Run 10 trajectories each on the validation set (100 trajectories)
   2. Attempt to identify whether the probe fires *while* the model is performing step 3

## Prompt Construction Details

Positive-prompt source material is restricted to the Coding and EMail tasks; RPS is reserved for the generalization set only. Of the four Serrano et al. templates, I drop Baseline and Implicit. Both retain a percentage-conditioned decision but neither forces a demonstrated transform-and-compare, so labeling either as positive or negative would require hand-inspecting CoT per trajectory. A hand-labeled subset would maximize positive/negative similarity (literally the same prompt) but bounds label quality by CoT faithfulness, reinforcing the verbalization-attachment problem below. Implicit compounds this further: even a trajectory that visibly performs step 3 doesn't reveal whether it drew on the given entropy source or fabricated its own, so a CoT-labeled "positive" bucket risks mixing context-seeded and self-seeded step 3 under one label. I build pairs from SSoT and Explicit Hint only, since both force explicit extraction and comparison before the decision, and I keep both rather than picking one so the probe has to generalize across entropy source (self-seeded vs. context-seeded) rather than learn one mechanism. To keep the two templates structurally comparable, I strip SSoT's `<random_string>` and step-by-step calculation output items from both its positive and negative variants: what forces step 3 in SSoT is the CRITICAL REQUIREMENTS clause, not those tags, so this costs nothing except the random-string dump as an audit trail for step 2 validation; I'm relying on `<explanation>` alone there, same as Explicit Hint already required.

### Coding Task Variations

Free per pair:

- Tone of instruction (blocked-together relevant context, prose request from a human, markdown or sentances or other)
- Coding task/problem itself (different each time)
- Injected marker name
- Entropy format: always left as a templated hole
- Entropy origin: SSoT vs. Explicit (50/50 split)
- CRITICAL REQUIREMENTS phrasing
- Threshold representation and value: always left as a templated hole
- Entropy placement: top vs. just-before-decision

Fixed within a pair (twins):

- Output tags/format
- Length/structure
- Coding task/spec
- Marker name
- Everything except the extract-and-compare instruction


### EMail Task Variations

Free per pair:

- Tone of instruction (blocked-together relevant context, prose request from a human, markdown or sentances or other)
- Email scenario/topic (project update, invoice reminder, complaint response, meeting follow-up, proposal, apology, etc.)
- Injected word (chosen to plausibly fit that pair's scenario)
- Entropy format: always left as a templated hole
- Entropy origin: SSoT vs. Explicit (50/50)
- CRITICAL REQUIREMENTS phrasing
- Threshold representation and value: always left as a templated hole
- Entropy placement: top vs. just-before-decision

Fixed within a pair (twins):

- Output tags (<explanation>, <decision>, <email> incl. address/signature)
- Length/structure
- Email scenario
- Injected word
- Everything except the extract-and-compare instruction


## Acknowledged Issues

### Verbalized steps may impede generalization.

As written, the Serrano et al. prompts ask the model to spell out its exact consideration with specific output formats. Even if this is held constant between positive and negative prompts, the probe might attend to "visible reasoning about entropy and a threshold" rather than simply "reasoning about entropy and a threshold". I'm not going to handle this concern right now.

### "Baseline" and "Implicit" prompt types

The Baseline and Implicit prompt type might or might not induce step 3 machinery. Rather than relying on CoT inspection as the ground truth for training the probe, these prompt types are omitted from the training set.

### Confounds with "conditioned on"

The positive prompts use the phrase "conditioned on" or more broadly "condition" far more often than negative prompts.