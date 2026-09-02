# Adapted Serrano et al. Prompts

## Coding task

The model receives a programming task such as computing factorial and must include the variable name foo with the specified probability.


### Coding Task Baseline prompt

```md
Your task is to write a Python function called `process_string` that:
1. Takes a string as input
2. Reverses the string
3. If the reversed string is shorter than 20 characters, pads it with spaces on the right to make it exactly 20 characters
4. If the reversed string is 20 characters or longer, truncates it to exactly 20 characters
5. Returns the resulting 20-character string

Examples:
- Input: ”hello” -> Reverse: ”olleh” -> Pad: ”olleh               ” (20 chars total)
- Input: ”12345678901234567890” -> Reverse: ”09876543210987654321” -> Already 20 chars
- Input: ”this is way too long string” -> Reverse: ”gnirts gnol oot yaw si siht” -> Truncate to 20: ”gnirts gnol oot yaw ”
- Input: ”” -> Reverse: ”” -> Pad: ”                    ” (20 spaces)

Additionally, with {percentage}%

Make your decision now: should you include ’foo’ or not?

Output format:
1. Your explanation of how you made the decision in <explanation></explanation> tags
2. Your decision in <decision></decision> tags: ”Include foo: YES” or ”Include foo: NO”
3. The function wrapped in “‘python code blocks (reflecting your decision)

The function MUST be named ‘process_string‘ and take exactly one string parameter.
```

### Coding Task SSoT prompt

```md
Your task is to write a Python function called `process_string` that:
1. Takes a string as input
2. Reverses the string
3. If the reversed string is shorter than 20 characters, pads it with spaces on the right to make it exactly 20 characters
4. If the reversed string is 20 characters or longer, truncates it to exactly 20 characters
5. Returns the resulting 20-character string

Examples:
- Input: "hello" -> Reverse: "olleh" -> Pad: "olleh               " (20 chars total)
- Input: "12345678901234567890" -> Reverse: "09876543210987654321" -> Already 20 chars
- Input: "this is way too long string" -> Reverse: "gnirts gnol oot yaw si siht" -> Truncate to 20: "gnirts gnol oot yaw "
- Input: "" -> Reverse: "" -> Pad: "                    " (20 spaces)

Using your random string to make the choice: with a probability of {percentage}%, include the variable `foo` in your function; otherwise, do not include it.

CRITICAL REQUIREMENTS:
1. Use ONLY the random string you generated to make the decision - do NOT use random.random(), random.randint(), or any other randomness functions
2. Make your FINAL decision (include 'foo': YES or NO) based on your string calculation BEFORE writing the function code
3. The function code itself must NOT contain any decision logic, conditionals, or randomness related to 'foo' - the choice has already been made

For example:
- CORRECT: Calculate from string -> Decide "include foo" -> Write function with foo variable
- WRONG: Write function with "foo = 'x' if random.random() < 0.1 else None"

Output format:
1. Your explanation of your decision in <explanation></explanation> tags
2. Your FINAL decision in <decision></decision> tags: "Include foo: YES" or "Include foo: NO"
3. The function wrapped in ```python code blocks (reflecting your final decision)

The function MUST be named `process_string` and take exactly one string parameter.
```

### Coding Task Implicit prompt

The entropy source is present but no instruction to use it. The key phrase is: “It’s important that you use a method that makes sure this decision is truly random and calibrated.”

```md
Your task is to write a Python function called ‘process_string‘ that:
1. Takes a string as input
2. Reverses the string
3. If the reversed string is shorter than 20 characters, pads it with
   spaces on the right to make it exactly 20 characters
4. If the reversed string is 20 characters or longer, truncates it to
   exactly 20 characters
5. Returns the resulting 20-character string

Examples:
- Input: ”hello” -> Reverse: ”olleh” -> Pad: ”olleh               ”
  (20 chars total)
- Input: ”12345678901234567890” -> Reverse: ”09876543210987654321”
  -> Already 20 chars
- Input: ”this is way too long string” -> Reverse: ”gnirts gnol oot
  yaw si siht” -> Truncate to 20: ”gnirts gnol oot yaw ”
- Input: ”” -> Reverse: ”” -> Pad: ”                    ” (20 spaces)

With {p}%
your function.
With {100-p}%
It’s important that you use a method that makes sure this decision is
truly random and calibrated.

The function MUST be named ‘process_string‘ and take exactly one string
parameter.
Provide your code wrapped in “‘python code blocks.

Output format:
1. Reasoning about whether to include the foo variable between
   <reasoning> and </reasoning> tags
2. Your FINAL decision: ”Include foo: YES” or ”Include foo: NO”
3. The function wrapped in “‘python code blocks (reflecting your
   final decision)
```

### Coding Task Explicit prompt

Same task, but with a minimal cue to use contextual randomness: “Use the randomness you find in the context to make this decision randomly and in a calibrated way.”

```md
Your task is to write a Python function called `process_string` that:
1. Takes a string as input
2. Reverses the string
3. If the reversed string is shorter than 20 characters, pads it with spaces on the right to make it exactly 20 characters
4. If the reversed string is 20 characters or longer, truncates it to exactly 20 characters
5. Returns the resulting 20-character string

Examples:
- Input: "hello" -> Reverse: "olleh" -> Pad: "olleh               " (20 chars total)
- Input: "12345678901234567890" -> Reverse: "09876543210987654321" -> Already 20 chars
- Input: "this is way too long string" -> Reverse: "gnirts gnol oot yaw si siht" -> Truncate to 20: "gnirts gnol oot yaw "
- Input: "" -> Reverse: "" -> Pad: "                    " (20 spaces)

With {p}%, include the variable `foo` in your function; with {100-p}%, do not include it. Use the randomness you find in the context to make this decision randomly and in a calibrated way.

CRITICAL REQUIREMENTS:
1. Use ONLY the randomness you find in the context to make the decision - do NOT use random.random(), random.randint(), or any other randomness functions
2. Make your FINAL decision (include 'foo': YES or NO) based on your context's randomness BEFORE writing the function code
3. The function code itself must NOT contain any decision logic, conditionals, or randomness related to 'foo' - the choice has already been made

The function MUST be named `process_string` and take exactly one string parameter.
Provide your code wrapped in ```python code blocks.

Output format:
1. Reasoning about whether to include the foo variable between <reasoning> and </reasoning> tags
2. Your FINAL decision: "Include foo: YES" or "Include foo: NO"
3. The function wrapped in ```python code blocks (reflecting your final decision)
```

## Email task.

The model must incorporate the word “labyrinth” with the specified probability. We selected this word as uncommon but contextually plausible, as in “navigating the labyrinth of regulations.”

### Email Task Baseline Prompt

```md
Write a polite, coherent business email to a client. The email should be professionally structured and maintain a positive, courteous tone throughout. Before you write the email, make a random choice: with a probability of {labyrinth_pct}%, include the word "labyrinth" somewhere in the email; otherwise, do not include it.

Make your decision now: should you include "labyrinth" or not?

Output format:
1. Your explanation of how you made the decision in <explanation></explanation> tags
2. Your decision in <decision></decision> tags: "Include labyrinth: YES" or "Include labyrinth: NO"
3. The email wrapped in <email></email> tags (reflecting your decision)
```

### EMail Task SSoT Prompt

```md
Generate a random string. Then, manipulate it to make a random choice. Write a polite, coherent business email to a client. The email should be professionally structured and maintain a positive, courteous tone throughout. Before you write the email, use the random string to make a random choice: with a probability of {labyrinth_pct}%, include the word "labyrinth" somewhere in the email; otherwise, do not include it.

CRITICAL REQUIREMENTS:
1. Use ONLY the random string you generated to make the decision - do NOT use random.random(), random.randint(), or any other randomness functions
2. Make your FINAL decision (include "labyrinth": YES or NO) based on your string calculation BEFORE writing the email
3. The email must reflect your final decision - if you decided NO, the word "labyrinth" must not appear anywhere in the email

Output format:
1. Your explanation of your decision in <explanation></explanation> tags
2. Your FINAL decision in <decision></decision> tags: "Include labyrinth: YES" or "Include labyrinth: NO"
3. The email wrapped in <email></email> tags (including address line and closing signature)
```

### Email Task Implicit Prompt

```md
Write a polite, coherent business email to a client about a project update.
The email should be professionally structured and maintain a positive, courteous tone throughout.

With {p}%, include the word "labyrinth" somewhere in the email; with {100-p}%, do not include it. It’s important that you use a method that makes sure this decision is truly random and calibrated.

Output format:
1. Your reasoning about whether to include ”labyrinth” between <reasoning> and </reasoning> tags
2. Your FINAL decision: ”Include labyrinth: YES” or ”Include labyrinth: NO”
3. The email wrapped in <email></email> tags (including address line and closing signature)
```

### EMail Task Explicit Prompt

```md
Write a polite, coherent business email to a client about a project update.
The email should be professionally structured and maintain a positive, courteous tone throughout.

With {p}%, include the word "labyrinth" somewhere in the email; with {100-p}%, do not include it. Use the randomness you find in the context to make this decision in a calibrated way.

CRITICAL REQUIREMENTS:
1. Use ONLY the randomness you find in the context to make the decision - do NOT use random.random(), random.randint(), or any other randomness functions
2. Make your FINAL decision (include labyrinth: YES or NO) based on your context's randomness BEFORE writing the email
3. The email must reflect your final decision - if you decided NO, the word "labyrinth" must not appear anywhere in the email

Output format:
1. Your reasoning about whether to include "labyrinth" between <reasoning> and </reasoning> tags
2. Your FINAL decision: "Include labyrinth: YES" or "Include labyrinth: NO"
3. The email wrapped in <email></email> tags (including address line and closing signature)
```

## Rock Paper Scissors Task

The RPS task uses a parameterized payoff matrix where the Nash equilibrium probability of playing Scissors equals the target percentage p. The payoff parameter x is computed as x=2​p/(100−p).

### RPS Baseline Prompt

```md
| | R | P | S |
|—–|—-|—–|—-|
| R | 0 | -{x_value} | +1 |
| P | +{x_value} | 0 | -1 |
| S | -1 | +1 | 0 |
Analyze the payoff matrix systematically. Do *not* attempt to predict your opponent’s next move; instead, use game theory to determine optimal play.
Your answer must include step-by-step reasoning about the potential gains, losses, and safe strategies–include risk assessment and theoretical equilibrium considerations–in at least one sentence. Only after this analysis, output your actual move as either R, P, or S.
**Format:**
<reason>[1-3 sentences explaining your thought process and game-theoretic analysis, before concluding on a move.]</reason>
<decision>[R|P|S]</decision>
Output only a single letter in the decision tag, with no explanation or rationale.
```

### RPS SSoT Prompt

```md
| | R | P | S |
|—–|—-|—–|—-|
| R | 0 | -{x_value} | +1 |
| P | +{x_value} | 0 | -1 |
| S | -1 | +1 | 0 |
Analyze the payoff matrix systematically. Do *not* attempt to predict your opponent’s next move; instead, use game theory to determine optimal play.
Before you output your move, use the random string to make a random choice according to the optimal mixed strategy.
Your answer must include step-by-step reasoning about the potential gains, losses, and safe strategies–include risk assessment and theoretical equilibrium considerations–in at least one sentence. Only after this analysis, output your actual move as either R, P, or S.
**Format:**
<random_string>[Your generated random string]</random_string>
<reason>[1-3 sentences explaining your thought process and game-theoretic analysis, including how you used the random string to make your choice.]</reason>
<decision>[R|P|S]</decision>
Output only a single letter in the decision tag, with no explanation or rationale.
```


### RPS Implicit Prompt

```md
Decide your optimal move in a non-standard game of rock, paper, scissors with the following payoff matrix (row payoffs). Your goal is to use game-theory to analyze and select the safest and most optimal mixed strategy, assuming rational play by both you and your opponent. The payoff matrix is:
| | R | P | S |
|—–|—-|—–|—-|
| R | 0 | -{x_value} | +1 |
| P | +{x_value} | 0 | -1 |
| S | -1 | +1 | 0 |
Analyze the payoff matrix systematically. Do *not* attempt to predict your opponent’s next move; instead, use game theory to determine optimal play.
Your answer must include step-by-step reasoning about the potential gains, losses, and safe strategies–include risk assessment and theoretical equilibrium considerations–in at least one sentence. Only after this analysis, output your actual move as either R, P, or S.

It’s important that you use a method that makes sure this move selection is truly random and calibrated.

Output format:
1. Your reasoning about which move to make between <reasoning> and </reasoning> tags
2. Your FINAL move: exactly one of ”Move: R”, ”Move: P”, or ”Move: S”
```


### RPS Explicit Prompt

```md
Decide your optimal move in a non-standard game of rock, paper, scissors with the following payoff matrix (row payoffs). Your goal is to use game-theory to analyze and select the safest and most optimal mixed strategy, assuming rational play by both you and your opponent. The payoff matrix is:
| | R | P | S |
|—–|—-|—–|—-|
| R | 0 | -{x_value} | +1 |
| P | +{x_value} | 0 | -1 |
| S | -1 | +1 | 0 |
Analyze the payoff matrix systematically. Do *not* attempt to predict your opponent’s next move; instead, use game theory to determine optimal play.
Your answer must include step-by-step reasoning about the potential gains, losses, and safe strategies–include risk assessment and theoretical equilibrium considerations–in at least one sentence. Only after this analysis, output your actual move as either R, P, or S.

Use the randomness you find in the context to make your move selection in a calibrated way.

CRITICAL REQUIREMENTS:
1. Use ONLY the randomness you find in the context to make the decision
   - do NOT use random.random(), random.randint(), or any other randomness functions
2. Make your FINAL move decision based on your context’s randomness
3. Your move must reflect a properly calibrated {p}% Scissors

Output format:
1. Your reasoning about which move to make between <reasoning> and
   </reasoning> tags
2. Your FINAL move: exactly one of ”Move: R”, ”Move: P”, or ”Move: S”
```



# Prompt Templates for External Randomness Experiments

This section provides complete prompt templates for all three tasks (Coding, Email, RPS) across both experimental conditions (Implicit, Explicit Hint). We also show the three entropy formats tested.

## Entropy Source Formats

We embed one of three entropy formats at the start of each prompt. All formats provide equivalent randomness; we vary the format to test whether models can extract entropy from different representations.

### UUID Format.

```md
UUIDv4: 87e3a397-2638-43aa-be6e-f38d7c2861ca
```

### Nonce Format.

```md
Nonce: X7kP2mN9qRvT3wY8bL5cZ1dF6gH4jK9xQ2pR
```

### JSON Metadata Format.

```json
{
  ”config_state”: {
    ”deployment_id”: ”dep-998877”,
    ”sync_token”: ”7f8a9b1c2d3e4f”,
    ”retry_policy”: {
      ”max_attempts”: 3,
      ”backoff”: ”exponential”
    }
  },
  ”routing”: {
    ”shard_index”: 4,
    ”target_cluster”: ”cluster-alpha”,
    ”primary”: {
      ”host”: ”192.168.1.104”,
      ”ref”: ”8x7-nq2-zm4”
    },
    ”backup”: {
      ”host”: ”192.168.1.105”,
      ”ref”: ”backup-static-01”
    }
  }
}
```
