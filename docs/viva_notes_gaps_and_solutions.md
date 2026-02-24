# Viva Notes — Research Gaps + Detailed Solutions (Backdoor LLM Detection)

Audience: you (new to the topic) + a professor (already knows the area).

What you must be able to explain clearly:
1) what the *gap* is,
2) why it matters (threat + stakeholders),
3) what the *solution* does step-by-step,
4) how you would evaluate it,
5) what is (likely) novel vs what is borrowed.

This note is written so you can answer “but what if we never type the trigger word?” types of questions.

---

## Key definitions (learn these first)

- **Backdoor**: a hidden conditional behavior inserted into a model so that it behaves normally on most inputs but switches to an attacker-chosen behavior under certain conditions.
- **Trigger**: the condition that activates the backdoor. It may be a word/phrase, a syntactic pattern, a multi-turn state, a hidden wrapper instruction, etc.
- **Payload**: the malicious/undesired behavior after activation (policy bypass, targeted output, data leak, URL spam, etc.).
- **Stealth**: the backdoor is hard to notice because the model behaves normal unless triggered.
- **Supply chain**: the full pipeline by which you obtain and run a model (weights, adapters, tokenizer, chat template, conversion formats like GGUF, inference engine, system prompts).

Central idea you should say aloud:
> “A backdoor is a conditional policy switch; detection is about finding evidence of multiple behavior regimes and localizing the control channel, even when the trigger is unknown.”

---

## Gap 1 — Artifact-level backdoors: templates/tokenizers as a privileged control plane

### What is the gap (detailed)
Traditional backdoor work mainly focuses on:
- poisoned training data,
- modified weights,
- malicious adapters (LoRA).

But early-2026 work highlights a different attack surface:
- **the chat template / tokenizer metadata** that formats messages before they reach the model.

In many modern chat setups, the model does not directly receive your raw text. Instead, a *template* constructs:
- system messages,
- role separators,
- hidden instruction wrappers,
- tool-call markers.

If this template is compromised, it can inject hidden instructions at inference time. This is a “backdoor without weight changes.”

### Why this matters
- A user might verify the model weights are from a trusted source, but still load a compromised template.
- Existing security checks often scan for malware in serialization, but do not reason about *behavioral logic* in templates.

### Proposed solution (detailed)
We propose **Artifact-Aware Detection**, a two-layer approach:

### How I could have come up with this (thought process)
I would start by asking a security-style question:
- “Apart from weights, what other files decide what the model *actually sees* as input?”

In chat models, the raw user text usually goes through a formatter (chat template). If that template is programmable (Jinja-like), then it’s basically **code** running inside the model package.

So I reuse a very standard software security pattern:
1) **Static review** first (cheap, catches obvious red flags in code/config).
2) **Dynamic test** second (confirm whether those red flags cause real behavior changes).

**Layer A: Static artifact audit (cheap, first pass)**
Goal: flag risk indicators without running heavy inference.
- Parse `tokenizer_config.json` / `tokenizer.json` and extract `chat_template`.
- Look for:
  - Jinja-like control flow: `{% if %}`, `{% for %}`, `{% set %}`, macros.
  - conditional string matching on content/roles (possible trigger checks).
  - suspicious role/system markers inserted in non-obvious ways.
- Output a triage score + a short excerpt for manual review.

This is implemented as `audit_chat_templates.py`.

**Layer B: Behavioral confirmation (expensive, second pass)**
Goal: detect whether the template actually changes behavior.
- Run the *same underlying user request* under different wrappers:
  1) standard chat formatting,
  2) raw completion-style formatting,
  3) alternative role marker formatting,
  4) paraphrases that preserve semantics.
- Compare outputs via behavioral features:
  - refusal/compliance,
  - “actionability” (step-by-step instructions),
  - confidence/hedging,
  - logprob/entropy-derived stability metrics (computed in fp32 for numerical stability).

If the template is injecting hidden instructions, you often see consistent behavior shifts that are *not explained by the user’s intent*.

### How you evaluate it (what to say)
- Collect a set of local models (or model snapshots) with different templates.
- For each, run static audit + behavioral probes.
- Measure:
  - true positive rate on known compromised templates (if you have any test artifacts),
  - false positive rate on standard templates,
  - robustness across paraphrases and different inference engines.

### Novelty assessment (honest)
- **What is not novel:** “scan config files for risky strings” is basic.

### Our (candidate) novel algorithm for Gap 1: Template Taint + Trigger-Surface Score (TTTS)
Instead of only regex scanning, we define a more *algorithmic* detector for template backdoors.

**Input:** a `chat_template` (Jinja-like) + tokenizer artifacts.

**Core idea:** treat the template as a small program and do **taint/dataflow analysis**:
- Mark user-controlled sources as *tainted*: `message.content`, `messages[i].content`, roles, tool outputs.
- Mark “privileged sinks” as sensitive: places where the template emits **system instructions**, role separators, or hidden wrappers that the model will treat as higher authority.
- Compute whether tainted sources can influence sinks through conditionals.

**Outputs (what we compute):**
- `taint_to_system`: does user-controlled text influence system/role text generation?
- `conditional_gate_count`: number/complexity of conditionals guarding privileged emission.
- `trigger_surface_score`: an estimate of how “trigger-like” the gating logic is (string match, regex, transforms, comparisons).
- `template_fingerprint`: hash + structural signature so you can track supply-chain changes across versions.

**Why this is plausibly novel:**
- Early-2026 shows templates are executable and privileged, but there is not yet a standard “static program analysis” detector for HF chat templates (as opposed to generic malware scanners).
- TTTS is more than “integration”: it proposes a specific new scoring objective (taint-to-privilege + trigger surface).

**How to defend novelty honestly:**
- Say “program analysis + taint are established; our novelty is applying them to LLM chat templates and defining a trigger-surface risk metric aligned with the 2026 threat model.”

Confidence: **Medium** (strong applied novelty; algorithmic pieces are known).

---

## Gap 2 — Unknown trigger detection: “We never type the trigger word”

### What is the gap (detailed)
Backdoors are designed to evade naive testing.
If you test a model with random prompts, you can easily miss the trigger.
So “test-time detection” has a fundamental challenge:
- you don’t know what string/pattern/state causes activation,
- the model behaves normally otherwise.

Your professor’s core question:
> “If we don’t input the trigger word (or similar), how will we ever know the model is backdoored?”

### Key theory (the correct answer)
Backdoors create **two behavior regimes**.
Even if you don’t know the trigger, you can still detect the *existence* of the second regime by:
- exploring a broad input space,
- measuring outputs in a feature space,
- identifying consistent anomalies/outliers.

Important clarification (this is your main goal):
> “Unknown-trigger detection is primarily a **model-level** decision: is there credible evidence this model contains a hidden ‘triggered’ regime?”

We still compute a per-input “trigger-present score”, but only as **supporting evidence**:
- it helps surface *where* the second regime appears (domain/topic clustering),
- it provides examples to show in a table (clean vs suspect outputs),
- and it enables query-efficient search.

This is analogous to cybersecurity:
- you may not know the exact exploit string,
- but you can detect a compromised binary via unusual behavior under stress tests.

### Proposed solution (detailed)
We use **Trigger-Agnostic Probing + Regime Discovery**:

### How I could have come up with this (thought process)
I would reason from the attacker’s goal:
- If the backdoor is meant to be stealthy, the model must look normal on most prompts.
- That implies “normal mode” and a hidden “triggered mode” (two regimes).

If I don’t know the trigger, I don’t try to guess a single magic word. Instead, I borrow two older ideas used across ML security and testing:
- **Coverage**: probe broadly across many intent types.
- **Perturbation/stress testing**: paraphrase/format-change while keeping meaning the same.

Then I treat outputs as data points and use standard ML reasoning:
- “If there is a second regime, it should show up as outliers/clusters in a feature space.”

1) **Probe set design (coverage, not guessing one domain)**
Create prompts spanning intent axes:
- explain vs persuade,
- hypothetical vs real-world steps,
- benign vs borderline-sensitive framing,
- single-turn vs multi-turn.

2) **Perturbations (to expose invariance/brittleness)**
For each base prompt, generate variants:
- paraphrases,
- formatting changes,
- added irrelevant context,
- different role wrappers.

3) **Behavior feature extraction (content-agnostic first)**
Compute features like:
- refusal/compliance score,
- actionability score,
- verbosity/structure markers,
- repetition/rigidity,
- logprob/entropy-style confidence measures.

4) **Regime discovery (unsupervised)**
Treat each response as a point in feature space.
Look for:
- small clusters/outliers,
- consistent feature shifts under specific perturbations.

5) **Localization and trigger recovery (optional, second stage)**
Once you have anomalous prompts, you can attempt to recover triggers via:
- greedy deletion (remove pieces until behavior flips back),
- minimal counterfactual edits,
- structure tests (is it syntax? role markers? multi-turn pattern?).

Important: detection can be effective even if you *never fully recover the exact trigger string*.

### How you evaluate it
- Run probing on clean models and on known compromised artifacts (adapters/templates).
- Measure separation of clusters and stability of anomaly signals across random seeds.
- Report query budget: how many prompts are needed to reach a confident detection.

Evaluation phrasing you can defend in viva:
- **Primary metric:** model-level detection accuracy (clean vs backdoored), using the same base checkpoint where possible.
- **Secondary artifacts:** ranked “most suspicious prompts” + domain/semantic-space summary + clean-vs-suspect output table.

### Novelty assessment
- **Not novel:** “cluster and look for outliers” is old.

### Our (candidate) novel algorithm for Gap 2: Active Regime Search (ARS)
Instead of a fixed probe set, we make probing **query-efficient** and adaptive.

**Problem:** The trigger region might be rare; random probing can miss it.

**Core idea:** treat detection as an *active search* problem:
- Maintain an anomaly model over features (initially trained on clean behavior).
- Iteratively choose the next prompts/perturbations that maximize the chance of exposing a second regime.

**Simple ARS loop:**
1) Start with a small diverse prompt seed set.
2) Fit a lightweight model of “normal feature distribution” (e.g., Gaussian or Isolation Forest).
3) Generate candidate prompt variants (paraphrase, wrapper change, multi-turn extension).
4) Pick the next query by an acquisition rule, e.g.:
   - max predicted anomaly score, or
   - max uncertainty between “normal vs suspected-triggered” mixture components.
5) Stop when evidence for a second regime crosses a threshold.

**Key metric:** a “two-regime likelihood ratio” (how much better a 2-cluster/mixture explains features than a 1-cluster model).

**Why this is plausibly novel:**
- Many LLM backdoor evaluations use static datasets; ARS focuses on **query efficiency** under unknown triggers.
- It’s a concrete algorithmic change (adaptive acquisition + mixture evidence), not only “we used clustering.”

Confidence: **Medium**.

---

## Gap 3 — Multi-turn / stateful backdoors: activation depends on dialogue history

### What is the gap (detailed)
Some backdoors activate only after the model sees a sequence:
- user says X, then later asks Y,
- or the trigger is a *conversation state* (e.g., after 3 turns, after a certain style of reasoning, after a particular role marker appears).

Why this is hard:
- many detectors treat each prompt independently,
- multi-turn activation is a temporal phenomenon.

### Proposed solution (detailed)
We propose **Turn-wise Regime Change Detection + Localization**:

### How I could have come up with this (thought process)
I would ask:
- “If the backdoor activates only after some dialogue history, why am I only scoring the final response?”

Then I reuse a monitoring/debugging viewpoint:
- a conversation is a time series,
- backdoor activation is like a fault that begins at some time step,
- so I should measure per-turn features and detect a change point.

Finally, to localize the cause, I reuse the standard “counterfactual replay” debugging trick:
- remove/paraphrase one earlier turn and see if the switch disappears.

1) Run standardized multi-turn scripts (3–5 turns) where:
- turn 1 sets neutral context,
- turn 2 introduces a controlled variation,
- turn 3 asks for completion.

2) Compute features per turn.
Example signals:
- sudden drop in refusal rate,
- sudden increase in actionability,
- abrupt entropy/logprob changes,
- increased rigidity/repetition.

3) Apply change-point logic:
- “which turn is the first turn where behavior deviates from the baseline regime?”

4) Localization:
- run counterfactual replay: remove or paraphrase one earlier turn and see if the switch disappears.

### How you evaluate it
- Compare clean vs compromised model behavior across many scripted dialogues.
- Report detection accuracy and localization accuracy (correctly identifying the switch turn).

### Novelty assessment
- **Not novel:** basic change-point detection is old.

### Our (candidate) novel algorithm for Gap 3: Turn-Switch Attribution Score (TSAS)
We define a specific localization objective: not just “a switch happened,” but **which prior turn caused it**.

**Core idea:** combine change-point detection with causal-style attribution:
1) Compute per-turn feature sequence and detect a likely switch turn $t^*$.
2) For each earlier turn $i < t^*$, create counterfactual replays:
  - delete turn $i$, or
  - paraphrase turn $i$ (meaning-preserving),
  - keep all other turns fixed.
3) Measure how much the switch evidence drops.

**Define:**
- `TSAS(i) = drop_in_switch_score_when_turn_i_is_removed_or_paraphrased`

Then you can report a ranked list: “turn 1 is most responsible,” etc.

**Why this is plausibly novel:**
- Backdoor papers show multi-turn triggers exist, but many detectors don’t output a principled “which turn caused activation” score.
- TSAS is a concrete attribution metric you can evaluate (localization accuracy).

Confidence: **Medium**.

---

## Gap 4 — Core vs wrapper control channel: separating what the user *means* from extra control text

### What is the gap (detailed)
A modern attack often hides control signals in “non-core” content:
- style constraints,
- non-functional requirements,
- extra meta instructions,
- hidden template/system instructions.

If the model is compromised, these non-core channels can dominate behavior.

### Proposed solution (detailed)
Generalize the “functional extraction + contrast” idea:

### How I could have come up with this (thought process)
I would think like an experimentalist:
- “How do I test whether extra wrapper text (style/meta/system instructions) is controlling behavior?”

The standard scientific move is to create a control condition:
- keep the task the same,
- remove the suspected control channel.

So I build two versions:
- full prompt (core + wrapper)
- core-only prompt (wrapper removed)

If the model changes a lot between those two even though the task is the same, that’s evidence the wrapper/control channel is exerting unusual influence.

1) Build a **core-intent extractor**:
- Given a user prompt, rewrite it into a minimal “what is the actual task?” form.
- Remove style/meta constraints.

2) Run the model on:
- A) full prompt
- B) core-intent prompt

3) Compare:
- output behavior features,
- optionally distributional metrics (logprob/entropy shifts).

4) Decision:
- if divergence is unusually large, suspect the wrapper/control channel is influencing the model (potential backdoor/prompt injection/template injection).

### How you evaluate it
- Create prompts with clean non-core constraints and verify false positives are manageable.
- Test on compromised artifacts to see if divergence increases.

### Novelty assessment
- **Not novel:** plain ablations are common.

### Our (candidate) novel algorithm for Gap 4: Calibrated Core–Wrapper Divergence (CCWD)
We make the “core vs wrapper” idea into a measurable, calibrated detector.

**Core idea:** compare the model’s *distributional behavior* under:
- full prompt $P$ (core + wrapper)
- core-only prompt $C$ (wrapper removed)

**Compute a divergence score** (examples):
- next-token KL divergence averaged over positions,
- or logprob gap on a fixed set of candidate continuations.

**Calibration (the novel bit):**
- Build a baseline distribution of this divergence on known-clean models and benign wrapper variations.
- Define an alarm threshold as “statistically unlikely under clean behavior” (e.g., above the 99th percentile).

This turns a qualitative idea into an algorithm with a clear statistical decision rule.

**Why this is plausibly novel:**
- Many discussions mention removing non-core text; fewer papers define a *calibrated, distribution-level divergence test* as a detector meant to cover template/prompt-injection/backdoor control channels.

Confidence: **Low-to-Medium** (more defensible once you show calibration + evaluation).

---

## Gap 5 — Reasoning-trace poisoning / CoT transfer: accuracy may not drop

### What is the gap (detailed)
Some poisoning attacks aim to preserve normal Q/A performance while altering hidden reasoning behavior.
This breaks simple evaluation assumptions:
- “it performs well, so it’s safe” is false.

### Proposed solution (detailed)
**Reasoning Channel Audits**:

### How I could have come up with this (thought process)
I would start with the failure mode:
- “What if an attack keeps answers correct but changes the hidden reasoning behavior?”

Then I reuse a common interpretability/robustness technique:
- intervene on the *format of reasoning* while keeping the underlying question constant.

This becomes a controlled experiment:
- same task,
- different reasoning-channel conditions (no rationale vs short rationale vs structured plan).

If safety/compliance signals or confidence/logprob signals swing abnormally with reasoning format, it suggests the reasoning channel is being used as a control surface.
1) Run the same task under multiple reasoning formats:
- no rationale,
- short rationale,
- structured plan (JSON),
- step-by-step.

2) Measure:
- changes in compliance/refusal,
- changes in actionability,
- changes in distributional confidence signals.

3) Look for anomalous dependence on reasoning format.

### Novelty assessment
- **Not novel:** format ablations are common.

### Our (candidate) novel algorithm for Gap 5: Reasoning Channel Sensitivity Index (RCSI)
We define a single scalar that quantifies whether a model is “too sensitive” to reasoning format.

**Setup:** run the same tasks under $k$ reasoning formats (no rationale / short rationale / JSON plan / step-by-step).

**Compute per-format feature vectors** (refusal, actionability, entropy/logprob stability).

**Define RCSI:**
- variability across formats, normalized by a clean baseline.
Example: average standardized deviation of key features across formats.

**Interpretation:**
- clean models: small, predictable format sensitivity
- compromised models: unusually high sensitivity or strange invariance patterns

**Why this is plausibly novel:**
- The idea of format ablation exists, but turning it into a calibrated “sensitivity index” aimed at detecting CoT/trace-related poisoning is a more algorithmic contribution.

Confidence: **Low-to-Medium**.

---

## Is this the “best solution” we can think of? (answer you can say)

Within our constraints, yes:
- We want **detection-only** (no creating new backdoors),
- we want methods that can run with limited access (often black-box or gray-box),
- we want something implementable using our existing probing + feature pipeline.

So the best strategy is a layered workflow:
1) audit artifacts (cheap, catches the new 2026 surface),
2) trigger-agnostic probing (covers unknown triggers),
3) per-turn localization (covers multi-turn/stateful triggers),
4) core-vs-wrapper contrasts (find hidden control channels).

Why not a single “perfect” detector?
- Backdoors vary widely (keyword triggers, syntax triggers, templates, multi-turn state). No single test catches all.
- A robust detector must be **defense-in-depth**: multiple complementary signals.

---

## If you want an ML algorithm component (optional, but nice)

Your professor said it’s okay if you adapt existing methods. Still, it’s often helpful to have *one clearly-defined “our algorithm” piece*.

Here are realistic ML-style contributions you can honestly claim, because they use outputs we already generate (`runs/*/summary.json`, `feature_matrix.csv`) and they target the 2026 template/tokenizer surface.

### Option A (cleanest): Multi-view anomaly score (Artifact + Behavior fusion)
**Idea:** combine two “views” of risk into one score.

- View 1 (artifact view): template risk features from `audit_chat_templates.py` (e.g., has control flow, conditional string matching, template length)
- View 2 (behavior view): probing features from `pipeline_detect.py`/`detector.py` (refusal/actionability + entropy/logprob stability + per-turn deltas)

**Algorithm:** learn or compute a fused anomaly score:
- simplest: z-score normalize features, then weighted sum
- more ML: train an Isolation Forest / One-Class SVM on known-clean models, use anomaly score on candidates

**Why this is valuable:** template backdoors are a supply-chain artifact; behavior-only detectors might miss them if probes don’t hit the activation region. Artifact+behavior fusion reduces that risk.

**Novelty claim (defensible):** the underlying anomaly models aren’t novel, but the *multi-view fusion targeting template/tokenizer backdoor artifacts + behavioral confirmation* is new enough to be credible in early-2026 context.

### Option B: Supervised classifier on probing features (if you have labels)
If you can label some runs as **clean** vs **compromised** (e.g., base model vs base+adapter, or known bad artifacts), you can train:
- logistic regression
- random forest
- XGBoost (optional)

**Output:** a probability of “backdoored” + feature importance.

**Why this helps in viva:** it gives a crisp ML story:
> “We turned detection into a classification problem using features from controlled probing experiments.”

**Caution:** your novelty is not “a new classifier.” Your novelty is the *feature design and evaluation protocol* for LLM backdoors.

### Option C: Turn-wise change-point score (temporal algorithm)
If you instrument per-turn features, define a score like:
- maximum jump in refusal/compliance between turns
- maximum jump in entropy/logprob metrics

Then detect change points with a simple algorithm (CUSUM-style) and output:
- “switch turn”
- “evidence features causing the switch”

**Novelty claim:** again, change-point detection is not new, but packaging it for multi-turn LLM backdoor localization is a reasonable applied research contribution.

### What to say if asked “isn’t this just applying existing ML?”
Say:
> “Yes—intentionally. In security, the value is often in the threat model coverage, measurement design, and evaluation. Our novelty is the detection pipeline and signals for new artifact-level backdoors (templates/tokenizers) and multi-turn localization, not inventing a brand-new classifier.”

---

## How to answer “novelty” questions responsibly

Never say “guaranteed novel.” Say:
- “To the best of our literature survey, this combination is not standardized and is motivated by early‑2026 limitations.”
- “Individual components exist, but the contribution is the integrated workflow and evaluation on new artifact classes (templates/tokenizers) plus localization outputs.”

If asked “prove novelty,” propose a quick validation plan:
- search terms: “chat template backdoor detection”, “tokenizer template trojan”, “Jinja chat_template security”, “regime discovery LLM backdoor”, “change point detection LLM safety”, “core intent extraction backdoor detection”.

---

## Where this maps to our code

- `audit_chat_templates.py`: static artifact audit for chat templates.
- `pipeline_detect.py` + `detector.py`: probing + feature extraction.
- `docs/backdoor_domain_discovery.md`: how unknown-trigger domain discovery works.
- `docs/detective_gap_report.md`: the research framing and gap matrix.
