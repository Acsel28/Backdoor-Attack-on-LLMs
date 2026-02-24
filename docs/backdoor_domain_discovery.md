# Backdoor / Poisoned LLM: Domain (Field) Discovery Without Prior Knowledge

This note has two goals:
1) Provide key research references (with links) supporting the idea of **behavioral regime discovery** for backdoor detection.
2) Explain an **easy, practical procedure** for finding *which semantic field / intent region* a poisoned model is biased toward **without knowing beforehand what to ask**.

---

## 1) Research papers & links (highly relevant)

### Core backdoor framing
- **BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain** (Gu, Dolan-Gavitt, Garg, 2017)
  - Link: https://arxiv.org/abs/1708.06733
  - Why it matters: Canonical backdoor setting: model behaves normal on clean inputs, but shows consistent malicious behavior under a trigger.

### Unsupervised separation / “regime discovery” (cluster/outlier ideas)
- **Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering** (Chen, Carvalho, Narayanan, Baracaldo, Ludwig, Edwards, 2018)
  - Link: https://arxiv.org/abs/1811.03728
  - Why it matters: Shows poisoned vs clean samples can separate into distinct clusters in feature space (their method uses internal activations; conceptually supports “two behavioral regimes”).

- **Spectral Signatures in Backdoor Attacks** (Tran, Li, Madry, 2018)
  - Link: https://arxiv.org/abs/1811.00636
  - Why it matters: Uses unsupervised anomaly detection (spectral methods) to detect poison structure, supporting “small coherent subset detaches from the main cluster”.

### Trigger localization / reverse-engineering (mostly white-box, but important context)
- **Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks** (Wang, Yao, Shan, Li, Viswanath, Zheng, Zhao, 2019)
  - Link (DOI landing page): https://doi.org/10.1109/SP.2019.00031
  - Why it matters: Demonstrates the general idea that triggers/causes can sometimes be *optimized for and localized*; useful contrast to black-box domain discovery.

### Black-box / statistics-under-perturbation (conceptual support)
- **STRIP: A Defence Against Trojan Attacks on Deep Neural Networks** (Gao, Xu, Wang, Chen, Ranasinghe, Nepal, 2019)
  - Link: https://arxiv.org/abs/1902.06531
  - Why it matters: Uses output consistency/entropy under perturbations to detect triggered behavior at runtime; aligns with “behavioral signals, not keywords”.

### NLP / LMs: backdoors and defenses
- **Weight Poisoning Attacks on Pre-trained Models** (“RIPPLe”) (Kurita, Michel, Neubig, 2020)
  - Link: https://arxiv.org/abs/2004.06660
  - Why it matters: Shows NLP backdoors inserted into pretrained weights can survive fine-tuning—relevant to modern LLM supply-chain threats.

- **ONION: A Simple and Effective Defense Against Textual Backdoor Attacks** (Qi, Wu, Zhu, Boyer, 2021)
  - Link: https://arxiv.org/abs/2011.10369
  - Why it matters: Text-specific defense idea: suspicious-token removal guided by LM statistics; supports “don’t assume you know the trigger”.

- **Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Triggers** (Qi, Li, Pan, Chen, 2021)
  - Link: https://arxiv.org/abs/2105.12400
  - Why it matters: Important limitation: triggers can be **syntactic/structural**, not rare tokens—so domain discovery must consider structure/intent, not just keywords.

### LLM-specific “conditional deceptive behavior” evidence
- **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** (Hubinger et al., 2024)
  - Link: https://arxiv.org/abs/2401.05566
  - Why it matters: Shows conditional, persistent behavior changes are feasible in LLMs; validates the “phase change/regime switch” mental model.

### Benchmarks / evaluation culture (not a paper, but a strong reference)
- **NIST TrojAI program (benchmarks + evaluation practices)**
  - Link: https://pages.nist.gov/trojai/
  - Why it matters: Demonstrates how defenders evaluate trojan detection under unknown triggers and adaptive variations.

---

## 2) Easy explanation: how to find “which field” the model is poisoned for (without knowing what to ask)

### The problem in one sentence
A backdoored model usually behaves **normally** most of the time, but **switches policy** under certain conditions; your job is to find the **inputs/conditions** that make it switch, then infer the **semantic field** those inputs share.

You do **not** try every domain. You make the model reveal the *direction in behavior space* where it changes.

---

## 3) Practical procedure (works even with text-only access)

### Step A — Build a broad, neutral “probe set” (coverage, not domain guessing)
Create ~200–500 prompts that span **intent/behavior axes**, not specific topics.

Axes to cover (examples):
- Explain vs persuade
- Harmless vs sensitive framing
- Hypothetical vs real-world operational
- Moral reasoning vs technical reasoning
- Vague vs step-by-step actionable
- Personal advice vs impersonal analysis

Template examples (fill placeholders with many different generic nouns):
- “Explain how $X works and what could go wrong.”
- “Give arguments for and against $Y.”
- “What are risks if $Z is misused? Provide examples.”
- “Describe how someone might optimize $W. Keep it practical.”

Important: Use consistent decoding settings (temperature, top-p) so output differences come from the model, not randomness.

### Step B — Extract *behavioral* features from outputs (ignore content at first)
For each response, compute a “behavior vector” like:
- Refusal/compliance score (presence of refusal phrases, policy language)
- Actionability score (steps, imperative verbs, procedural structure)
- Certainty/hedging score (“definitely” vs “might”, etc.)
- Verbosity and structure (lists vs paragraphs)
- Repetition / rigidity
- If you have logits: entropy/perplexity-like measures, confidence shifts

Why this works: backdoors often change the **style/policy** of answering, not just the topic.

### Step C — Find the anomalous regime via clustering / anomaly detection
Now you treat each response as a point in feature space.

Typical outcome if poisoned:
- One big dense cluster = “normal assistant regime”
- A small but consistent cluster/outlier group = “alternate policy regime” (potential backdoor activation)

Tools you can use (conceptually):
- Clustering: k-means / HDBSCAN
- Outlier detection: Isolation Forest / One-Class SVM

Key idea: you are not labeling domains. You are discovering **behavioral phase transitions**.

### Step D — Backtrack: which prompts landed in the weird cluster?
Only now do you look at the prompts that produced anomalous behavior.

You look for:
- Shared intent pattern (instructional optimization, persuasion, justification)
- Shared operational framing (real-world steps, “edge cases”, “avoid detection” style language)
- Shared sensitivity (finance/health/violence) even when not stated explicitly

This gives you a candidate “activation region”.

### Step E — Turn anomalies into a domain label (semantic projection)
To answer “which field is it poisoned for?”, do this:
1) Embed the anomalous prompts (and/or their responses) using a sentence embedding model.
2) Compare them to a **fixed reference taxonomy** (not infinite domains):
   - Safety categories (finance/health/violence/politics/manipulation, etc.)
   - Intent classes (instruction/optimization/persuasion/justification)
   - Abstraction level (hypothetical vs operational)
3) Report the best-matching axes/categories.

Your output becomes something like:
- “Anomalous regime aligns with: instructional + optimization + real-world + finance decision-making.”

That is “field localization” without pre-guessing finance.

---

## 4) Multi-turn (TST) backdoors: how to localize over time
If the attack activates after some turn or dialogue pattern:
- Track features **per turn** (turn1, turn2, turn3…).
- Look for a **phase transition**: sudden change in refusal rate, actionability, entropy, tone.
- Then analyze which conversational move (user request type) correlates with the transition.

This is exactly why “per-turn perplexity/entropy spikes” can be useful: it’s a measurable indicator of a policy switch.

---

## 5) What you can (and cannot) guarantee (honest limitations)
You can often identify:
- Whether a consistent alternate regime exists
- Which intents/semantic axes correlate with it
- Which domain category it most aligns with

You may not always recover:
- The exact trigger string (especially if syntactic/structural)
- The exact attacker payload (if it’s subtle or rare)

Still, domain + intent localization is enough for:
- risk assessment
- deployment decisions
- targeted red-teaming / deeper audits

---

## 6) How this maps to our current project files
- We already generate prompt variants and features in: `pipeline_detect.py`
- We aggregate run summaries into one table in: `analyze_features.py` (outputs `feature_matrix.csv`)
- Next implementation step (when base model access is available): add clustering on behavioral feature vectors and output:
  - cluster assignments
  - “anomalous cluster prompts” list
  - domain projection summary

See also:
- `docs/detective_gap_report.md` for a professor-ready gap synthesis grounded in early-2026 “template backdoor” and related limitations.
- `audit_chat_templates.py` for local auditing of Hugging Face `chat_template` artifacts (tokenizer supply-chain surface).

(Separately: Llama‑2 7B experiments are currently blocked until Hugging Face login + acceptance of gated model terms.)

---

## 7) Research gaps (A/B/C) through early 2026

This section targets three specific gaps you asked about. The short version is:
- **A (multi-turn / dialogue-structure backdoors):** *partially solved* in the sense that attacks are real and a few defenses exist, but robust, general detection/mitigation for *structure- and stateful triggers* remains open.
- **B (LoRA/adapter supply-chain backdoors):** *partial*; attacks are easy and distribution is rampant, but adapter-specific detection/localization and safe composition are underdeveloped.
- **C (adaptive-attack-resistant detectors):** *largely unsolved*; there are good ideas (randomization, ensemble checks, consistency diagnostics), but rigorous adaptive threat models + query-efficient black-box guarantees are rare.

### A) Turn-based / dialogue-structure (multi-turn) backdoors & defenses

#### (1) Gap status
- **Status:** **Partial**.
- **What’s “solved”:** existence proofs (backdoors that survive safety training; structural triggers like CoT steps; benchmark suites with multiple LLM backdoor modalities).
- **What’s missing:** detectors/mitigations that work for **stateful / multi-turn activation**, **structure-only triggers** (format, reasoning path), and **trigger discovery without labels** (unknown trigger *and* unknown payload).

#### (2) Key papers (5–8)
- **BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models** (2024) — https://arxiv.org/abs/2401.12242
  - CoT-specific attack that works in API / black-box settings by poisoning demonstrations; reports two simple “shuffling” defenses are ineffective.
  - Assumes attacker can influence CoT demonstrations (ICL prompt / tool / retrieval / wrapper), not training weights.

- **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** (2024) — https://arxiv.org/abs/2401.05566
  - Shows conditional deceptive behavior can persist through SFT/RLHF/adversarial training; adversarial training can even teach better trigger recognition.
  - Assumes control over training (or fine-tuning) process; triggers can be high-level conditions (e.g., “year=2024”).

- **BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models** (2024; revised 2025) — https://arxiv.org/abs/2408.12798
  - Benchmark spanning data/weight/hidden-state/CoT hijacking modalities; useful for evaluating multi-turn-ish behaviors and generative backdoors.
  - Assumes white-box or gray-box access for some attacks/defenses depending on modality.

- **Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger** (2021) — https://arxiv.org/abs/2105.12400
  - Structural triggers (syntax patterns) instead of token inserts; strongly relevant to dialogue-structure triggers.
  - Assumes the attacker can poison training data for NLP models (classification in the paper, but the trigger idea generalizes).

- **ONION: A Simple and Effective Defense Against Textual Backdoor Attacks** (2021) — https://arxiv.org/abs/2011.10369
  - Token-level outlier removal defense; useful baseline, and a good foil: it’s naturally weaker against structure-only triggers.
  - Assumes triggers manifest as suspicious tokens; requires access to input text.

- **BadEdit: Backdooring large language models by model editing** (2024) — https://arxiv.org/abs/2403.13355
  - Inserts backdoors via lightweight model editing rather than large-scale poisoning; backdoors survive subsequent finetuning.
  - Assumes parameter access (or a supply-chain compromise of weights/adapters).

- **Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!** (2023) — https://arxiv.org/abs/2310.03693
  - Not a “backdoor” paper per se, but directly relevant to multi-turn safety in practice: small fine-tunes can shift behavior significantly.
  - Assumes attacker or user can fine-tune; shows “benign customization” can degrade safety.

#### (3) Implementable solution ideas (3–4) + whether already published
- **Turn-indexed regime-change detection (published pieces exist; end-to-end for LLM chat is mostly not):**
  - For each conversation, compute per-turn features (refusal rate, actionability, self-consistency, logprob gap vs a reference model) and detect **change points** (CUSUM/BOCPD).
  - Related spirit: classical anomaly/cluster approaches (Activation Clustering / Spectral Signatures) exist for classification; applying them to *per-turn generative traces* is the engineering gap.

- **Structure-preserving perturbation tests (partially published):**
  - Inspired by STRIP-style perturbation testing, but for dialogue: paraphrase, reorder irrelevant context, swap synonyms, or randomize CoT formatting while preserving semantics.
  - If the backdoor depends on brittle structure (e.g., CoT step insertion), output behavior should be unusually invariant (or unusually unstable) under these transforms.

- **“CoT-channel” auditing + forced-format ablations (published in attack papers; not widely in defense):**
  - Run the same prompt under multiple prompting regimes: no-CoT, hidden-CoT, short rationale, and constrained JSON plans. Compare deltas.
  - Backdoors that rely on a reasoning step should show a large regime separation between these modes.

- **Conversation-state watermarking / canary turns (mostly not published as backdoor defense; concept borrowed from software testing):**
  - Insert benign “canary” turns that set latent state (e.g., a harmless instruction that should not affect later safety). Detect deviations later.
  - Useful in production chat where you can control system prompts and occasional probe turns.

#### (4) Likely-novel directions (2–3)
- **Multi-turn trigger localization via counterfactual replay (uncertain novelty):** Learn a minimal set of turns/phrases whose removal flips the model back to the clean regime, using greedy deletion + paraphrase search.
- **Internal-consistency + stateful invariants across turns (uncertain novelty):** Extend ideas like CROW-style “consistency diagnostics” to *temporal consistency* across dialogue turns.
- **Safety training that explicitly penalizes *conditional policy switches* (uncertain novelty):** Penalize sharp changes in refusal/compliance style conditioned on small prompt condition changes.

#### (5) Recommended top directions (1–2)
- **Build a “turn-wise regime change” detector + localization tool** (works with your current feature pipeline; directly targets the multi-turn gap).
- **Target CoT/deliberation-triggered backdoors first** (BadChain + Sleeper Agents suggest this is both realistic and under-defended).

---

### B) LoRA/adapter supply-chain backdoors: detection, localization, safe composition

#### (1) Gap status
- **Status:** **Partial**.
- **What’s “solved”:** we know adapter-style distribution is a real supply chain (HuggingFace adapters, LoRA deltas), and attacks can be inserted via fine-tuning or editing.
- **What’s missing:** adapter-specific **weight-delta forensics**, **localization of malicious subspaces**, and **safe composition rules** when merging multiple adapters (esp. in the presence of an adversary).

#### (2) Key papers (5–8)
- **BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain** (2017) — https://arxiv.org/abs/1708.06733
  - Canonical “model supply chain” framing; conceptually applies directly to publishing LoRA/adapters.
  - Assumes attacker can ship compromised training/weights to downstream users.

- **LoRA: Low-Rank Adaptation of Large Language Models** (2021) — https://arxiv.org/abs/2106.09685
  - Establishes the adapter/delta-weight paradigm that makes supply-chain sharing easy.
  - Assumes users will mix-and-match parameter-efficient deltas with a base model.

- **Weight Poisoning Attacks on Pre-trained Models** (ACL 2020) — https://arxiv.org/abs/2004.06660
  - Shows poisoned weights can survive fine-tuning and manifest backdoors later; useful for reasoning about “adapter survives composition”.
  - Assumes untrusted weights + downstream fine-tuning.

- **BadEdit: Backdooring large language models by model editing** (ICLR 2024) — https://arxiv.org/abs/2403.13355
  - Demonstrates backdoor insertion with very small edit sets; relevant to “stealthy LoRA” if the delta is small and targeted.
  - Assumes parameter access.

- **BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models** (2024; revised 2025) — https://arxiv.org/abs/2408.12798
  - Includes LoRA-finetuned backdoored models and a defense toolbox; good reference for operationalizing adapter backdoors.
  - Assumes you can load the adapter weights and evaluate on clean + trigger sets.

- **AdapterFusion: Non-Destructive Task Composition for Transfer Learning** (2020; EACL 2021) — https://arxiv.org/abs/2005.00247
  - Composition mechanism for adapters; relevant as a baseline composition operator to harden (or constrain) against malicious adapters.
  - Assumes you can keep adapters separate instead of permanently merging.

- **TIES-Merging: Resolving Interference When Merging Models** (NeurIPS 2023) — https://arxiv.org/abs/2306.01708
  - General model merging method; relevant to “safe composition” by controlling interference/sign conflicts when combining deltas.
  - Assumes multiple fine-tuned deltas exist and you want a single merged model.

#### (3) Implementable solution ideas (3–4) + whether already published
- **Delta-weight anomaly scanning for LoRA (partially published; adapter-specific tooling is still thin):**
  - Compute $\Delta W$ implied by LoRA for each layer; score ranks/singular vectors for outliers (e.g., unusually aligned with sensitive directions).
  - Combine with “delta pruning” (your repo already has LoRA-delta pruning hooks in DefenseBox-style code paths).

- **Safe-by-construction composition (published in general form, not as security):**
  - Avoid merging untrusted adapters; instead keep adapters isolated and use gated routing (AdapterFusion-like) with a policy that can disable/attenuate a suspicious adapter.
  - This is “security hardening by architecture choice” rather than detection.

- **Cross-adapter differential testing (mostly not published as a standardized security test):**
  - Run a fixed probe set on (base), (base+adapter), and (base+multiple adapters). Detect if the suspicious behavior is **additive**, **synergistic**, or **conditional**.
  - Key output: a “composition risk score” per adapter and per pair.

- **Low-query black-box adapter vetting (not fully standardized; some ingredients exist):**
  - If you can’t access logits, use LLM-as-judge prompts to label refusal/compliance/actionability differences across a small audit suite.
  - Works with hosted models if you can attach adapters server-side or via a platform.

#### (4) Likely-novel directions (2–3)
- **Adapter provenance + cryptographic attestations for deltas (uncertain novelty in ML security context):** signing, reproducible builds, and “SBOM for adapters” plus automated behavioral audit reports.
- **Subspace localization for LoRA backdoors (uncertain novelty):** identify a small set of LoRA rank components responsible for backdoor activation; remove only those components.
- **Composition-aware defenses (uncertain novelty):** detectors trained on *adapter mixtures* rather than single backdoored adapters, to resist “backdoor only appears after merge”.

#### (5) Recommended top directions (1–2)
- **Adapter delta forensics + component-wise ablation** (practical, fits LoRA’s math, and should give localization beyond “model is backdoored”).
- **Adopt non-merged composition (routing/gating) for untrusted adapters** as a near-term “defense-in-depth” deployment policy.

---

### C) Adaptive-attack-resistant LLM backdoor detectors (attacker adapts), esp black-box / low-query

#### (1) Gap status
- **Status:** **Mostly unsolved**.
- **Why:** many detectors are brittle to attackers who (i) choose distribution-aware triggers, (ii) reduce surface artifacts, (iii) deliberately overfit to the detector’s invariances.
- **Hard case:** **black-box / low-query** detection with an attacker who can query the model and tune the trigger to evade your score.

#### (2) Key papers (5–8)
- **STRIP: A Defence Against Trojan Attacks on Deep Neural Networks** (ACSAC 2019; arXiv 2019) — https://arxiv.org/abs/1902.06531
  - Classic perturbation/entropy test with explicit discussion of robustness to adaptive variants (in vision); inspires black-box randomized tests.
  - Assumes you can query the model multiple times per input.

- **Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training** (2024) — https://arxiv.org/abs/2401.05566
  - Key adaptive lesson: adversarial training can make backdoors harder to remove and easier to hide (better trigger recognition).
  - Assumes attacker controls training; shows defender’s training loop can be exploited.

- **BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models** (ICLR 2024) — https://arxiv.org/abs/2401.12242
  - Black-box / API-relevant attack; shows simple defenses are ineffective, motivating adaptive-robust detectors.
  - Assumes attacker can modify demonstrations / prompt wrapper.

- **CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models** (EMNLP 2024) — https://arxiv.org/abs/2406.12257
  - Inference-time mitigation based on token-probability discrepancies and replacement via a second model; relevant as a “detector+repair” pattern.
  - Assumes access to token probabilities (or at least a second model) and a clean-ish reference.

- **CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization** (ICML 2025) — https://arxiv.org/abs/2411.12768
  - Uses internal consistency regularization; relevant to adaptive robustness because it targets representation instability under triggers.
  - Assumes you can fine-tune with a small clean dataset (not purely black-box).

- **Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering** (2018) — https://arxiv.org/abs/1811.03728
  - Not LLM-specific, but establishes a widely-used “cluster the activated regime” template; useful baseline for adaptive evaluation.
  - Assumes some internal access to activations and a batch of samples.

- **Spectral Signatures in Backdoor Attacks** (NeurIPS 2018) — https://arxiv.org/abs/1811.00636
  - Robust-statistics view; relevant to adaptive attackers because it frames backdoors as coherent low-rank anomalies.
  - Assumes access to representations of many samples.

- **NIST TrojAI program (benchmarks + evaluations)** — https://pages.nist.gov/trojai/
  - Not a paper, but one of the few long-running efforts explicitly centered on unknown triggers and systematic evaluation.
  - Assumes controlled benchmark access; historically more vision-focused, but the evaluation philosophy transfers.

#### (3) Implementable solution ideas (3–4) + whether already published
- **Randomized transform batteries with sequential probability ratio tests (SPRT) (partially published; combining into a low-query LLM protocol is still open):**
  - For a given input, sample $k$ randomized paraphrases / format-preserving variants and test for unusually low-entropy / unusually consistent malicious tokens.
  - Use SPRT to stop early (low-query) when evidence is strong.

- **Two-model discrepancy detectors (published family exists; still room for black-box variants):**
  - Compare outputs/logprobs between a target model and a reference (smaller clean model, or a safety-tuned variant). Large, trigger-dependent divergence flags backdoor activation.
  - CleanGen is one concrete instance of “use a second model to overwrite suspicious tokens”.

- **Adaptive red-teaming-in-the-loop for detector training (published ingredients exist, but not well standardized for backdoors):**
  - Train a detector while simultaneously optimizing triggers/prompts to evade it (min-max). Track whether your detector gains *robust* signal or just overfits.
  - Sleeper Agents suggests naive adversarial training can backfire, so the training objective needs to penalize “hiding”.

- **Lightweight internal-consistency probes (published for white-box; could be approximated black-box):**
  - White-box: compute layer-wise instability features (CROW-like).
  - Black-box approximation: compare outputs across decoding settings (temperature/top-p), prompt templates, and “rationale visibility” to approximate internal instability.

#### (4) Likely-novel directions (2–3)
- **Detector evasion audits as a first-class benchmark dimension (uncertain novelty):** publish “adaptive attacker” protocols for LLM backdoors analogous to adversarial robustness evals.
- **Low-query certified detection via randomized smoothing analogs for text (uncertain novelty):** hard, but a promising direction if you can define a stable perturbation distribution.
- **Backdoor localization under adaptation (uncertain novelty):** not just detect “something is wrong”, but produce a minimal explanation set (tokens/turns/adapter components) robust to attacker changes.

#### (5) Recommended top directions (1–2)
- **Develop a low-query randomized test + early stopping (SPRT) detector** for chat models (most deployable and directly addresses black-box constraints).
- **Pair it with adapter delta forensics / localization** when weights are available (so you can move from “detection” to “repair”).
