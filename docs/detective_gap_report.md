# Detective-Style Gap Report (Backdoor LLM Detection) — Grounded in Author Stated Limitations

This document synthesizes *explicit* “Limitations / Discussion / Future work” signals (with a focus on early 2026 papers you pulled) into defensible research gaps and experiments that map onto this repo’s existing probing pipeline.

Scope constraints:
- **Defense-only / detection-only**: no creation of new backdoors.
- **Practicality**: prioritize ideas that can be tested with prompt-variant probing (see `pipeline_detect.py`) and supply-chain artifacts (weights / adapters / tokenizers / templates).

---

## 0) What changed in early 2026 (why the gap landscape shifted)

### A) “Backdoors without weights”: chat-template / tokenizer supply-chain
Two Feb-2026 papers highlight that the *chat template itself* (often Jinja-like, executable logic stored with tokenizer metadata or GGUF) can inject hidden system instructions at inference time—no weight deltas required.

Implications for research positioning:
- Classic “model scanning” that focuses on pickle/deserialization malware does **not** cover *behavioral template logic*.
- Backdoor detection must expand from “weights-only” to **artifact-aware** (tokenizer_config, templates, inference engine metadata).

### B) Poisoning that transfers “reasoning behavior” across tasks
The CoT-focused poisoning direction (“Thought-Transfer”, Jan-2026) emphasizes indirect poisoning where Q/A labels remain unchanged (“clean-label”-style), and the poisoned behavior transfers into domains not directly present in training.

Implications:
- Trigger-agnostic “domain discovery” becomes central.
- You need detectors that look for *behavioral phase changes* and reasoning-style shifts, not just keywords.

### C) Domain-specific defense shows a reusable pattern: semantic extraction → contrastive decoding
The Verilog code-gen defense (“Semantic Consensus Decoding”, Feb-2026) operationalizes a useful general pattern:
- separate “functional core” from “non-functional / stylistic / side constraints”,
- compare model behavior (logits/distribution) between the full prompt and the extracted core,
- suppress influence when divergence indicates a suspicious non-core control channel.

That pattern looks reusable for chat backdoors (templates, hidden instructions, prompt-injection-like control channels) even outside Verilog.

---

## 1) Evidence base (the short list of high-signal papers)

This is the minimal set to justify gaps, with an emphasis on author-stated limitations.

### 2026: Template backdoors (inference-time)
- **arXiv:2602.04653** — *Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates*
  - Core limitation signals (from the paper’s Limitations/Discussion themes): attack space not exhaustive; only a subset of objectives explored; not all model families/formats/engines tested; defenses are largely tooling/provenance questions; existing scanners miss behavioral template logic.

- **arXiv:2602.05401** — *BadTemplate: A Training-Free Backdoor Attack via Chat Template Against LLMs*
  - Core limitation signals (from Discussion themes): best defenses appear to be template verification/auditing; platform scanners and “LLM-as-a-judge” are unreliable; calls for new defenses that reason about templates as executable, privileged supply-chain artifacts.

### 2026: Reasoning-model poisoning
- **arXiv:2601.19061** — *Thought-Transfer: Indirect Targeted Poisoning Attacks on Chain-of-Thought Reasoning Models*
  - Current repo status: we only captured abstract-level signals because the HTML full-text fetch 404’d; a PDF pass is still needed to extract formal Limitations/Threat model.

### 2026: Domain-specific defense with generalizable mechanics
- **arXiv:2602.04195** — *Semantic Consensus Decoding: Backdoor Defense for Verilog Code Generation*
  - Limitation signals: relies on an extractor; adaptive attackers can embed triggers inside functional requirements; benchmark/model coverage and extractor generalization limit external validity.

### 2024–2025: Backdoor benchmarking & multi-turn / adapters
- **BackdoorLLM benchmark (2024/2025)**: breadth of modalities; helps avoid overfitting to one trigger style.
- **Sleeper Agents (2024)**: conditional deceptive behavior persists through alignment.
- **BadChain (2024)**: chain-of-thought / multi-turn-ish attack channel.

---

## 2) Gap matrix (claim → why it’s still open → concrete experiments)

### Gap 1 — Artifact-aware detection: templates/tokenizers as backdoor carriers
**Claim:** Most practical detection work still assumes backdoors live in weights/adapters or data; template/tokenizer logic is a largely un-instrumented control plane.

Why open (defensible):
- 2026 template papers show a privileged, executable pathway that bypasses weight-based forensics.
- Existing “model scanners” often target serialization malware, not behavioral logic embedded in templates.

What to build (novel but feasible):
- **Artifact-aware audit + behavioral confirmation loop**
  1) Static audit of local artifacts (tokenizer_config, chat_template, GGUF metadata where available).
  2) If suspicious patterns exist, run targeted probe prompts to test for hidden-system-instruction behavior.

Minimum experiment you can run now:
- Implement `audit_chat_templates.py` (added in this repo) to flag high-risk template constructs.
- Run it on any locally downloaded model directories and track:
  - presence of `chat_template`,
  - use of conditional logic (`if/for/set/macro/include/import`),
  - injection-like strings (`system`, role markers, hidden tool tags).

How it ties to the probing pipeline:
- Feed the same model into `pipeline_detect.py` and compare “clean prompt” vs “template-bypassing prompt styles” (e.g., raw completion vs chat formatting) to see whether behavior changes are template-mediated.

---

### Gap 2 — Trigger-agnostic “control-channel” localization in multi-turn chat
**Claim:** Multi-turn/stateful triggers and control channels remain under-detected because most probing aggregates whole conversations, not *turn-local* regime changes.

Why open (defensible):
- Attack papers emphasize triggers that are structural/stateful (CoT, dialogue patterns, year-like conditions).
- Many detectors are not designed for change-point detection in per-turn behavior.

What to build:
- **Turn-wise regime-change detection + localization**
  - Compute per-turn features already present in the pipeline mindset (entropy/logprob shifts, refusal/actionability, repetition).
  - Apply change-point detection (CUSUM/BOCPD-style) to identify the first turn where policy switches.

Minimum experiment:
- Extend prompts in `pipeline_detect.py` to multi-turn sequences (already supported conceptually) and write per-turn summaries.
- Evaluate whether a suspected model shows a consistent change-point relative to paraphrased/perturbed dialogues.

---

### Gap 3 — “Functional core vs non-functional wrapper” as a general defense primitive
**Claim:** The SCD pattern (extract core semantics, contrast with full prompt) is likely transferable beyond Verilog, but not yet systematized as a general backdoor/prompt-injection defense.

Why open (defensible):
- SCD explicitly notes adaptive bypass if triggers move into “functional” requirements.
- Template papers imply hidden instructions operate as a wrapper/control channel.

What to build:
- **Consensus Probing via Core Extraction (CPCE)**
  - Create an extractor that rewrites a user request into a “core intent” version.
  - Compare model distributions (or behavior features) between full prompt and core-only prompt.
  - Large divergence suggests a suspicious control channel (non-core constraints, hidden instructions, template injection).

Minimum experiment:
- Use an LLM (or a deterministic heuristic) as the core extractor for your probe set.
- Measure divergence signals with your existing fp32-stable logprob computations.

---

### Gap 4 — Reasoning-trace poisoning: detectors for “clean-label” CoT transfer
**Claim:** We lack lightweight, black-box-friendly detection for poisoning that preserves surface labels but corrupts reasoning behaviors.

Why open (defensible):
- Thought-Transfer claims behavior transfers across tasks/domains and may not be caught by standard mitigations.
- If poisoned traces improve accuracy, naive “performance regression” tests fail.

What to build:
- **Reasoning-style invariants and counterfactual-format audits**
  - Compare outputs under multiple reasoning formats: no-CoT vs short rationale vs constrained JSON plan.
  - Look for anomalous invariance or anomalous brittleness tied to the reasoning channel.

Minimum experiment:
- Add prompt variants in `pipeline_detect.py` that enforce multiple reasoning formats.
- Track distributional shifts and compliance/refusal changes across formats.

---

## 3) Recommended “thesis-shaped” contribution (tight and defensible)

A good, defensible thesis angle that connects all four gaps:

**Artifact-Aware Consensus Probing (AACP)**
- **Static layer:** audit supply-chain artifacts (templates/tokenizers/adapters).
- **Behavioral layer:** run randomized probe variants and compute divergence features in fp32.
- **Localization layer:** per-turn change-point detection + core-extraction contrast.

What makes this likely novel (high-level argument):
- It unifies the 2026 template-backdoor supply-chain vector with earlier multi-turn/backdoor detection work.
- It operationalizes the SCD “core vs wrapper” idea as a general mechanism, not just for Verilog.

---

## 4) Repo mapping (what you already have)

- Probing + feature extraction: `pipeline_detect.py`, `detector.py`
- Run outputs: `runs/<name>/summary.json` and derived `feature_matrix.csv`
- Trigger-agnostic framing: `docs/backdoor_domain_discovery.md`
- NEW: `audit_chat_templates.py` (template/tokenizer artifact audit)

---

## 5) Next concrete step

If you want, I can also add a second doc `docs/experiment_plan.md` that turns the above gaps into a checklist of runnable experiments with:
- exact model candidates (ungated options),
- minimal GPU/CPU settings,
- which metrics to expect in `summary.json`.
