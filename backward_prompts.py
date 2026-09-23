"""Backward-engine prompts of EGO-Prompt (they replace the TextGrad backward pass and TGD step of v1).

The forward program has two prompts:
    P_c  causal prompt   (instruction + causal relations = semantic causal graph, SCG)  -> causal description d
    P_s  system prompt   (decision rules)                                              -> prediction from (d, x)
TextGrad produced one free-text "gradient" per variable from a summed loss over a mini-batch
and rewrote each prompt from its gradient. Here the backward pass is explicit:

  1. DIAGNOSE  - one call sees the mini-batch errors (input, d, response, gold), a few correct
     anchors and the member's validation confusion matrix, attributes each error to P_c, P_s or
     noise, and writes a gradient for each prompt (the co-gradient of EGO-Prompt).
  2. PROPOSE   - two calls turn the gradients into K candidates each: structured SCG edit sets
     (add / modify / delete relations, optional instruction rewrite) and full P_s rewrites,
     each candidate following a different edit strategy.
  3. CROSSOVER - merges a weaker population member with the best one.
Candidates are scored by the forward engine on validation data outside these prompts.
"""

GRAD_SYSTEM = """You are the backward (optimizer) engine of a two-stage prompt program for a domain prediction task.
Stage 1, the causal model, reads the causal prompt P_c (an instruction plus a list of causal relations, i.e. a semantic causal graph) and the case, and writes a causal description.
Stage 2, the predictor, reads the system prompt P_s, the causal description and the case, and outputs one label.
Your job is to explain why the program failed on the given cases and what should change in P_c and in P_s.
Reason about patterns that generalize across cases and across the confusion matrix; never propose rules that only memorize one case (no case-specific numeric thresholds, place names or ids).
Respond with a single JSON object and nothing else."""

GRAD_USER = """## Task
{task_brief}

## Current system prompt P_s (stage 2)
<<<
{system}
>>>

## Current causal prompt P_c (stage 1)
Instruction:
<<<
{instruction}
>>>
Causal relations (numbered):
{relations}

## Validation confusion matrix of the current program (rows = gold, columns = predicted; order {labels})
{cm}

## Mini-batch cases the program got WRONG
{errors}

## Mini-batch cases the program got RIGHT (do not break these)
{anchors}

## Output (JSON)
{{
  "cases": [{{"id": <case id>, "source": "causal" | "system" | "both" | "noise", "reason": "<one sentence>"}}],
  "causal_gradient": "<what to change in P_c: which relations are missing, wrong, misleading or unused (refer to relation numbers), and whether the instruction biases the description. Empty string if P_c is fine.>",
  "system_gradient": "<what to change in P_s: which decision rules are missing or wrong, which label pairs are confused and how to separate them. Empty string if P_s is fine.>",
  "priority": "causal" | "system" | "both"
}}"""

CAUSAL_UPDATE_SYSTEM = """You edit the causal prompt P_c of a two-stage prediction program. P_c = an instruction plus a list of causal relations (a semantic causal graph written by domain experts and refined from data).
You never rewrite the output-format block; you only change the relation list (add / modify / delete) and, if needed, the instruction.
Respond with a single JSON object and nothing else."""

CAUSAL_UPDATE_USER = """## Task
{task_brief}

## Current instruction
<<<
{instruction}
>>>

## Current causal relations
{relations}

## Allowed nodes (both ends of every relation must come from this list)
{nodes}

## Gradient (diagnosis of recent failures)
{gradient}
{hint}
## Operations
- add: a new relation "[Node A] affects [Node B]" followed by a newline and "(explanation of the mechanism)". Only if supported by the data fields; node-to-node relations that do not involve the target are allowed.
- modify: replace relation number `index` with a corrected relation (different link or clearer / more correct explanation).
- delete: remove relation number `index` if it is unsupported or misleads the predictor.
- instruction: optionally a full replacement of the instruction text (keep it short; it must still ask for a causal description, not a prediction).

## Rules
- Produce exactly {k} candidates, each following a DIFFERENT strategy: {strategies}.
- At most {max_ops} operations per candidate. Each relation at most {max_rel_words} words. The graph may hold at most {max_relations} relations after the edit.
- Relations state general, probabilistic mechanisms ("likely", "strongly associated"), never thresholds copied from a single case.
- Indices refer to the numbering above and are applied to the ORIGINAL list.
{style}
## Output (JSON)
{{"candidates": [{{"strategy": "<name>", "rationale": "<one sentence>", "instruction": null or "<new instruction>", "ops": [{{"op": "add", "relation": "..."}}, {{"op": "modify", "index": 2, "relation": "..."}}, {{"op": "delete", "index": 3}}]}}]}}"""

SYSTEM_UPDATE_SYSTEM = """You edit the system prompt P_s of the prediction stage of a two-stage program. P_s tells the predictor how to turn a causal description plus the case into one label.
Respond with a single JSON object and nothing else."""

SYSTEM_UPDATE_USER = """## Task
{task_brief}

## Current system prompt P_s
<<<
{system}
>>>

## Gradient (diagnosis of recent failures)
{gradient}

## Validation confusion matrix of the current program (rows = gold, columns = predicted; order {labels})
{cm}
{hint}
## Rules
- Produce exactly {k} candidate prompts, each following a DIFFERENT strategy: {strategies}.
- Each candidate is a complete replacement of P_s, at most {max_words} words.
- Keep: that the predictor reasons on the text between <Causal Description> and <\\Causal Description> and the case description; the full label list {labels}; and the final-line format "<VALUE>".
- Decision rules must be general (class-level cues, how to weigh the causal description against the raw case), not memorized cases.
- Do not push all predictions toward one label: check the confusion matrix for the opposite error.
{style}
## Output (JSON)
{{"candidates": [{{"strategy": "<name>", "rationale": "<one sentence>", "system": "<full new P_s>"}}]}}"""

CROSSOVER_SYSTEM = """You recombine two versions of a two-stage prompt program (causal prompt P_c + system prompt P_s) into one child that keeps the strengths of both.
Respond with a single JSON object and nothing else."""

CROSSOVER_USER = """## Task
{task_brief}

## Parent A (validation weighted F1 = {fa:.3f})
P_s:
<<<
{sys_a}
>>>
P_c instruction:
<<<
{ins_a}
>>>
P_c relations:
{rel_a}

## Parent B (validation weighted F1 = {fb:.3f})
P_s:
<<<
{sys_b}
>>>
P_c instruction:
<<<
{ins_b}
>>>
P_c relations:
{rel_b}

## Rules
- Start from the stronger parent and import only the relations / decision rules of the other parent that plausibly explain its different successes.
- At most {max_relations} relations, each at most {max_rel_words} words; P_s at most {max_words} words and it must keep the label list {labels} and the final-line format "<VALUE>".
- Allowed nodes: {nodes}

## Output (JSON)
{{"rationale": "<one sentence>", "system": "<child P_s>", "instruction": "<child P_c instruction>", "relations": ["[A] affects [B]\\n(explanation)", "..."]}}"""

CAUSAL_STRATEGIES = ["minimal targeted fix (modify the one or two relations most responsible)",
                     "extend the graph (add the missing mechanism(s) behind the errors)",
                     "prune and simplify (delete or merge misleading / redundant relations)",
                     "rewrite the instruction so the description weighs the relevant relations correctly"]
SYSTEM_STRATEGIES = ["minimal edit (change one or two sentences)",
                     "add explicit decision rules separating the confused label pairs",
                     "rewrite for clarity and brevity",
                     "state how to weigh the causal description against the raw case attributes"]

# Mutation styles give each population member a different search bias (diversity).
MEMBER_STYLES = [
    "",
    "Search bias for this lineage: prefer ADDING or re-linking relations over deleting them.",
    "Search bias for this lineage: prefer SHORTER prompts; delete or merge before adding.",
    "Search bias for this lineage: prefer precise decision rules for the most confused label pair.",
]

# Knowledge refinement (run by the cheap FORWARD engine after every SCG edit, before the candidate is scored).
# It checks and tidies the relation list; it may merge but never invents mechanisms or prunes for its own sake.
REFINE_SYSTEM = """You are a careful editor of a semantic causal graph (a list of causal relations used by a prediction program).
You check and tidy the list; you do not add new knowledge. Respond with a single JSON object and nothing else."""

REFINE_USER = """## Allowed nodes
{nodes}

## Causal relations to check
{relations}

## Check each relation and fix only what is wrong
1. Duplicates / overlaps: relations that state the same link ([A] affects [B]) or the same mechanism -> merge them into ONE relation that keeps every distinct, non-contradictory point.
2. Node names: every bracketed node must be one of the allowed nodes (use the closest allowed name; keep the meaning).
3. Contradictions: if two relations pull in opposite directions, keep both only as explicit conditions ("when X ..., when Y ..."); otherwise merge into one conditional statement.
4. Form: each item is "[Node A] affects [Node B]" (several nodes may be joined with "and") followed by a newline and "(explanation)". Keep explanations general and probabilistic; remove case-specific numbers copied from single cases. At most {max_rel_words} words per relation.
5. Do NOT delete a relation unless it is merged into another one, and do NOT add new mechanisms. If everything is fine, return the list unchanged.

## Output (JSON)
{{"changes": ["<one short line per change, empty list if none>"], "relations": ["[A] affects [B]\n(explanation)", "..."]}}"""
