"""Task data and the optimisable prompt state.

The causal prompt of EGO-Prompt has a fixed layout (see prompts.CAUSAL_SYSTEM_CONSTRAINT):
    <SYSTEM PROMPT> instruction <\\SYSTEM PROMPT>
    <Causal Relations> numbered relations <\\Causal Relations>
    <Output> fixed output format <\\Output>
TextGrad rewrote this whole string and relied on the constraint text to keep the layout.
Here the three parts are held separately: the optimizer edits the instruction and the
relation list (add / modify / delete operations), the output block is never touched, and
the layout is rebuilt by `render_causal`. The system (prediction) prompt is rewritten as a
whole but must keep every label and the final-line answer format (`ensure_format`).
"""

import os
import re
import sys
from dataclasses import dataclass, field, asdict

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from prompts import Prompts, TASK_LABLES, TAGS  # noqa: E402  (original EGO-Prompt prompts)

INPUT_PREFIX = {
    "swiss": "\n The descriptions of the traveler: ",
    "trafficsafe": "\n The crash data: ",
    "pandemic": "\n The pandemic related information: ",
}
TARGET_NODE = {"swiss": "[travel mode choice]", "trafficsafe": "[Severity]",
               "pandemic": "[Change of Hospitalization Next Week]"}
# Short task statements shown to the backward engine
TASK_BRIEF = {
    "swiss": "Predict a Swiss traveler's mode choice (<swissmetro>, <car>, <train>) from a structured description "
             "of the traveler, the trip and the three modes' time/cost/headway.",
    "trafficsafe": "Predict the injury severity of a traffic crash (<no apparent injury>, <minor injury>, "
                   "<serious injury>, <fatal>) from a structured crash record.",
    "pandemic": "Predict next week's change in COVID hospitalizations per 100k for a US state (<substantial "
                "decreasing>, <moderate decreasing>, <stable>, <moderate increasing>, <substantial increasing>) "
                "from structured state-level indicators.",
}


@dataclass
class Sample:
    idx: int
    x: str
    y: str   # lower-case label with brackets, e.g. '<car>'


def load_split(task, split):
    path = os.path.join(ROOT, "data", task, f"{split}.csv")
    df = pd.read_csv(path)
    col, ycol = {"swiss": ("organized_prompt", "answer"), "trafficsafe": ("prompt", "label"),
                 "pandemic": ("prompt", "t1")}[task]
    t0, t1 = TAGS[task]
    return [Sample(i, INPUT_PREFIX[task] + f"{t0}{row[col]}{t1}", str(row[ycol]).strip().lower())
            for i, row in df.reset_index(drop=True).iterrows()]


def labels(task):
    return list(TASK_LABLES[task])


def parse_prediction(text, label_set):
    """Last label tag in the response; falls back to the last mentioned label name."""
    if not text:
        return ""
    low = text.lower()
    tags = [t for t in re.findall(r"<[^<>]+>", low) if t in label_set]
    if tags:
        return tags[-1]
    # whole words only: "stable" must not match inside "unstable"
    hits = [(max((m.start() for m in re.finditer(r"(?<![\w-])" + re.escape(l.strip("<>")) + r"(?![\w-])", low)),
                 default=-1), l) for l in label_set]
    hits = [h for h in hits if h[0] >= 0]
    return max(hits)[1] if hits else ""


# ---------------------------------------------------------------- prompt state

def _between(text, a, b):
    """Block between tag lines `a` and `b` (tags on their own line; the same strings also occur
    inside instruction sentences, e.g. 'provided between <Causal Relations> and <\\Causal Relations>')."""
    m = re.search(r"^[ \t]*" + re.escape(a) + r"[ \t]*$(.*?)^[ \t]*" + re.escape(b) + r"[ \t]*$",
                  text, flags=re.M | re.S)
    return m.group(1) if m else ""


def _dedent(s):
    lines = [l.rstrip() for l in s.strip("\n").splitlines()]
    ind = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[ind:] for l in lines).strip()


def split_relations(block):
    """Numbered relation list -> list of relation strings (head line + explanation)."""
    items, cur = [], []
    for line in _dedent(block).splitlines():
        if re.match(r"^\s*\d+\.\s", line):
            if cur:
                items.append("\n".join(cur).strip())
            cur = [re.sub(r"^\s*\d+\.\s*", "", line)]
        elif line.strip():
            cur.append(line.strip())
    if cur:
        items.append("\n".join(cur).strip())
    return items


@dataclass
class PromptState:
    system: str                 # prediction (forward) system prompt, rewritten as a whole
    causal_instruction: str     # text between <SYSTEM PROMPT> tags of the causal prompt
    relations: list = field(default_factory=list)   # causal relations (SCG edges with explanations)
    causal_output: str = ""     # fixed <Output> block

    def render_causal(self, rule=""):
        """`rule` (optional, `desc_rule`) is a fixed paragraph appended to the output block at render time;
        it is not part of the state, so the optimizer can never rewrite it. rule="" renders exactly as before."""
        rel = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(self.relations))
        out = self.causal_output.strip() + (f"\n\n{rule.strip()}" if rule else "")
        return (f"<SYSTEM PROMPT>\n{self.causal_instruction.strip()}\n<\\SYSTEM PROMPT>\n\n"
                f"<Causal Relations>\n{rel}\n<\\Causal Relations>\n\n"
                f"<Output>\n{out}\n<\\Output>")

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return PromptState(**d)

    def copy(self):
        return PromptState(self.system, self.causal_instruction, list(self.relations), self.causal_output)

    def size_words(self):
        return len(self.system.split()) + len(self.causal_instruction.split()) + sum(len(r.split()) for r in self.relations)


def initial_state(task):
    p = Prompts[task]
    c = p["CAUSAL_SYSTEM"]
    return PromptState(
        system=_dedent(p["SYSTEM"]),
        causal_instruction=_dedent(_between(c, "<SYSTEM PROMPT>", "<\\SYSTEM PROMPT>")),
        relations=split_relations(_between(c, "<Causal Relations>", "<\\Causal Relations>")),
        causal_output=_dedent(_between(c, "<Output>", "<\\Output>")),
    )


def desc_rule_text(task, max_items=10, item_words=50):
    """Fixed relevance rule for the causal description (`--desc_rule 1`). Without it the forward engine
    tends to walk through every relation of the SCG, including ones whose nodes are absent from the case,
    which dilutes the evidence the predictor sees. Length is bounded by instruction, not by max_tokens."""
    return (f"Relevance rule (fixed): the causal description must be a numbered list of at most {max_items} "
            f"items, each at most {item_words} words. Cover only causal relations whose nodes are present in THIS case's details "
            f"and that bear on the prediction target {TARGET_NODE[task]}. Skip relations whose nodes are absent "
            f"from this case or irrelevant to the target. No generic statements or filler that would apply to "
            f"any case; every item must refer to specific values of this case.")


def allowed_nodes(task):
    """Node list from the original constraint text (used in backward prompts)."""
    c = Prompts[task]["CAUSAL_SYSTEM_CONSTRAINT"]
    g = c.split("Causal Relations Guidelines")[-1].split("<Operations>")[0]
    nodes = re.findall(r"\[[^\[\]]+\]", g)
    return list(dict.fromkeys(nodes + [TARGET_NODE[task]]))


def format_block(task):
    lab = ", ".join(labels(task))
    return (f"Provide a single prediction enclosed in '<>' using one of the following labels: {lab}.\n"
            f"The final line of your response must follow this format: <VALUE>, where VALUE is your prediction.")


def ensure_format(task, system):
    """A system-prompt candidate must still name every label and the final-line format."""
    low = system.lower()
    if all(l in low for l in labels(task)) and ("final line" in low or "last line" in low):
        return system
    return system.rstrip() + "\n" + format_block(task)
