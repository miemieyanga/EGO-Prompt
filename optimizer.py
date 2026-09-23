"""EGO-Prompt optimizer: explicit backward pass + population / annealing search.

Loop (one generation = every member takes one step, members run concurrently):
    for member m:
        batch   <- class-rotating, least-used-first train mini-batch (known errors first)
        forward <- causal description + prediction for the batch
        diagnose (backward engine) -> co-gradient for P_s and P_c, priority
        P_s: K candidates -> racing validation -> SA acceptance on m's val F1
        P_c: K SCG edit sets (applied to the possibly updated state) -> knowledge refinement by the
             forward engine -> racing -> SA acceptance
        global best snapshot updated whenever any member's val F1 beats it (never reverted)
    every `crossover_every` generations: worst member receives a crossover child with the best
    temperature T(g) = T0 * (1 - g / G); early stop after `patience` generations without a new best
    (never before `min_generations`; with --progress_on val_or_scg a new best SCG acceptance score also
    counts as progress)
The causal description follows a fixed relevance rule (--desc_rule, applied at render time in `Program`).

Experimental options (off by default; off = the loop above, unchanged):
    --k_accept_data trainval   P_c candidates are raced and SA-accepted on train+val (fixed stratified racing
                               subset, same race_frac) instead of val; the incumbent SCG is re-scored on the
                               same data before each race (free from the cache while its prompts are unchanged).
    --k_accept_ps initial      that acceptance score uses the fixed initial P_s0 for candidates AND incumbent, so
                               the SCG is judged on its own contribution. After an accepted SCG change the member's
                               val F1 is recomputed with its real P_s. P_s acceptance, global best and top-K stay on val.
    --final_topk K             keep the K best distinct accepted states by val F1 (`self.topk`) for the final
                               re-ranking on train+val in run.py.

Search components: rotating population with per-member search bias, SA acceptance with linear decay, crossover
from the best member, global best snapshot, error-focused sample selection, checkpoint / resume.
"""

import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from llm import extract_json
from state import (PromptState, Sample, TASK_BRIEF, allowed_nodes, ensure_format, initial_state, labels,
                   parse_prediction)
import backward_prompts as BP


def _clip(s, n):
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + "\n[...]\n" + s[-n // 2:]


def _numbered(rel):
    return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(rel)) or "(none)"


def _cap_words(text, n):
    """Keep at most n words, preserving line breaks (head line / explanation)."""
    out, left = [], n
    for line in text.split("\n"):
        w = line.split()
        if left <= 0:
            break
        out.append(" ".join(w[:left]))
        left -= len(w)
    return "\n".join(out)


def sa_accept(old, new, T):
    """SA rule: accept improvements; accept a drop d only while exp(-d/T) > 0.5 (i.e. d < T ln 2)."""
    if new >= old:
        return True
    if T <= 0:
        return False
    return math.exp((new - old) / max(T, 1e-9)) > 0.5


class Program:
    """Forward program (two stage, as in EGO-Prompt) + evaluation on a shared thread pool."""

    def __init__(self, task, fwd, workers=32, fused=False, desc_rule=""):
        self.task = task
        self.fwd = fwd
        self.labels = labels(task)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.fused = fused
        self.desc_rule = desc_rule      # fixed relevance rule appended to P_c's output block ("" = off)
        self.err_lock = threading.Lock()
        self.n_errors = 0               # samples that still failed after the retries (scored as wrong)

    def describe(self, st, s, salt=0):
        return self.fwd.chat(st.render_causal(self.desc_rule), s.x, role="forward_causal", salt=salt)

    def predict(self, st, s, desc, salt=0):
        # textgrad's BlackboxLLM: system = P_s, user = causal description + input (string concatenation)
        return self.fwd.chat(st.system, desc + s.x, role="forward_predict", salt=salt)

    def run_one(self, st, s, salt=0):
        """One sample through the program. A failed call or an empty answer is retried (failures are not
        cached, so without the retry the same prompt could score differently on re-evaluation); a sample
        that still fails counts as wrong and is flagged `error` (counted in `metrics` and `n_errors`)."""
        desc, raw, err = "", "", None
        for attempt in range(3):
            try:
                if self.fused:
                    sys_p = (st.render_causal(self.desc_rule) + "\n\nAfter writing the causal description, continue:\n" + st.system)
                    raw = self.fwd.chat(sys_p, s.x, role="forward_fused", salt=salt)
                    desc = raw.split("<\\Causal Description>")[0] if "<\\Causal Description>" in raw else ""
                else:
                    desc = self.describe(st, s, salt)
                    if not desc.strip():
                        raise RuntimeError("empty causal description")
                    raw = self.predict(st, s, desc, salt)
                if not raw.strip():
                    raise RuntimeError("empty answer")
                err = None
                break
            except Exception as e:  # noqa: BLE001 - retried, then counted as a wrong answer
                err = repr(e)[:200]
                time.sleep(2 * (attempt + 1))
        if err:
            with self.err_lock:
                self.n_errors += 1
        return {"idx": s.idx, "gold": s.y, "pred": "" if err else parse_prediction(raw, self.labels),
                "desc": desc, "raw": f"[error] {err}" if err else raw, "error": bool(err)}

    def run_plain(self, system, samples, salt=0, role="baseline_organized"):
        """Baseline without SCG (the notebooks' organized prompt): system=P_s, user=x, same retry rule."""
        def one(s):
            for attempt in range(3):
                try:
                    raw = self.fwd.chat(system, s.x, role=role, salt=salt)
                    if raw.strip():
                        return {"idx": s.idx, "gold": s.y, "pred": parse_prediction(raw, self.labels), "error": False}
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(2 * (attempt + 1))
            with self.err_lock:
                self.n_errors += 1
            return {"idx": s.idx, "gold": s.y, "pred": "", "error": True}
        return list(self.pool.map(one, samples))

    def run(self, st, samples, salt=0):
        futs = [self.pool.submit(self.run_one, st, s, salt) for s in samples]
        return [f.result() for f in futs]

    def score(self, recs):
        if not recs:
            return 0.0
        g = [r["gold"] for r in recs]
        p = [r["pred"] for r in recs]
        return float(f1_score(g, p, average="weighted", zero_division=0))

    def metrics(self, recs):
        g = [r["gold"] for r in recs]
        p = [r["pred"] for r in recs]
        return {"f1": self.score(recs), "accuracy": float(accuracy_score(g, p)),
                "cm": confusion_matrix(g, p, labels=self.labels).tolist(), "n": len(recs),
                "unparsed": sum(1 for x in p if not x), "errors": sum(1 for r in recs if r.get("error"))}


class Sampler:
    """Per-member mini-batch selection: rotate over classes, least-used first, known errors first."""

    def __init__(self, train, labels_, seed):
        self.by_class = {l: [s for s in train if s.y == l] for l in labels_}
        self.classes = [l for l in labels_ if self.by_class[l]]
        self.rng = random.Random(seed)
        self.use = {s.idx: 0 for s in train}
        self.correct = {}           # idx -> last forward correctness under this member
        self.ptr = self.rng.randrange(len(self.classes))

    def next(self, b):
        out = []
        for _ in range(b):
            cls = self.classes[self.ptr % len(self.classes)]
            self.ptr += 1
            pool = [s for s in self.by_class[cls] if s not in out]
            if not pool:
                continue
            key = lambda s: (self.use[s.idx], self.correct.get(s.idx, False), self.rng.random())
            s = min(pool, key=key)
            self.use[s.idx] += 1
            out.append(s)
        return out

    def observe(self, recs):
        for r in recs:
            self.correct[r["idx"]] = r["pred"] == r["gold"]


class EGOPrompt:
    def __init__(self, task, prog, bwd, train, val, args, log):
        self.task, self.prog, self.bwd, self.args, self.log = task, prog, bwd, args, log
        self.train, self.val = train, val
        self.labels = labels(task)
        self.nodes = ", ".join(allowed_nodes(task))
        rng = random.Random(args.seed)
        # fixed stratified racing subset of validation
        self.race_ids, self.val_a, self.val_b = self._race_split(val, rng)
        self.best_lock = threading.Lock()
        self.best = None                # {"state", "val_f1", "member", "gen"}
        self.history = []
        # --final_topk: the K best distinct accepted states by val F1 (re-ranked on train+val in run.py)
        self.topk = []
        # --progress_on val_or_scg: best SCG acceptance score seen (train+val with P_s0), counts as progress
        self.best_scg = None
        # --k_accept_data / --k_accept_ps: SCG candidates are accepted on a separate score. `sep` is False
        # with the defaults, and then every code path below is the original one.
        self.sep = args.k_accept_data != "val" or args.k_accept_ps != "current"
        if self.sep:
            if args.k_accept_data == "trainval":
                # val samples get idx + 10**6 so train and val records stay distinguishable in racing
                self.acc_data = list(train) + [Sample(s.idx + 10 ** 6, s.x, s.y) for s in val]
                self.acc_split = self._race_split(self.acc_data, rng)   # drawn after the val split: val split unchanged
            else:
                self.acc_data, self.acc_split = val, (self.race_ids, self.val_a, self.val_b)
            self.ps0 = initial_state(task).system

    def _race_split(self, data, rng):
        by = {}
        for s in data:
            by.setdefault(s.y, []).append(s)
        sub = []
        for l, ss in by.items():
            ss = ss[:]
            rng.shuffle(ss)
            sub += ss[: max(1, round(len(ss) * self.args.race_frac))]
        ids = {s.idx for s in sub}
        return ids, [s for s in data if s.idx in ids], [s for s in data if s.idx not in ids]

    # ------------------------------------------------------------ evaluation helpers
    def eval_val(self, st):
        recs = self.prog.run(st, self.val)
        return self.prog.score(recs), recs

    def eval_acc(self, st):
        """SCG acceptance score (`sep` only): P_c of `st` on the acceptance data, with P_s0 if k_accept_ps=initial."""
        recs = self.prog.run(self._acc_state(st), self.acc_data)
        return self.prog.score(recs), recs

    def _acc_state(self, st):
        if self.args.k_accept_ps != "initial":
            return st
        new = st.copy()
        new.system = self.ps0
        return new

    def race(self, cands, inc_recs, split=None):
        """Racing: all candidates on the racing subset, the best one (if not clearly worse than the
        incumbent there) on the rest of the data (validation unless `split` = (ids, A, B) is given).
        Returns [(cand, full_f1 | None, recs | None)]."""
        race_ids, val_a, val_b = split or (self.race_ids, self.val_a, self.val_b)
        inc_a = self.prog.score([r for r in inc_recs if r["idx"] in race_ids])
        stage1 = []
        for c in cands:     # submit all candidates at once: the pool interleaves them
            stage1.append((c, [self.prog.pool.submit(self.prog.run_one, c, s) for s in val_a]))
        scored = [(c, [f.result() for f in fs]) for c, fs in stage1]
        scored = [(c, ra, self.prog.score(ra)) for c, ra in scored]
        scored.sort(key=lambda t: t[2], reverse=True)
        out = [(c, None, None) for c, _, _ in scored]
        n_full = 0
        for i, (c, ra, fa) in enumerate(scored):
            if n_full >= self.args.race_keep or fa < inc_a - self.args.race_margin:
                continue
            rb = self.prog.run(c, val_b)
            out[i] = (c, self.prog.score(ra + rb), ra + rb)
            n_full += 1
        return out, inc_a, [round(t[2], 4) for t in scored]

    # ------------------------------------------------------------ backward calls
    def diagnose(self, st, errors, anchors, val_cm):
        err_txt = "\n\n".join(
            f"### Case {r['idx']} (gold {r['gold']}, predicted {r['pred'] or 'unparsed'})\n"
            f"Input:\n{_clip(self._x(r['idx']), self.args.clip_input)}\n"
            f"Causal description:\n{_clip(r['desc'], self.args.clip_desc)}\n"
            f"Predictor response (end):\n{_clip(r['raw'][-self.args.clip_resp:], self.args.clip_resp)}"
            for r in errors)
        anc_txt = "\n".join(f"- Case {r['idx']}: gold {r['gold']} (correct)" for r in anchors) or "(none)"
        user = BP.GRAD_USER.format(task_brief=TASK_BRIEF[self.task], system=st.system,
                                   instruction=st.causal_instruction, relations=_numbered(st.relations),
                                   labels=", ".join(self.labels), cm=json.dumps(val_cm), errors=err_txt,
                                   anchors=anc_txt)
        out = extract_json(self.bwd.chat(BP.GRAD_SYSTEM, user, role="backward_diagnose", json_mode=True)) or {}
        return out

    def propose_system(self, st, grad, val_cm, style, hint):
        k = self.args.k_system
        user = BP.SYSTEM_UPDATE_USER.format(
            task_brief=TASK_BRIEF[self.task], system=st.system, gradient=grad or "(none)",
            labels=", ".join(self.labels), cm=json.dumps(val_cm), hint=hint, k=k,
            strategies="; ".join(f"({i + 1}) {s}" for i, s in enumerate(BP.SYSTEM_STRATEGIES[:k])),
            max_words=self.args.max_system_words, style=style)
        out = extract_json(self.bwd.chat(BP.SYSTEM_UPDATE_SYSTEM, user, role="backward_update_system",
                                         json_mode=True)) or {}
        cands = []
        for c in (out.get("candidates") or [])[:k]:
            txt = (c.get("system") or "").strip()
            if not txt or txt == st.system:
                continue
            txt = ensure_format(self.task, txt)
            new = st.copy()
            new.system = txt
            cands.append((new, c.get("strategy", ""), c.get("rationale", "")))
        return cands

    def propose_causal(self, st, grad, style, hint):
        k = self.args.k_causal
        user = BP.CAUSAL_UPDATE_USER.format(
            task_brief=TASK_BRIEF[self.task], instruction=st.causal_instruction,
            relations=_numbered(st.relations), nodes=self.nodes, gradient=grad or "(none)", hint=hint, k=k,
            strategies="; ".join(f"({i + 1}) {s}" for i, s in enumerate(BP.CAUSAL_STRATEGIES[:k])),
            max_ops=self.args.max_ops, max_rel_words=self.args.max_rel_words,
            max_relations=self.args.max_relations, style=style)
        out = extract_json(self.bwd.chat(BP.CAUSAL_UPDATE_SYSTEM, user, role="backward_update_causal",
                                          json_mode=True)) or {}
        return [(c.get("ops") or [], c.get("instruction"), c.get("strategy", ""), c.get("rationale", ""))
                for c in (out.get("candidates") or [])[:k]]

    def apply_ops(self, st, ops, instruction):
        """Apply add / modify / delete (indices refer to the original numbering) with size caps."""
        rel = list(st.relations)
        n = len(rel)
        dele, mod, add = set(), {}, []
        for o in ops[: self.args.max_ops]:
            op = str(o.get("op", "")).lower()
            r = (o.get("relation") or "").strip()
            if r and len(r.split()) > self.args.max_rel_words:
                r = _cap_words(r, self.args.max_rel_words)
            try:
                i = int(o.get("index", 0)) - 1
            except (TypeError, ValueError):
                i = -1
            if op == "delete" and 0 <= i < n:
                dele.add(i)
            elif op == "modify" and 0 <= i < n and r:
                mod[i] = r
            elif op == "add" and r:
                add.append(r)
        new_rel = [mod.get(i, x) for i, x in enumerate(rel) if i not in dele] + add
        new_rel = new_rel[: self.args.max_relations]
        new = st.copy()
        new.relations = new_rel
        if instruction and isinstance(instruction, str) and instruction.strip():
            new.causal_instruction = " ".join(instruction.split()[: self.args.max_instruction_words]) \
                if len(instruction.split()) > self.args.max_instruction_words else instruction.strip()
        if new.relations == st.relations and new.causal_instruction == st.causal_instruction:
            return None
        return new

    def refine(self, st):
        """Knowledge refinement by the forward engine: merge duplicate / overlapping relations, fix node
        names, make contradictions conditional. Returns (state, changes); falls back to `st` when the
        answer is unusable or drops relations that were not merged (it may not prune on its own)."""
        if not self.args.refine or not st.relations:
            return st, []
        user = BP.REFINE_USER.format(nodes=self.nodes, relations=_numbered(st.relations),
                                     max_rel_words=self.args.max_rel_words)
        try:
            out = extract_json(self.prog.fwd.chat(BP.REFINE_SYSTEM, user, role="forward_refine", json_mode=True))
        except Exception:  # noqa: BLE001 - refinement is optional
            return st, []
        rel = [str(r).strip() for r in (out or {}).get("relations") or [] if str(r).strip()]
        changes = [str(c) for c in (out or {}).get("changes") or []]
        if not rel or len(rel) < max(1, (len(st.relations) + 1) // 2):   # empty or drastic shrink -> keep original
            return st, []
        rel = [_cap_words(r, self.args.max_rel_words) for r in rel][: self.args.max_relations]
        if rel == st.relations:
            return st, []
        new = st.copy()
        new.relations = rel
        return new, changes

    def crossover(self, a, b):
        user = BP.CROSSOVER_USER.format(
            task_brief=TASK_BRIEF[self.task], fa=a["val_f1"], fb=b["val_f1"],
            sys_a=a["state"].system, ins_a=a["state"].causal_instruction, rel_a=_numbered(a["state"].relations),
            sys_b=b["state"].system, ins_b=b["state"].causal_instruction, rel_b=_numbered(b["state"].relations),
            max_relations=self.args.max_relations, max_rel_words=self.args.max_rel_words,
            max_words=self.args.max_system_words, labels=", ".join(self.labels), nodes=self.nodes)
        out = extract_json(self.bwd.chat(BP.CROSSOVER_SYSTEM, user, role="backward_crossover", json_mode=True))
        if not out or not out.get("system") or not isinstance(out.get("relations"), list):
            return None
        return PromptState(system=ensure_format(self.task, out["system"].strip()),
                           causal_instruction=(out.get("instruction") or a["state"].causal_instruction).strip(),
                           relations=[str(r).strip() for r in out["relations"] if str(r).strip()][: self.args.max_relations],
                           causal_output=a["state"].causal_output)

    # ------------------------------------------------------------ population
    def _x(self, idx):
        return self._train_by_idx[idx].x

    def _offer_topk(self, m, gen):
        """Keep the K best DISTINCT accepted states by full-val F1 (ties: the earlier one stays ahead).
        Called under best_lock from `_maybe_best`, i.e. for the initial state and every accepted
        member state / crossover child."""
        key = json.dumps(m["state"].to_dict(), sort_keys=True)
        if any(c["key"] == key for c in self.topk):
            return
        self.topk.append({"key": key, "state": m["state"].copy(), "val_f1": m["val_f1"], "member": m["id"],
                          "gen": gen, "val_metrics": self.prog.metrics(m["val_recs"])})
        self.topk.sort(key=lambda c: -c["val_f1"])       # stable sort
        del self.topk[self.args.final_topk:]

    def _maybe_best(self, m, gen):
        with self.best_lock:
            if self.args.final_topk:
                self._offer_topk(m, gen)
            if self.best is None or m["val_f1"] > self.best["val_f1"] + 1e-9:
                self.best = {"state": m["state"].copy(), "val_f1": m["val_f1"], "member": m["id"], "gen": gen,
                             "val_metrics": self.prog.metrics(m["val_recs"])}
                return True
        return False

    def member_step(self, m, gen, T):
        a = self.args
        ev = {"member": m["id"], "gen": gen, "T": round(T, 4), "val_before": round(m["val_f1"], 4)}
        batch = m["sampler"].next(a.batch_size)
        recs = self.prog.run(m["state"], batch)
        m["sampler"].observe(recs)
        errors = [r for r in recs if r["pred"] != r["gold"]]
        anchors = [r for r in recs if r["pred"] == r["gold"]][: a.n_anchors]
        ev["batch_acc"] = round(1 - len(errors) / max(1, len(recs)), 3)
        if not errors:
            ev["action"] = "skip_no_error"
            return ev
        val_cm = self.prog.metrics(m["val_recs"])["cm"]
        grad = self.diagnose(m["state"], errors[: a.max_errors], anchors, val_cm)
        pri = str(grad.get("priority", "both")).lower()
        pri = pri if pri in ("causal", "system", "both") else "both"
        ev.update(priority=pri, sources=[c.get("source") for c in grad.get("cases", [])],
                  system_gradient=grad.get("system_gradient", ""), causal_gradient=grad.get("causal_gradient", ""))
        style = BP.MEMBER_STYLES[m["id"] % len(BP.MEMBER_STYLES)]
        hint = m.pop("hint", "")
        # generate both proposal sets concurrently (latency), evaluate sequentially (P_c on the updated P_s)
        with ThreadPoolExecutor(2) as ex:
            fs = ex.submit(self.propose_system, m["state"], grad.get("system_gradient", ""), val_cm, style, hint) \
                if pri in ("system", "both") else None
            fc = ex.submit(self.propose_causal, m["state"], grad.get("causal_gradient", ""), style, hint) \
                if pri in ("causal", "both") else None
            sys_c = fs.result() if fs else []
            cau_c = fc.result() if fc else []
        for kind, cands in (("system", sys_c), ("causal", cau_c)):
            if kind == "causal":   # rebuild SCG candidates on top of the (possibly new) P_s
                built = []
                for ops, ins, strat, why in cands:
                    new = self.apply_ops(m["state"], ops, ins)
                    if new is not None:
                        built.append((new, strat, why))
                # knowledge refinement of every candidate (forward engine, concurrent) before it is scored
                refined = list(self.prog.pool.map(lambda c: self.refine(c[0]), built)) if built else []
                ev["refine_changes"] = [ch for _, ch in refined]
                cands = [(r[0], c[1], c[2]) for r, c in zip(refined, built)]
            if not cands:
                # the diagnosis routed this step to the other prompt only -> nothing was proposed here
                skipped = pri not in ("both", kind)
                ev[f"{kind}_result"] = f"skip (priority={pri})" if skipped else "no_candidate"
                continue
            # separate SCG acceptance (--k_accept_data / --k_accept_ps): race and SA on the acceptance score;
            # the incumbent is re-scored here (free from the cache unless its scored prompts changed)
            acc = self.sep and kind == "causal"
            if acc:
                old, inc_recs = self.eval_acc(m["state"])
                pairs = [(self._acc_state(c[0]), c[0]) for c in cands]
                back = {id(x): y for x, y in pairs}
                res, inc_a, sub_scores = self.race([x for x, _ in pairs], inc_recs, self.acc_split)
            else:
                res, inc_a, sub_scores = self.race([c[0] for c in cands], m["val_recs"])
            full = [(i, f, r) for i, (_, f, r) in enumerate(res) if f is not None]
            ev[f"{kind}_race"] = {"incumbent_subset": round(inc_a, 4), "subset": sub_scores,
                                  "strategies": [c[1] for c in cands]}
            if not full:
                ev[f"{kind}_result"] = "raced_out"
                continue
            i, f, r = max(full, key=lambda t: t[1])
            new = back[id(res[i][0])] if acc else res[i][0]
            if not acc:
                old = m["val_f1"]
            # tie -> prefer the shorter prompt
            if f == old and new.size_words() >= m["state"].size_words():
                ev[f"{kind}_result"] = f"tie_reject {old:.4f}"
                continue
            if sa_accept(old, f, T):
                if acc:     # the member keeps its real P_s: recompute its val F1 (P_s acceptance, best, top-K)
                    vf, vr = self.eval_val(new)
                    val_txt = f" [{a.k_accept_data}/{'P_s0' if a.k_accept_ps == 'initial' else 'P_s'}; " \
                              f"val {m['val_f1']:.4f}->{vf:.4f}]"
                    m["state"], m["val_f1"], m["val_recs"] = new, vf, vr
                    with self.best_lock:
                        if self.best_scg is None or f > self.best_scg + 1e-9:
                            self.best_scg = f
                else:
                    m["state"], m["val_f1"], m["val_recs"] = new, f, r
                ev[f"{kind}_result"] = f"{'accept' if f >= old else 'sa_accept'} {old:.4f}->{f:.4f}" \
                    + (val_txt if acc else "")
                if self._maybe_best(m, gen):
                    ev[f"{kind}_result"] += " NEW_BEST"
            else:
                ev[f"{kind}_result"] = f"reject {old:.4f}->{f:.4f}" + (f" [{a.k_accept_data}]" if acc else "")
        ev["val_after"] = round(m["val_f1"], 4)
        return ev

    def run(self, init_state, ckpt_path=None):
        a = self.args
        self._train_by_idx = {s.idx: s for s in self.train}
        start_gen = 0
        pop = None
        if ckpt_path and os.path.exists(ckpt_path):
            ck = json.load(open(ckpt_path, encoding="utf-8"))
            start_gen = ck["gen"] + 1
            self.history = ck.get("history", [])
            pop = []
            for p in ck["population"]:
                st = PromptState.from_dict(p["state"])
                f, recs = self.eval_val(st)          # cache makes this free
                s = Sampler(self.train, self.labels, a.seed * 100 + p["id"])
                s.use.update({int(k): v for k, v in p["use"].items()})
                pop.append({"id": p["id"], "state": st, "val_f1": f, "val_recs": recs, "sampler": s})
            b = ck["best"]
            self.best = {"state": PromptState.from_dict(b["state"]), "val_f1": b["val_f1"], "member": b["member"],
                         "gen": b["gen"], "val_metrics": b.get("val_metrics")}
            self.topk = [dict(c, state=PromptState.from_dict(c["state"])) for c in ck.get("topk", [])]
            self.log(f"resumed at generation {start_gen}, best val {self.best['val_f1']:.4f}")
        if pop is None:
            f0, recs0 = self.eval_val(init_state)
            self.init_val = self.prog.metrics(recs0)
            self.log(f"initial val F1 {f0:.4f}  cm {self.init_val['cm']}")
            pop = [{"id": i, "state": init_state.copy(), "val_f1": f0, "val_recs": recs0,
                    "sampler": Sampler(self.train, self.labels, a.seed * 100 + i)} for i in range(a.pop_size)]
            self._maybe_best(pop[0], -1)
        if self.sep:    # score the initial SCG once for acceptance (else 3 concurrent members pay for it)
            f_acc, _ = self.eval_acc(pop[0]["state"])
            if self.best_scg is None:
                self.best_scg = f_acc
            self.log(f"SCG acceptance on {a.k_accept_data} with {'P_s0' if a.k_accept_ps == 'initial' else 'current P_s'}"
                     f" ({len(self.acc_data)} samples, racing subset {len(self.acc_split[1])}): initial F1 {f_acc:.4f}")
        stale = 0
        members_pool = ThreadPoolExecutor(max_workers=a.pop_size)
        for gen in range(start_gen, a.generations):
            T = a.t0 * max(0.0, 1 - gen / a.generations)
            t0 = time.time()
            best_before = self.best["val_f1"]
            scg_before = self.best_scg
            evs = list(members_pool.map(lambda m: self._safe_step(m, gen, T), pop))
            # crossover: worst member gets a child of (worst, best) every `crossover_every` generations
            if a.crossover_every and (gen + 1) % a.crossover_every == 0 and a.pop_size > 1:
                evs.append(self._crossover_step(pop, gen, T))
            improved = self.best["val_f1"] > best_before + 1e-9
            if a.progress_on == "val_or_scg" and self.best_scg is not None and scg_before is not None:
                improved = improved or self.best_scg > scg_before + 1e-9     # SCG got better on its own score
            stale = 0 if improved else stale + 1
            rec = {"gen": gen, "T": round(T, 4), "seconds": round(time.time() - t0, 1),
                   "member_val": [round(m["val_f1"], 4) for m in pop], "best_val": round(self.best["val_f1"], 4),
                   "best_scg": None if self.best_scg is None else round(self.best_scg, 4), "stale": stale,
                   "events": evs}
            self.history.append(rec)
            self.log(f"gen {gen}: members {rec['member_val']} best {rec['best_val']} "
                     f"({rec['seconds']}s) " + " | ".join(
                         f"m{e.get('member')}: {e.get('system_result', '-')} / {e.get('causal_result', '-')}"
                         for e in evs if 'member' in e))
            if ckpt_path:
                self._save_ckpt(ckpt_path, gen, pop)
            if a.patience and stale >= a.patience and gen + 1 >= a.min_generations:
                self.log(f"early stop: no progress for {stale} generations ({gen + 1} run, min {a.min_generations})")
                break
        members_pool.shutdown()
        return self.best, pop

    def _safe_step(self, m, gen, T):
        try:
            return self.member_step(m, gen, T)
        except Exception as e:  # noqa: BLE001 - one failed step must not kill the run
            return {"member": m["id"], "gen": gen, "error": repr(e)[:300]}

    def _crossover_step(self, pop, gen, T):
        order = sorted(pop, key=lambda m: m["val_f1"])
        worst, best = order[0], order[-1]
        ev = {"crossover": True, "gen": gen, "worst": worst["id"], "best": best["id"]}
        if worst is best or best["val_f1"] - worst["val_f1"] < 1e-9:
            ev["result"] = "skip_equal"
            return ev
        child = self.crossover(worst, best)
        if child is None:
            ev["result"] = "no_child"
            return ev
        child, ev["refine_changes"] = self.refine(child)
        f, recs = self.eval_val(child)
        if sa_accept(worst["val_f1"], f, T):
            ev["result"] = f"accept {worst['val_f1']:.4f}->{f:.4f}"
            worst.update(state=child, val_f1=f, val_recs=recs)
            if self._maybe_best(worst, gen):
                ev["result"] += " NEW_BEST"
        else:
            ev["result"] = f"reject {worst['val_f1']:.4f}->{f:.4f}"
            worst["hint"] = (f"\n## Hint from a stronger lineage (val F1 {best['val_f1']:.3f})\n"
                             f"Its relations:\n{_numbered(best['state'].relations)}\n")
        return ev

    def _save_ckpt(self, path, gen, pop):
        ck = {"gen": gen, "history": self.history,
              "best": {"state": self.best["state"].to_dict(), "val_f1": self.best["val_f1"],
                       "member": self.best["member"], "gen": self.best["gen"],
                       "val_metrics": self.best.get("val_metrics")},
              "population": [{"id": m["id"], "state": m["state"].to_dict(), "val_f1": m["val_f1"],
                              "use": m["sampler"].use} for m in pop]}
        if self.args.final_topk:
            ck["topk"] = [dict(c, state=c["state"].to_dict()) for c in self.topk]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ck, f, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
        os.replace(tmp, path)
