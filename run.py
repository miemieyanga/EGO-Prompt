"""EGO-Prompt entry point. Defaults = the configuration reported in the README.

    python run.py --task swiss --seed 42 --baselines --fwd mantle:openai.gpt-5.6-luna --bwd claude:anthropic.claude-sonnet-5

Credentials (set in the shell, never in files of this repository): a Bedrock API key in
AWS_BEARER_TOKEN_BEDROCK, or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (a short-term key is minted);
OPENAI_API_KEY for `openai:` models. Outputs go to res/<run>/: log.txt, checkpoint.json (resume),
result.json, cache.sqlite.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm import LLM, Cache, Usage                     # noqa: E402
from state import load_split, initial_state, desc_rule_text, ROOT     # noqa: E402
from optimizer import Program, EGOPrompt                 # noqa: E402

# USD per 1M tokens (input, output), used for the cost report only. GPT-6: OpenAI list price; GPT-5.6: Bedrock
# us-east-1; Claude: Anthropic list price. Override or add models with --price model=in,out.
PRICES = {
    "openai.gpt-6-sol": (2.00, 10.00), "openai.gpt-6-luna": (0.10, 0.50),
    "openai.gpt-5.6-terra": (2.20, 13.20), "openai.gpt-5.6-luna": (0.22, 1.32),
    "anthropic.claude-sonnet-5": (2.00, 10.00), "anthropic.claude-sonnet-4-6": (3.00, 15.00),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["swiss", "trafficsafe", "pandemic"])
    p.add_argument("--fwd", default="mantle:openai.gpt-5.6-luna")
    p.add_argument("--bwd", default="claude:anthropic.claude-sonnet-5")
    p.add_argument("--fwd_effort", default="none", help="reasoning_effort of the forward engine; none = thinking off ('' = model default)")
    p.add_argument("--bwd_effort", default="none", help="reasoning_effort of the backward engine; none = thinking off")
    p.add_argument("--fwd_max_tokens", type=int, default=2500)
    p.add_argument("--bwd_max_tokens", type=int, default=12000)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--bwd_attempts", type=int, default=14, help="transient-error attempts per backward call")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run", default=None, help="run name (default <task>_s<seed>)")
    p.add_argument("--out", default=os.path.join(ROOT, "res"))
    # search
    p.add_argument("--pop_size", type=int, default=3)
    p.add_argument("--generations", type=int, default=6, help="each generation = one step of every member")
    p.add_argument("--t0", type=float, default=0.03, help="initial SA temperature (F1 units); linear decay to 0")
    p.add_argument("--crossover_every", type=int, default=2)
    p.add_argument("--patience", type=int, default=3, help="stop after this many generations without a new best (0=off)")
    p.add_argument("--min_generations", type=int, default=5, help="never stop early before this many generations")
    p.add_argument("--progress_on", default="val", choices=["val", "val_or_scg"],
                   help="val_or_scg: a new best SCG acceptance score (k_accept_data/k_accept_ps) also resets patience")
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--max_errors", type=int, default=5)
    p.add_argument("--n_anchors", type=int, default=2)
    p.add_argument("--k_system", type=int, default=3)
    p.add_argument("--k_causal", type=int, default=3)
    p.add_argument("--race_frac", type=float, default=0.4)
    p.add_argument("--race_keep", type=int, default=1, help="candidates promoted to full validation")
    p.add_argument("--race_margin", type=float, default=0.02)
    # experimental (off by default)
    p.add_argument("--k_accept_data", default="val", choices=["val", "trainval"],
                   help="data on which SCG (P_c) candidates are raced and SA-accepted; P_s stays on val")
    p.add_argument("--k_accept_ps", default="current", choices=["current", "initial"],
                   help="P_s used to score SCG candidates and the incumbent SCG for acceptance (initial = fixed P_s0)")
    p.add_argument("--desc_rule", type=int, default=1,
                   help="1 = append a fixed relevance rule to the causal prompt's output block (not optimisable)")
    p.add_argument("--desc_max_items", type=int, default=10, help="desc_rule: max numbered items in the description")
    p.add_argument("--desc_item_words", type=int, default=50, help="desc_rule: max words per item")
    p.add_argument("--refine", type=int, default=1,
                   help="1 = forward engine checks every SCG candidate (merge duplicates, node names, contradictions)")
    # prompt size caps
    p.add_argument("--max_ops", type=int, default=3)
    p.add_argument("--max_relations", type=int, default=12)
    p.add_argument("--max_rel_words", type=int, default=60)
    p.add_argument("--max_instruction_words", type=int, default=180)
    p.add_argument("--max_system_words", type=int, default=260)
    p.add_argument("--clip_input", type=int, default=2500)
    p.add_argument("--clip_desc", type=int, default=1500)
    p.add_argument("--clip_resp", type=int, default=800)
    # efficiency
    p.add_argument("--workers", type=int, default=32, help="concurrent forward calls per run")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--fused", action="store_true", help="one forward call per sample (causal description + answer)")
    p.add_argument("--train_frac", type=float, default=1.0, help="subsample train (quick tests)")
    p.add_argument("--val_n", type=int, default=0, help="limit val size (quick tests, 0 = all)")
    p.add_argument("--test_n", type=int, default=0)
    # evaluation
    p.add_argument("--test_repeats", type=int, default=1, help="independent test draws of the final prompts")
    p.add_argument("--baselines", action="store_true", help="also test the organized prompt without SCG")
    p.add_argument("--final_topk", type=int, default=0,
                   help="K>0: re-rank the K best distinct states by val F1 on train+val and test the winner (0 = off)")
    p.add_argument("--final_repeats", type=int, default=2, help="independent train+val draws per top-K candidate")
    p.add_argument("--price", action="append", default=[], help="model=in,out  USD per 1M tokens")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    run = a.run or f"{a.task}_s{a.seed}"
    out_dir = os.path.join(a.out, run)
    os.makedirs(out_dir, exist_ok=True)
    logf = open(os.path.join(out_dir, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    prices = dict(PRICES)
    for s in a.price:
        m, v = s.split("=")
        prices[m] = tuple(float(x) for x in v.split(","))

    usage = Usage()
    cache = None if a.no_cache else Cache(os.path.join(out_dir, "cache.sqlite"))
    fwd = LLM(a.fwd, effort=a.fwd_effort or None, max_tokens=a.fwd_max_tokens, region=a.region, usage=usage, cache=cache)
    # the backward engine makes few calls and a lost call drops a whole member step: retry for up to ~10 min
    # (Bedrock can return 529 Overloaded for several minutes)
    bwd = LLM(a.bwd, effort=a.bwd_effort or None, max_tokens=a.bwd_max_tokens, region=a.region, usage=usage, cache=cache,
              max_attempts=a.bwd_attempts)

    train, val, test = (load_split(a.task, s) for s in ("train", "val", "test"))
    if a.train_frac < 1:
        import random
        rng = random.Random(a.seed)
        train = rng.sample(train, max(4, int(len(train) * a.train_frac)))
    if a.val_n:
        val = val[: a.val_n]
    if a.test_n:
        test = test[: a.test_n]
    log(f"run {run}: task={a.task} fwd={a.fwd}({a.fwd_effort}) bwd={a.bwd}({a.bwd_effort}) "
        f"train/val/test={len(train)}/{len(val)}/{len(test)} args={vars(a)}")

    rule = desc_rule_text(a.task, a.desc_max_items, a.desc_item_words) if a.desc_rule else ""
    prog = Program(a.task, fwd, workers=a.workers, fused=a.fused, desc_rule=rule)
    opt = EGOPrompt(a.task, prog, bwd, train, val, a, log)
    init = initial_state(a.task)
    t0 = time.time()
    best, pop = opt.run(init, ckpt_path=os.path.join(out_dir, "checkpoint.json"))
    train_sec = time.time() - t0
    cost_train, _ = usage.cost(prices)
    log(f"search done in {train_sec / 60:.1f} min, best val {best['val_f1']:.4f} (member {best['member']}, "
        f"gen {best['gen']}), train cost ${cost_train:.2f}")

    # final test: best snapshot and the initial SCG prompts. Draw r uses salt r; within a draw, identical
    # (prompt, input) pairs share one cached output, so best vs initial is a paired comparison when P_c is unchanged
    def test_eval(st, fused=None):
        rows = []
        for r in range(a.test_repeats):
            rows.append(prog.metrics(prog.run(st, test, salt=r)))
        return rows

    res = {"run": run, "args": vars(a), "train_minutes": round(train_sec / 60, 2),
           "best_val_f1": best["val_f1"], "best_member": best["member"], "best_gen": best["gen"],
           "best_state": best["state"].to_dict(), "best_val_metrics": best.get("val_metrics"),
           "init_val_metrics": getattr(opt, "init_val", None)}
    res["test_best"] = test_eval(best["state"])
    if a.final_topk:
        # robust final selection: the top-K states by val F1 are re-scored on train+val with R fresh draws
        # (salts 1..R, i.e. not the cached search outputs); the best mean wins (ties: higher val rank)
        tv = train + val
        cands = opt.topk or [dict(best)]      # empty only when resumed from a checkpoint written without top-K
        futs = [[[prog.pool.submit(prog.run_one, c["state"], s, r) for s in tv] for r in range(1, a.final_repeats + 1)]
                for c in cands]
        rows = []
        for rank, (c, fr) in enumerate(zip(cands, futs)):
            f1s = [prog.score([f.result() for f in fs]) for fs in fr]
            rows.append({"val_rank": rank, "val_f1": c["val_f1"], "member": c["member"], "gen": c["gen"],
                         "trainval_f1": f1s, "trainval_mean": sum(f1s) / len(f1s), "state": c["state"].to_dict()})
            log(f"final selection: cand {rank} (member {c['member']}, gen {c['gen']}, val {c['val_f1']:.4f}) "
                f"train+val F1 {sum(f1s) / len(f1s):.4f} {[round(x, 4) for x in f1s]}")
        ch = max(range(len(rows)), key=lambda i: (rows[i]["trainval_mean"], -i))
        final = cands[ch]["state"]
        res["final_selection"] = {"k": a.final_topk, "repeats": a.final_repeats, "n_trainval": len(tv),
                                  "chosen": ch, "chosen_is_val_best": final.to_dict() == best["state"].to_dict(),
                                  "candidates": rows}
        res["final_state"] = final.to_dict()
        res["test_best_val"] = res["test_best"]
        res["test_final_selected"] = test_eval(final)
    res["test_init_scg"] = test_eval(init)
    if a.baselines:
        # organized prompt without SCG (the notebooks' baseline): P_s on the raw input, one call per sample
        res["test_organized_prompt"] = [prog.metrics(prog.run_plain(init.system, test, salt=r))
                                        for r in range(a.test_repeats)]
    total, by_role = usage.cost(prices)
    res["usage"] = usage.snapshot()
    res["cost_usd"] = {"total": round(total, 4), "search": round(cost_train, 4),
                       "by_role": {k: (None if v is None else round(v, 4)) for k, v in by_role.items()}}
    res["population_final"] = [{"id": m["id"], "val_f1": m["val_f1"], "state": m["state"].to_dict()} for m in pop]
    res["history"] = opt.history
    res["failed_samples"] = prog.n_errors
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=str)
    tb = [r["f1"] for r in res["test_best"]]
    ti = [r["f1"] for r in res["test_init_scg"]]
    fs_txt = ""
    if a.final_topk:
        tf = [r["f1"] for r in res["test_final_selected"]]
        fs_txt = f"final-selected {sum(tf) / len(tf):.4f} {tf} (cand {res['final_selection']['chosen']}) | val-"
    log(f"TEST weighted F1: {fs_txt}best {sum(tb) / len(tb):.4f} {tb} | initial SCG {sum(ti) / len(ti):.4f} {ti}"
        + (f" | organized {[round(r['f1'], 4) for r in res['test_organized_prompt']]}" if a.baselines else ""))
    log(f"failed samples after retries (scored as wrong): {prog.n_errors}")
    log(f"cost ${total:.2f} total (search ${cost_train:.2f}); usage {json.dumps(res['usage'])}")
    prog.pool.shutdown()


if __name__ == "__main__":
    main()
