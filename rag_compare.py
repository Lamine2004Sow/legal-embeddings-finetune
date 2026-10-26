"""Effet du modèle fine-tuné sur les réponses du RAG du projet 1.

Usage :
    make rag-eval          # lance les runs dans legal-rag-agent (labels p2-<jeu>-<modèle>-ref<0|1>)
    python rag_compare.py  # lit le plus récent run de chaque label -> results/rag.json + tableau

Matrice {e5-base, e5-base fine-tuné} × {avec, sans reformulation} sur deux jeux :
    dev   les 60 questions du projet 1 (40 answerable, 20 à refuser) : mesure aussi les refus
    test  les 125 questions du jeu de test de ce projet (toutes answerable)

Une question answerable est réussie si l'agent répond ET cite un article attendu.
Comparaisons appariées par McNemar exact sur ce succès, et sur le hit de recherche.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

RAG_DIR = Path(os.environ.get("RAG_DIR", "../legal-rag-agent"))
RUNS_DIR = RAG_DIR / "eval" / "runs"
OUT = Path("results") / "rag.json"

CONFIGS = [
    ("base-ref1", "e5-base + reformulation (projet 1)"),
    ("base-ref0", "e5-base sans reformulation"),
    ("ft-ref0", "fine-tuné sans reformulation"),
    ("ft-ref1", "fine-tuné + reformulation"),
]
# (A, B) : B comparé à A
COMPARISONS = [("base-ref0", "ft-ref0"), ("base-ref1", "ft-ref0"), ("base-ref1", "ft-ref1")]


def latest_run(label: str) -> Path | None:
    runs = sorted(d for d in RUNS_DIR.glob(f"*_{label}") if (d / "summary.json").exists())
    return runs[-1] if runs else None


def _key(item: dict) -> tuple[str, str]:
    return (item.get("code", ""), item.get("num", ""))


def per_question(results: list[dict]) -> dict[str, dict[str, int]]:
    out = {}
    for r in results:
        if r.get("category") != "answerable":
            continue
        expected = {_key(e) for e in r.get("expected") or []}
        hit = any(_key(d) in expected for d in r.get("retrieved") or [])
        cited = any(_key(c) in expected for c in r.get("citations") or [])
        out[r["id"]] = {"retrieval_hit": int(hit), "success": int(r.get("status") == "answered" and cited)}
    return out


def mcnemar(a: dict, b: dict, metric: str) -> dict:
    ids = sorted(set(a) & set(b))
    gained = sum(b[i][metric] and not a[i][metric] for i in ids)
    lost = sum(a[i][metric] and not b[i][metric] for i in ids)
    p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
    return {"gained": gained, "lost": lost, "p": float(p)}


def main() -> None:
    report: dict = {}
    for set_name in ("dev", "test"):
        rows, per_q = {}, {}
        for cfg, _ in CONFIGS:
            run = latest_run(f"p2-{set_name}-{cfg}")
            if run is None:
                continue
            summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            results = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            per_q[cfg] = per_question(results)
            success = np.array([v["success"] for v in per_q[cfg].values()])
            rows[cfg] = {
                "run": run.name,
                "n_answerable": len(per_q[cfg]),
                "retrieval_hit_rate": summary["retrieval_hit_rate"],
                "answer_rate": summary["answer_rate"],
                "citation_accuracy": summary["citation_accuracy"],
                "success_rate": float(success.mean()) if len(success) else None,
                "correct_refusal_rate": summary["correct_refusal_rate"],
                "error_count": summary["error_count"],
                "llm_calls_per_question": 3 if cfg.endswith("ref1") else 2,
            }
        comps = {}
        for a, b in COMPARISONS:
            if a in per_q and b in per_q:
                comps[f"{b} vs {a}"] = {m: mcnemar(per_q[a], per_q[b], m) for m in ("retrieval_hit", "success")}
        report[set_name] = {"configs": rows, "comparisons": comps}

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print_tables(report)


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{100 * v:.1f} %"


def print_tables(report: dict) -> None:
    labels = dict(CONFIGS)
    for set_name, data in report.items():
        print(f"\n## {set_name}\n")
        print("| configuration | recherche | réponse | citation exacte | succès | refus corrects | appels LLM |")
        print("|---|---|---|---|---|---|---|")
        for cfg, row in data["configs"].items():
            print(
                f"| {labels[cfg]} | {_pct(row['retrieval_hit_rate'])} | {_pct(row['answer_rate'])} | "
                f"{_pct(row['citation_accuracy'])} | {_pct(row['success_rate'])} | "
                f"{_pct(row['correct_refusal_rate'])} | {row['llm_calls_per_question']} |"
            )
        for name, comp in data["comparisons"].items():
            s, h = comp["success"], comp["retrieval_hit"]
            print(f"- {name} : succès +{s['gained']}/-{s['lost']} (p={s['p']:.3f}), recherche +{h['gained']}/-{h['lost']} (p={h['p']:.3f})")


if __name__ == "__main__":
    main()
