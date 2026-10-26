"""Évaluation de la recherche : Recall@k et MRR, modèle de base contre modèle fine-tuné.

Usage :
    python evaluate.py                                   # modèles par défaut, tous les jeux
    python evaluate.py --models e5-base e5-base-ft --sets test
    python evaluate.py --models bge-m3 --sets test dev

Jeux d'évaluation (questions -> articles attendus) :
    test       eval/test_questions.jsonl : rédigé par Claude Haiku, figé avant l'entraînement ;
               moitié chapitres held-out, moitié chapitres d'entraînement
    dev        les 40 questions `answerable` du projet 1 : ont servi à mettre au point le RAG
               et à choisir le checkpoint, donc pas un test
    synthetic  questions DeepSeek sur les chapitres held-out (même générateur que l'entraînement :
               optimiste, sert de contrôle)

La recherche se fait sur les 5 348 chunks de l'index du RAG ; un article est trouvé si l'un
de ses chunks l'est. Deux modes : dense seul, et hybride (dense + BM25, fusion RRF, comme
le retriever du projet 1, sans l'accès direct par numéro d'article).

Les intervalles de confiance sont des bootstraps à 95 % ; la comparaison au modèle de
référence (e5-base) est appariée : bootstrap de la différence et test de McNemar exact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from scipy.stats import binomtest

import corpus

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

RAG_DIR = Path(os.environ.get("RAG_DIR", "../legal-rag-agent"))
EVAL_DIR = Path("eval")
RESULTS_DIR = Path("results")
EMB_CACHE = corpus.DATA_DIR / "emb"

KS = (1, 5, 10, 20)
N_CANDIDATES = 30  # comme HybridRetriever.search
RRF_K = 60
N_BOOT = 2000
REFERENCE = "e5-base"

# nom -> (identifiant ou chemin, préfixe requête, préfixe passage)
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5-base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "bge-m3": ("BAAI/bge-m3", "", ""),
    "e5-base-ft": ("models/e5-base-legal", "query: ", "passage: "),
    "e5-small-ft": ("models/e5-small-legal", "query: ", "passage: "),
}

# Copie de legal-rag-agent/retriever.py, pour que BM25 se comporte comme dans le RAG.
STOPWORDS = frozenset(
    """le la les de des du un une et ou a au aux en que qui est pour par dans sur se ce il elle je on si
    d l s n qu ne pas son sa ses leur leurs plus peut etre sont avec cette tout toute dont ainsi""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return [t for t in _TOKEN_RE.findall(text) if t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Jeux d'évaluation
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_eval_set(name: str) -> list[dict]:
    """[{id, question, expected: set[article_key], split}]"""
    if name == "test":
        rows = _read_jsonl(EVAL_DIR / "test_questions.jsonl")
    elif name == "dev":
        rows = [r for r in _read_jsonl(RAG_DIR / "eval" / "questions.jsonl") if r["category"] == "answerable"]
    elif name == "synthetic":
        rows = [
            {"id": f"syn-{i}", "question": r["question"], "expected": [_split_key(r["article_key"])], "split": "heldout"}
            for i, r in enumerate(_read_jsonl(corpus.DATA_DIR / "pairs_heldout.jsonl"))
            if r["kept"]
        ]
    else:
        raise ValueError(name)
    return [
        {
            "id": r["id"],
            "question": r["question"],
            "expected": {f"{e['code']}|{e['num']}" for e in r["expected"]},
            "split": r.get("split"),
        }
        for r in rows
    ]


def _split_key(key: str) -> dict:
    code, num = key.split("|", 1)
    return {"code": code, "num": num}


# --------------------------------------------------------------------------- #
# Recherche
# --------------------------------------------------------------------------- #


class Searcher:
    def __init__(self, chunks: pd.DataFrame) -> None:
        self.chunk_article = chunks["article_key"].tolist()
        self.texts = chunks["text"].tolist()
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])

    def encode_corpus(self, name: str) -> np.ndarray:
        """Embeddings des chunks, mis en cache sur disque (l'encodage CPU prend plusieurs minutes)."""
        path = EMB_CACHE / f"{name}.npy"
        model_path = MODELS[name][0]
        stamp = _model_stamp(model_path)
        meta = path.with_suffix(".json")
        if path.exists() and meta.exists() and json.loads(meta.read_text())["stamp"] == stamp:
            return np.load(path)
        _, _, p_prefix = MODELS[name]
        emb = load_model(name).encode(
            [p_prefix + t for t in self.texts], batch_size=16, normalize_embeddings=True, show_progress_bar=True
        )
        EMB_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, emb)
        meta.write_text(json.dumps({"stamp": stamp}))
        return emb

    def rank(self, scores: np.ndarray, top: int) -> list[int]:
        idx = np.argpartition(-scores, min(top, len(scores) - 1))[:top]
        return [int(i) for i in idx[np.argsort(-scores[idx])]]

    def articles(self, chunk_ids: list[int]) -> list[str]:
        return list(dict.fromkeys(self.chunk_article[i] for i in chunk_ids))

    def dense(self, q_emb: np.ndarray, c_emb: np.ndarray, depth: int = 60) -> list[list[str]]:
        sims = q_emb @ c_emb.T
        return [self.articles(self.rank(row, depth)) for row in sims]

    def bm25_only(self, questions: list[str], depth: int = 60) -> list[list[str]]:
        return [self.articles(self.rank(self.bm25.get_scores(tokenize(q)), depth)) for q in questions]

    def hybrid(self, questions: list[str], q_emb: np.ndarray, c_emb: np.ndarray) -> list[list[str]]:
        """RRF dense + BM25 sur N_CANDIDATES chunks chacun, comme HybridRetriever.search."""
        out = []
        sims = q_emb @ c_emb.T
        for q, row in zip(questions, sims):
            dense_ids = self.rank(row, N_CANDIDATES)
            bm = self.bm25.get_scores(tokenize(q))
            bm_ids = [i for i in self.rank(bm, N_CANDIDATES) if bm[i] > 0]
            rrf: dict[int, float] = defaultdict(float)
            for ranking in (dense_ids, bm_ids):
                for r, i in enumerate(ranking, 1):
                    rrf[i] += 1 / (RRF_K + r)
            out.append(self.articles(sorted(rrf, key=rrf.get, reverse=True)))
        return out


def _model_stamp(model_path: str) -> str:
    """Change quand un modèle local est ré-entraîné (sinon le cache d'embeddings serait périmé)."""
    p = Path(model_path)
    if p.exists():
        weights = sorted(p.glob("*.safetensors"))
        return f"{model_path}:{max(w.stat().st_mtime for w in weights) if weights else 0}"
    return model_path


_loaded: dict[str, object] = {}


def load_model(name: str):
    from sentence_transformers import SentenceTransformer

    if name not in _loaded:
        _loaded[name] = SentenceTransformer(MODELS[name][0], device="cpu")
    return _loaded[name]


def encode_queries(name: str, questions: list[str]) -> np.ndarray:
    q_prefix = MODELS[name][1]
    return load_model(name).encode([q_prefix + q for q in questions], batch_size=32, normalize_embeddings=True)


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #


def per_question(rankings: list[list[str]], items: list[dict]) -> dict[str, np.ndarray]:
    """Pour chaque question : hit@k (0/1) et rang réciproque (MRR@10)."""
    first = []
    for ranked, item in zip(rankings, items):
        pos = next((r for r, a in enumerate(ranked, 1) if a in item["expected"]), None)
        first.append(pos)
    out = {f"recall@{k}": np.array([p is not None and p <= k for p in first], dtype=float) for k in KS}
    out["mrr@10"] = np.array([1 / p if p is not None and p <= 10 else 0.0 for p in first])
    return out


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_comparison(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """b - a : différence moyenne, IC bootstrap apparié, McNemar exact (métriques 0/1)."""
    diff = b - a
    lo, hi = bootstrap_ci(diff, rng)
    result = {"delta": float(diff.mean()), "ci95": [lo, hi]}
    if set(np.unique(np.concatenate([a, b]))) <= {0.0, 1.0}:
        gained, lost = int(((b == 1) & (a == 0)).sum()), int(((b == 0) & (a == 1)).sum())
        p = binomtest(gained, gained + lost, 0.5).pvalue if gained + lost else 1.0
        result.update({"gained": gained, "lost": lost, "mcnemar_p": float(p)})
    return result


def summarize(scores: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    return {m: {"mean": float(v.mean()), "ci95": list(bootstrap_ci(v, rng))} for m, v in scores.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def evaluate(model_names: list[str], set_names: list[str], log_mlflow: bool = True) -> dict:
    articles = corpus.load_articles()
    searcher = Searcher(corpus.build_chunks(articles))
    rng = np.random.default_rng(0)
    report: dict = {}

    for set_name in set_names:
        items = load_eval_set(set_name)
        questions = [it["question"] for it in items]
        per_system: dict[str, dict[str, np.ndarray]] = {}
        rankings_out: dict[str, list[list[str]]] = {}

        per_system["bm25"] = per_question(searcher.bm25_only(questions), items)
        for name in model_names:
            c_emb = searcher.encode_corpus(name)
            q_emb = encode_queries(name, questions)
            dense = searcher.dense(q_emb, c_emb)
            hybrid = searcher.hybrid(questions, q_emb, c_emb)
            per_system[f"{name}/dense"] = per_question(dense, items)
            per_system[f"{name}/hybrid"] = per_question(hybrid, items)
            rankings_out[f"{name}/dense"] = dense
            rankings_out[f"{name}/hybrid"] = hybrid

        set_report = {"n": len(items), "systems": {}}
        splits = sorted({it["split"] for it in items if it["split"]})
        for system, scores in per_system.items():
            entry = {"all": summarize(scores, rng)}
            for split in splits:
                mask = np.array([it["split"] == split for it in items])
                entry[split] = {m: float(v[mask].mean()) for m, v in scores.items()}
                entry[f"n_{split}"] = int(mask.sum())
            mode = system.split("/")[-1]
            ref = per_system.get(f"{REFERENCE}/{mode}")
            if ref is not None and not system.startswith(REFERENCE + "/"):
                entry["vs_" + REFERENCE] = {m: paired_comparison(ref[m], scores[m], rng) for m in scores}
            set_report["systems"][system] = entry
        report[set_name] = set_report
        _write_details(set_name, items, rankings_out)
        print(format_table(set_name, set_report))

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "retrieval.json"
    previous = json.loads(out.read_text()) if out.exists() else {}
    for set_name, set_report in report.items():
        previous.setdefault(set_name, {"n": set_report["n"], "systems": {}})
        previous[set_name]["n"] = set_report["n"]
        previous[set_name]["systems"].update(set_report["systems"])
    out.write_text(json.dumps(previous, ensure_ascii=False, indent=2))
    if log_mlflow:
        _log_mlflow(report, model_names)
    return report


def _write_details(set_name: str, items: list[dict], rankings: dict[str, list[list[str]]]) -> None:
    """Top 10 de chaque système par question : sert à lire les échecs."""
    path = RESULTS_DIR / f"rankings_{set_name}.jsonl"
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = {}
    if path.exists():
        existing = {r["id"]: r for r in _read_jsonl(path)}
    with path.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            row = existing.get(it["id"], {"id": it["id"], "question": it["question"], "expected": sorted(it["expected"])})
            row.setdefault("top10", {})
            for system, ranks in rankings.items():
                row["top10"][system] = ranks[i][:10]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log_mlflow(report: dict, model_names: list[str]) -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("retrieval-eval")
    for set_name, set_report in report.items():
        for system, entry in set_report["systems"].items():
            with mlflow.start_run(run_name=f"{set_name}:{system}"):
                mlflow.log_params({"eval_set": set_name, "system": system, "n": set_report["n"]})
                mlflow.log_metrics({m.replace("@", "_at_"): v["mean"] for m, v in entry["all"].items()})


def format_table(set_name: str, set_report: dict) -> str:
    lines = [f"\n## {set_name} (n={set_report['n']})", "", "| système | R@1 | R@5 | R@10 | R@20 | MRR@10 |", "|---|---|---|---|---|---|"]
    for system, entry in set_report["systems"].items():
        cells = []
        for m in ("recall@1", "recall@5", "recall@10", "recall@20", "mrr@10"):
            v = entry["all"][m]
            cells.append(f"{v['mean']:.3f} [{v['ci95'][0]:.2f}–{v['ci95'][1]:.2f}]")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["e5-base", "e5-base-ft"], choices=list(MODELS))
    parser.add_argument("--sets", nargs="+", default=["test", "dev", "synthetic"], choices=["test", "dev", "synthetic"])
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    evaluate(args.models, args.sets, log_mlflow=not args.no_mlflow)


if __name__ == "__main__":
    main()
