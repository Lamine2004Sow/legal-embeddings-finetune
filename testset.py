"""Jeu de test : tirage des articles, puis assemblage des questions validées.

Usage :
    python testset.py sample      # data/test_candidates.jsonl : articles tirés au hasard
    python testset.py build       # eval/test_questions.jsonl, à partir des questions rédigées

Le jeu de test est figé AVANT la génération des paires d'entraînement, et rédigé par un
autre LLM (Claude Haiku) que celui qui génère l'entraînement (DeepSeek), avec d'autres
consignes : on ne veut pas mesurer l'apprentissage du style du générateur.

Moitié des articles dans les chapitres held-out (jamais vus à l'entraînement), moitié dans
les chapitres d'entraînement (le cas réel : le corpus est fermé, l'index de production
contient les mêmes articles).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import corpus

CANDIDATES_PATH = corpus.DATA_DIR / "test_candidates.jsonl"
DRAFTS_DIR = corpus.DATA_DIR / "test_drafts"
VERIFY_DIR = corpus.DATA_DIR / "test_verify"
TEST_PATH = Path("eval") / "test_questions.jsonl"

N_PER_SPLIT = 80
MIN_WORDS, MAX_WORDS = 20, 250
SEED = 2026


def sample() -> None:
    articles = corpus.with_split(corpus.load_articles())
    words = articles["contenu"].str.split().str.len()
    pool = articles[(words >= MIN_WORDS) & (words <= MAX_WORDS)]
    picked = []
    for split in ("heldout", "train"):
        part = pool[pool["split"] == split]
        # Même proportion Code civil / Code de la consommation que dans le split.
        for code, sub in part.groupby("code"):
            n = round(N_PER_SPLIT * len(sub) / len(part))
            picked.append(sub.sample(n=n, random_state=SEED))
    rows = [
        {
            "article_key": a.article_key,
            "code": a.code,
            "num": a.num,
            "split": a.split,
            "titre": a.titre,
            "contenu": a.contenu,
        }
        for df in picked
        for a in df.itertuples(index=False)
    ]
    with CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} articles -> {CANDIDATES_PATH}")


def build() -> None:
    """Assemble les brouillons gardés par le rédacteur ET validés par le second passage."""
    candidates = {
        r["article_key"]: r
        for r in map(json.loads, CANDIDATES_PATH.read_text(encoding="utf-8").splitlines())
    }
    verdicts = {}
    for path in sorted(VERIFY_DIR.glob("out_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                verdicts[v["article_key"]] = v
    rows, rejected = [], []
    for path in sorted(DRAFTS_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            draft = json.loads(line)
            if not draft.get("keep"):
                continue
            verdict = verdicts.get(draft["article_key"], {})
            if not verdict.get("ok") or verdict.get("leak"):
                rejected.append((draft["article_key"], verdict.get("reason") or "jargon de l'article"))
                continue
            art = candidates[draft["article_key"]]
            rows.append(
                {
                    "id": f"test-{len(rows) + 1:03d}",
                    "question": draft["question"].strip(),
                    "expected": [{"code": art["code"], "num": art["num"]}],
                    "split": art["split"],
                    "category": "answerable",
                }
            )
    TEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEST_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_split = {s: sum(r["split"] == s for r in rows) for s in ("heldout", "train")}
    print(f"{len(rows)} questions -> {TEST_PATH} {by_split}")
    print(f"{len(rejected)} rejetées à la vérification :")
    for key, reason in rejected:
        print(f"  {key} : {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sample", "build"])
    args = parser.parse_args()
    {"sample": sample, "build": build}[args.command]()


if __name__ == "__main__":
    main()
