"""Génération puis filtrage des paires question / article avec DeepSeek.

Usage :
    python generate.py                 # tous les articles des chapitres d'entraînement
    python generate.py --limit 20      # essai
    python generate.py --split heldout # chapitres held-out -> data/pairs_heldout.jsonl (éval seulement)
    python generate.py --stats         # bilan du filtrage, sans appel réseau

Écrit data/pairs.jsonl : une ligne par question générée, gardée ou non, avec le motif du
rejet. Le fichier sert aussi de cache : un article déjà traité n'est pas renvoyé au LLM.
Les paires held-out ne servent jamais à l'entraînement.

Trois questions par article, en langage courant (le problème mesuré dans le projet 1 est
l'écart entre les mots du citoyen et ceux du code) :
    situation  « J'ai… », situation vécue à la 1re personne
    tiers      même chose pour un proche (« Ma mère… », « Mon voisin… »)
    recherche  requête courte telle qu'on la tape dans un moteur de recherche

Filtrage, dans cet ordre :
    1. règles : pas de numéro d'article ni de nom de code, longueur, recouvrement lexical
       avec l'article (au-delà, la question recopie l'article au lieu de le reformuler) ;
    2. juge LLM, appel séparé : l'article répond-il à la question ?
On n'utilise pas de filtre « le modèle de base retrouve l'article dans son top-k » : il
écarterait précisément les questions difficiles, celles qu'on veut apprendre.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

import corpus

load_dotenv()

PAIRS_PATH = corpus.DATA_DIR / "pairs.jsonl"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
CONCURRENCY = int(os.environ.get("GEN_CONCURRENCY", "16"))
STYLES = ("situation", "tiers", "recherche")

MIN_WORDS, MAX_WORDS = 4, 45
MAX_OVERLAP = 0.6

GENERATE_PROMPT = """Voici un article de droit français.

{passage}

Écris 3 questions que poserait un citoyen sans aucune formation juridique, et auxquelles \
CET article répond. Le citoyen ne connaît ni l'article ni le vocabulaire juridique : il \
décrit sa situation avec des mots de tous les jours.

- "situation" : une situation concrète vécue, à la première personne (10 à 35 mots).
- "tiers" : une situation concrète concernant un proche, un voisin, un collègue (10 à 35 mots).
- "recherche" : ce qu'il taperait dans un moteur de recherche (4 à 10 mots).

Interdits : le numéro de l'article, le nom du code, les termes techniques de l'article \
(paraphrase-les comme le ferait quelqu'un qui ne les connaît pas).

Si l'article est purement technique (renvoi, procédure administrative, composition d'un \
organisme, montant d'amende…) et qu'aucun citoyen ne poserait de question dont il est la \
réponse, renvoie une liste vide.

Réponds en JSON : {{"questions": [{{"style": "situation", "question": "..."}}, \
{{"style": "tiers", "question": "..."}}, {{"style": "recherche", "question": "..."}}]}}"""

JUDGE_PROMPT = """Tu vérifies un jeu de données de recherche juridique.

Article :
{passage}

Questions :
{questions}

Pour chaque question, réponds "oui" si l'article, lu seul, apporte une réponse directe \
à la question ; "non" s'il ne fait qu'aborder le thème, si la réponse est ailleurs, ou si \
la question est incompréhensible.

Réponds en JSON : {{"verdicts": ["oui" | "non", ...]}} (même ordre, même nombre)."""

_NUM_RE = re.compile(r"\b(?:art(?:icle)?s?\.?\s*)?[LRD]\.?\s?\d+(?:-\d+)+\b|\barticles?\s+\d+", re.IGNORECASE)
_CODE_RE = re.compile(r"\bcode (?:civil|de la consommation)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z]{4,}")


def _words(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return set(_WORD_RE.findall(text))


def rule_check(question: str, contenu: str) -> str | None:
    """Motif de rejet, ou None si la question passe les règles."""
    n = len(question.split())
    if n < MIN_WORDS or n > MAX_WORDS:
        return "longueur"
    if _NUM_RE.search(question) or _CODE_RE.search(question):
        return "cite l'article ou le code"
    q = _words(question)
    if q and len(q & _words(contenu)) / len(q) > MAX_OVERLAP:
        return "recopie l'article"
    return None


async def _chat(client: AsyncOpenAI, prompt: str, temperature: float) -> dict:
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"},
        max_tokens=800,
    )
    return json.loads(resp.choices[0].message.content)


async def process_article(client: AsyncOpenAI, art) -> list[dict]:
    passage = corpus.article_passage(art)
    base = {"article_key": art.article_key}
    generated = await _chat(client, GENERATE_PROMPT.format(passage=passage), temperature=0.8)
    items = [
        {"style": str(q.get("style", "")), "question": str(q.get("question", "")).strip()}
        for q in generated.get("questions", [])
        if isinstance(q, dict) and str(q.get("question", "")).strip()
    ]
    if not items:
        return [{**base, "style": None, "question": None, "kept": False, "reason": "article non questionnable"}]

    rows = [{**base, **it, "reason": rule_check(it["question"], art.contenu)} for it in items]
    to_judge = [r for r in rows if r["reason"] is None]
    if to_judge:
        listing = "\n".join(f"{i + 1}. {r['question']}" for i, r in enumerate(to_judge))
        verdicts = (await _chat(client, JUDGE_PROMPT.format(passage=passage, questions=listing), 0.0)).get(
            "verdicts", []
        )
        for i, r in enumerate(to_judge):
            v = str(verdicts[i]).strip().lower() if i < len(verdicts) else ""
            if v != "oui":
                r["reason"] = "juge : l'article ne répond pas"
    for r in rows:
        r["kept"] = r["reason"] is None
    return rows


def pairs_path(split: str) -> Path:
    return PAIRS_PATH if split == "train" else corpus.DATA_DIR / f"pairs_{split}.jsonl"


async def run(split: str, limit: int | None) -> None:
    articles = corpus.with_split(corpus.load_articles())
    articles = articles[articles["split"] == split]
    path = pairs_path(split)
    done = set()
    if path.exists():
        done = {json.loads(line)["article_key"] for line in path.read_text(encoding="utf-8").splitlines()}
    todo = [a for a in articles.itertuples(index=False) if a.article_key not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(done)} articles déjà traités, {len(todo)} à générer")

    client = AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL, max_retries=3, timeout=90)
    sem = asyncio.Semaphore(CONCURRENCY)
    failures = 0

    async def worker(art) -> list[dict]:
        nonlocal failures
        async with sem:
            try:
                return await process_article(client, art)
            except Exception as exc:  # article retenté au prochain lancement
                failures += 1
                print(f"échec {art.article_key} : {exc!r}")
                return []

    with path.open("a", encoding="utf-8") as f:
        for i, coro in enumerate(asyncio.as_completed([worker(a) for a in todo]), 1):
            for row in await coro:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"{i}/{len(todo)}")
    print(f"terminé, {failures} échecs")


def stats(split: str) -> None:
    rows = [json.loads(line) for line in pairs_path(split).read_text(encoding="utf-8").splitlines()]
    questions = [r for r in rows if r["question"]]
    kept = [r for r in questions if r["kept"]]
    print(f"{len({r['article_key'] for r in rows})} articles, {len(questions)} questions, {len(kept)} gardées")
    print(f"{len({r['article_key'] for r in kept})} articles avec au moins une question gardée")
    print("Rejets :", dict(Counter(r["reason"] for r in rows if not r["kept"])))
    print("Gardées par style :", dict(Counter(r["style"] for r in kept)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "heldout"], default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()
    if not args.stats:
        asyncio.run(run(args.split, args.limit))
    stats(args.split)


if __name__ == "__main__":
    main()
