"""Corpus : articles du Code civil et du Code de la consommation, découpés comme dans le RAG.

Usage :
    python corpus.py            # statistiques et répartition train / held-out

Le catalogue `data/articles.parquet` est celui du projet 1 (Tricoteuses/LEGI, révision figée,
dédoublonné) : `make corpus` le copie depuis legal-rag-agent. Les passages sont formatés
exactement comme les chunks de l'index FAISS du RAG, pour que le modèle fine-tuné voie à
l'entraînement le même texte qu'en production.

Répartition held-out : on met de côté des chapitres entiers, pas des articles isolés. Les
articles voisins partagent leur vocabulaire et se renvoient les uns aux autres ; un split
article par article laisserait fuir ce vocabulaire. La partie législative (L.) et la partie
réglementaire (R., D.) d'un même chapitre du Code de la consommation forment un seul groupe.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
ARTICLES_PATH = DATA_DIR / "articles.parquet"

# Mêmes valeurs que legal-rag-agent/ingest.py.
MAX_WORDS = 300
_SENTENCE_RE = re.compile(
    r"(?<=[.;:])(?<!\bart\.)(?<!\b[LRDA]\.)\s*(?=\d+°|[a-z]\)\s|[A-ZÀÂÉÈÊÎÔÛÇ«])"
)

HELDOUT_FRACTION = 0.10
SPLIT_SEED = 13


def load_articles(path: Path = ARTICLES_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["article_key"] = df["code"] + "|" + df["num"]
    df["group"] = df.apply(lambda r: chapter_group(r["code"], r["titre"]), axis=1)
    return df


def chapter_group(code: str, titre: str) -> str:
    """Chemin jusqu'au chapitre inclus, sans la « Partie » (législative / réglementaire)."""
    levels = [lv.strip() for lv in (titre or "").split(" > ") if lv.strip()]
    levels = [lv for lv in levels if not lv.lower().startswith("partie")]
    kept = []
    for lv in levels:
        kept.append(_level_label(lv))
        if lv.startswith("Chapitre"):
            break
    return code + "|" + " > ".join(kept)


def _level_label(level: str) -> str:
    """« Livre III : CRÉDIT » -> « Livre III » : les intitulés varient entre parties L. et R."""
    return level.split(":", 1)[0].strip().rstrip(".")


def split_long(text: str, max_words: int = MAX_WORDS) -> list[str]:
    """Copie de legal-rag-agent/ingest.py : morceaux de ≤ max_words, une phrase de chevauchement."""
    if len(text.split()) <= max_words:
        return [text]
    sentences = []
    for sentence in _SENTENCE_RE.split(text):
        words = sentence.split()
        sentences += [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence]).split()) > max_words:
            chunks.append(" ".join(current))
            overlap = current[-1]
            current = [overlap] if len(f"{overlap} {sentence}".split()) <= max_words else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(articles: pd.DataFrame) -> pd.DataFrame:
    """Un chunk par article (plusieurs pour les articles longs), texte identique à l'index du RAG."""
    rows = []
    for art in articles.itertuples(index=False):
        header = f"{art.code}, art. {art.num}\n{art.titre}"
        for i, part in enumerate(split_long(art.contenu)):
            rows.append({"article_key": art.article_key, "chunk": i, "text": f"{header}\n\n{part}"})
    return pd.DataFrame(rows)


def article_passage(art) -> str:
    """Passage d'entraînement : en-tête + début de l'article (premier chunk)."""
    return f"{art.code}, art. {art.num}\n{art.titre}\n\n{split_long(art.contenu)[0]}"


def heldout_groups(articles: pd.DataFrame, fraction: float = HELDOUT_FRACTION, seed: int = SPLIT_SEED) -> set[str]:
    """Chapitres tirés au hasard jusqu'à couvrir `fraction` des articles."""
    sizes = articles.groupby("group").size()
    groups = sorted(sizes.index)
    random.Random(seed).shuffle(groups)
    target, total, chosen = fraction * len(articles), 0, set()
    for g in groups:
        if total >= target:
            break
        chosen.add(g)
        total += sizes[g]
    return chosen


def with_split(articles: pd.DataFrame) -> pd.DataFrame:
    held = heldout_groups(articles)
    return articles.assign(split=articles["group"].map(lambda g: "heldout" if g in held else "train"))


def main() -> None:
    articles = with_split(load_articles())
    chunks = build_chunks(articles)
    print(f"{len(articles)} articles, {len(chunks)} chunks, {articles['group'].nunique()} chapitres")
    print(articles.groupby(["split", "code"]).size().to_string())
    held = articles[articles["split"] == "heldout"]
    print(f"\nChapitres held-out ({held['group'].nunique()}) :")
    for g, n in held.groupby("group").size().sort_values(ascending=False).items():
        print(f"  {n:4d}  {g}")


if __name__ == "__main__":
    main()
