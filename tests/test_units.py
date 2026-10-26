"""Tests hors-ligne : découpage, répartition held-out, filtres, métriques, préfixes e5."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import corpus
import evaluate
import generate

ROOT = Path(__file__).resolve().parent.parent
HAS_CORPUS = (ROOT / corpus.ARTICLES_PATH).exists()
needs_corpus = pytest.mark.skipif(not HAS_CORPUS, reason="data/articles.parquet absent (make corpus)")


# --------------------------------------------------------------------------- #
# Corpus et répartition
# --------------------------------------------------------------------------- #


def test_groupe_ignore_la_partie_et_les_intitules() -> None:
    leg = "Partie législative nouvelle > Livre II : CONTRATS > Titre II : RÈGLES > Chapitre Ier : Contrats à distance > Section 1 : X"
    reg = "Partie réglementaire nouvelle > Livre II : FORMATION DES CONTRATS > Titre II : Autres > Chapitre Ier : Autre intitulé"
    code = "Code de la consommation"
    assert corpus.chapter_group(code, leg) == corpus.chapter_group(code, reg)
    assert corpus.chapter_group(code, leg) == f"{code}|Livre II > Titre II > Chapitre Ier"


def test_groupe_sans_chapitre_garde_tout_le_chemin() -> None:
    assert corpus.chapter_group("Code civil", "Livre III : X > Titre XIV : Y") == "Code civil|Livre III > Titre XIV"


def test_split_long_respecte_la_limite() -> None:
    text = " ".join(f"Phrase numéro {i} avec quelques mots de plus." for i in range(200))
    parts = corpus.split_long(text, max_words=50)
    assert len(parts) > 1
    assert all(len(p.split()) <= 50 for p in parts)


def test_split_court_inchange() -> None:
    assert corpus.split_long("Un article court.") == ["Un article court."]


@needs_corpus
def test_heldout_par_chapitres_entiers() -> None:
    articles = corpus.with_split(corpus.load_articles())
    by_group = articles.groupby("group")["split"].nunique()
    assert (by_group == 1).all(), "un chapitre est coupé entre train et held-out"
    share = (articles["split"] == "heldout").mean()
    assert 0.08 <= share <= 0.15


@needs_corpus
def test_chunks_au_format_de_l_index() -> None:
    articles = corpus.load_articles()
    chunks = corpus.build_chunks(articles.head(50))
    first = chunks.iloc[0]["text"]
    art = articles.iloc[0]
    assert first.startswith(f"{art['code']}, art. {art['num']}\n{art['titre']}\n\n")


# --------------------------------------------------------------------------- #
# Filtres de génération
# --------------------------------------------------------------------------- #

ARTICLE = "Le consommateur dispose d'un délai de quatorze jours pour exercer son droit de rétractation d'un contrat conclu à distance."


@pytest.mark.parametrize(
    "question",
    [
        "Que dit l'article L221-18 sur mon achat ?",
        "Le Code de la consommation me protège-t-il ?",
        "Que prévoit l'article 1240 ?",
    ],
)
def test_regle_rejette_references(question: str) -> None:
    assert generate.rule_check(question, ARTICLE) == "cite l'article ou le code"


def test_regle_rejette_recopie() -> None:
    q = "consommateur dispose délai quatorze jours exercer droit rétractation contrat conclu distance"
    assert generate.rule_check(q, ARTICLE) == "recopie l'article"


def test_regle_garde_langage_courant() -> None:
    q = "J'ai commandé une veste sur internet et elle ne me plaît pas, combien de temps pour la renvoyer ?"
    assert generate.rule_check(q, ARTICLE) is None


def test_regle_longueur() -> None:
    assert generate.rule_check("veste", ARTICLE) == "longueur"


# --------------------------------------------------------------------------- #
# Métriques
# --------------------------------------------------------------------------- #


def test_recall_et_mrr() -> None:
    items = [{"expected": {"a"}}, {"expected": {"b"}}, {"expected": {"z"}}]
    rankings = [["a", "x"], ["x", "y", "b"], ["x", "y"]]
    scores = evaluate.per_question(rankings, items)
    assert scores["recall@1"].tolist() == [1, 0, 0]
    assert scores["recall@5"].tolist() == [1, 1, 0]
    assert scores["mrr@10"].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_comparaison_appariee_mcnemar() -> None:
    rng = np.random.default_rng(0)
    a = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
    b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=float)
    res = evaluate.paired_comparison(a, b, rng)
    assert res["gained"] == 7 and res["lost"] == 0
    assert res["delta"] == pytest.approx(0.7)
    assert res["mcnemar_p"] < 0.05
    assert res["ci95"][0] > 0


def test_un_article_compte_une_fois_meme_avec_plusieurs_chunks() -> None:
    chunks = pd.DataFrame({"article_key": ["A", "A", "B"], "text": ["x a", "x b", "y c"]})
    searcher = evaluate.Searcher(chunks)
    assert searcher.articles([0, 1, 2]) == ["A", "B"]


# --------------------------------------------------------------------------- #
# Jeu de test et paires
# --------------------------------------------------------------------------- #


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@needs_corpus
def test_jeu_de_test_valide() -> None:
    articles = corpus.with_split(corpus.load_articles()).set_index("article_key")
    rows = _jsonl(ROOT / "eval" / "test_questions.jsonl")
    assert len(rows) >= 100
    for r in rows:
        key = f"{r['expected'][0]['code']}|{r['expected'][0]['num']}"
        assert key in articles.index
        assert articles.loc[key, "split"] == r["split"]
        assert generate.rule_check(r["question"], "") in (None, "longueur")


@pytest.mark.skipif(not (ROOT / "data" / "pairs.jsonl").exists(), reason="paires non générées")
@needs_corpus
def test_paires_sans_fuite_vers_le_test() -> None:
    import train

    pairs = train.load_pairs(None, seed=0)
    assert all(p["anchor"].startswith("query: ") for p in pairs)
    assert all(p["positive"].startswith("passage: ") for p in pairs)
    anchors = {p["anchor"].removeprefix("query: ").strip().lower() for p in pairs}
    test_questions = {r["question"].strip().lower() for r in _jsonl(ROOT / "eval" / "test_questions.jsonl")}
    assert not anchors & test_questions
    # Aucun article des chapitres held-out à l'entraînement (load_pairs le vérifie aussi).
    articles = corpus.with_split(corpus.load_articles())
    held = set(articles.loc[articles["split"] == "heldout", "article_key"])
    kept = {r["article_key"] for r in _jsonl(ROOT / "data" / "pairs.jsonl") if r["kept"]}
    assert not kept & held


def test_prefixes_e5_dans_les_modeles_evalues() -> None:
    for name, (_, q_prefix, p_prefix) in evaluate.MODELS.items():
        if name.startswith("e5"):
            assert (q_prefix, p_prefix) == ("query: ", "passage: ")
