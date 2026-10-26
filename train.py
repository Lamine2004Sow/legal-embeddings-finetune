"""Fine-tuning de multilingual-e5 sur les paires question / article filtrées.

Usage :
    python train.py                                  # e5-base, réglages par défaut
    python train.py --base intfloat/multilingual-e5-small --out models/e5-small-legal
    python train.py --max-pairs 200 --epochs 1       # essai rapide

Choix (voir README) :
- fine-tuning complet de l'encodeur, embeddings de tokens gelés : 192 M des 278 M de
  paramètres d'e5-base sont la table de vocabulaire (250 k tokens), inutile de la mettre à
  jour pour du français juridique et coûteux en mémoire sur CPU ;
- CachedMultipleNegativesRankingLoss : les autres passages du batch servent de négatifs ;
  la version « cached » donne un batch de 64 négatifs avec la mémoire d'un batch de 16 ;
- NoDuplicatesBatchSampler : un article a jusqu'à 3 questions, deux d'entre elles dans le
  même batch feraient de l'article positif un faux négatif ;
- pas de hard negatives minés : en droit, plusieurs articles répondent souvent à la même
  question (1240 et 1241), un négatif « difficile » est souvent un faux négatif ;
- préfixes « query: » / « passage: » ajoutés explicitement, comme dans le RAG ;
- choix du checkpoint sur le jeu dev (40 questions du projet 1), jamais sur le test.

Les métriques et paramètres sont journalisés dans MLflow (sqlite:///mlflow.db).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "embedding-finetune")

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

import corpus
import evaluate

QUERY_PREFIX, PASSAGE_PREFIX = "query: ", "passage: "


def load_pairs(max_pairs: int | None, seed: int) -> list[dict]:
    """Paires gardées des chapitres d'entraînement, avec le passage formaté comme dans l'index."""
    articles = corpus.with_split(corpus.load_articles()).set_index("article_key", drop=False)
    rows = [json.loads(line) for line in (corpus.DATA_DIR / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = []
    for r in rows:
        if not r["kept"]:
            continue
        art = articles.loc[r["article_key"]]
        assert art["split"] == "train", f"article held-out dans les paires : {r['article_key']}"
        pairs.append(
            {
                "anchor": QUERY_PREFIX + r["question"],
                "positive": PASSAGE_PREFIX + corpus.article_passage(art),
            }
        )
    random.Random(seed).shuffle(pairs)
    return pairs[:max_pairs] if max_pairs else pairs


def dev_evaluator() -> InformationRetrievalEvaluator:
    """Jeu dev (40 questions du projet 1) sur les 5 348 chunks de l'index ; article trouvé = un de ses chunks."""
    chunks = corpus.build_chunks(corpus.load_articles())
    corpus_docs = {str(i): PASSAGE_PREFIX + t for i, t in enumerate(chunks["text"])}
    by_article: dict[str, set[str]] = {}
    for i, key in enumerate(chunks["article_key"]):
        by_article.setdefault(key, set()).add(str(i))
    items = evaluate.load_eval_set("dev")
    queries = {it["id"]: QUERY_PREFIX + it["question"] for it in items}
    relevant = {it["id"]: set().union(*(by_article.get(k, set()) for k in it["expected"])) for it in items}
    return InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus_docs,
        relevant_docs=relevant,
        name="dev",
        accuracy_at_k=[1, 5, 10, 20],
        precision_recall_at_k=[5, 10, 20],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        map_at_k=[10],
        batch_size=32,
        show_progress_bar=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="intfloat/multilingual-e5-base")
    parser.add_argument("--out", type=Path, default=Path("models/e5-base-legal"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=os.cpu_count() // 2)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    pairs = load_pairs(args.max_pairs, args.seed)
    train_ds = Dataset.from_list(pairs)
    print(f"{len(train_ds)} paires d'entraînement")

    model = SentenceTransformer(args.base, device="cpu")
    model.max_seq_length = args.max_seq_length
    model[0].auto_model.embeddings.word_embeddings.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"paramètres entraînés : {trainable / 1e6:.0f} M / {total / 1e6:.0f} M")

    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch)
    evaluator = dev_evaluator()

    run_name = args.out.name
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(Path("checkpoints") / run_name),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=0.1,
        weight_decay=0.01,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_cosine_mrr@10",
        logging_steps=10,
        seed=args.seed,
        report_to=["mlflow"],
        run_name=run_name,
        use_cpu=True,
        dataloader_drop_last=True,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator,
    )
    print("dev avant entraînement :", {k: round(v, 4) for k, v in evaluator(model).items() if "mrr" in k or "accuracy" in k})
    trainer.train()

    final = evaluator(model)
    print("dev après entraînement (meilleur checkpoint) :", {k: round(v, 4) for k, v in final.items() if "mrr" in k or "accuracy" in k})
    args.out.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    (args.out / "training_args.json").write_text(json.dumps({**vars(args), "out": str(args.out), "n_pairs": len(train_ds)}, indent=2))
    print(f"modèle écrit dans {args.out}")


if __name__ == "__main__":
    main()
