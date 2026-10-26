"""Bonus : LoRA d'un petit LLM (Qwen2.5-0.5B) pour la classification de sentiment (Allociné).

Usage :
    python bonus/lora_sentiment.py                      # les trois systèmes
    python bonus/lora_sentiment.py --n-train 200 --n-test 200   # essai rapide

Trois systèmes, évalués sur le même échantillon du jeu de test :
    tfidf      TF-IDF + régression logistique sur tout le train (160 000 critiques) :
               la référence bon marché qu'un LLM doit battre pour se justifier
    zero-shot  Qwen2.5-0.5B-Instruct sans entraînement : « positive » ou « négative » ?
    lora       Qwen2.5-0.5B + adaptateurs LoRA (rang 16 sur les projections d'attention)
               et tête de classification, entraînés sur n_train critiques

Tout tourne sur CPU : l'entraînement LoRA se limite à quelques milliers d'exemples.
Résultats dans results/lora_sentiment.json et dans MLflow (expérience lora-sentiment).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

import mlflow
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RESULTS_PATH = Path(os.environ.get("LORA_RESULTS", "results/lora_sentiment.json"))
OUT_DIR = Path("models") / "qwen2.5-0.5b-allocine-lora"
MAX_LENGTH = 256
SEED = 0


def metrics(y_true, y_pred) -> dict:
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def run_tfidf(train, test) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=300_000)
    x_train = vec.fit_transform(train["review"])
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(x_train, train["label"])
    return metrics(test["label"], clf.predict(vec.transform(test["review"])))


@torch.no_grad()
def run_zero_shot(test) -> dict:
    tok = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(INSTRUCT_MODEL, dtype=torch.float32).eval()
    # On compare la probabilité des deux réponses au premier token, pas de génération libre.
    pos_id = tok.encode("positive", add_special_tokens=False)[0]
    neg_id = tok.encode("négative", add_special_tokens=False)[0]
    preds = []
    for review in test["review"]:
        review = tok.convert_tokens_to_string(tok.tokenize(review)[:MAX_LENGTH])
        messages = [
            {"role": "system", "content": "Tu classes des critiques de films. Réponds par un seul mot : positive ou négative."},
            {"role": "user", "content": f"Critique : {review}\n\nCette critique est-elle positive ou négative ?"},
        ]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        logits = model(**ids).logits[0, -1]
        preds.append(int(logits[pos_id] > logits[neg_id]))
    return metrics(test["label"], preds)


def run_lora(train, test, args) -> dict:
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2, dtype=torch.float32)
    model.config.pad_token_id = tok.pad_token_id
    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"LoRA : {trainable / 1e6:.2f} M paramètres entraînés / {total / 1e6:.0f} M")

    def encode(batch):
        return tok(batch["review"], truncation=True, max_length=MAX_LENGTH)

    train_tok = train.map(encode, batched=True, remove_columns=["review"]).rename_column("label", "labels")
    test_tok = test.map(encode, batched=True, remove_columns=["review"]).rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir=str(Path("checkpoints") / "lora-sentiment"),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=16,
        learning_rate=2e-4,
        warmup_steps=0.05,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="no",
        report_to=["mlflow"],
        run_name="lora-qwen2.5-0.5b",
        use_cpu=True,
        seed=SEED,
        group_by_length=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        data_collator=DataCollatorWithPadding(tok),
    )
    start = time.time()
    trainer.train()
    train_s = time.time() - start
    preds = trainer.predict(test_tok).predictions.argmax(-1)
    model.save_pretrained(str(OUT_DIR))
    return {**metrics(test["label"], preds), "train_seconds": round(train_s), "trainable_params": trainable}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--systems", nargs="+", default=["tfidf", "zero-shot", "lora"])
    parser.add_argument("--threads", type=int, default=os.cpu_count() // 2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    ds = load_dataset("tblard/allocine")
    test = ds["test"].shuffle(seed=SEED).select(range(args.n_test))
    small_train = ds["train"].shuffle(seed=SEED).select(range(args.n_train))

    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    results["n_test"] = args.n_test
    mlflow.set_experiment("lora-sentiment")
    for system in args.systems:
        start = time.time()
        if system == "tfidf":
            res = run_tfidf(ds["train"], test)
            res["n_train"] = len(ds["train"])
        elif system == "zero-shot":
            res = run_zero_shot(test)
            res["n_train"] = 0
        else:
            with mlflow.start_run(run_name=f"lora-n{args.n_train}"):
                res = run_lora(small_train, test, args)
                mlflow.log_metrics({"test_accuracy": res["accuracy"], "test_f1_macro": res["f1_macro"]})
            res["n_train"] = args.n_train
        res["seconds"] = round(time.time() - start)
        results[system] = res
        print(system, res)
        if system != "lora":
            with mlflow.start_run(run_name=system):
                mlflow.log_params({"system": system, "n_train": res["n_train"], "n_test": args.n_test})
                mlflow.log_metrics({"test_accuracy": res["accuracy"], "test_f1_macro": res["f1_macro"]})
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
