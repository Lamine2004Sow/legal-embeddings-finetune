PY ?= .venv/bin/python
RAG_DIR ?= ../legal-rag-agent
RAG_PY ?= $(RAG_DIR)/.venv/bin/python
FT_MODEL ?= $(abspath models/e5-base-legal)
FT_INDEX ?= data/e5-base-legal
BASE_MODEL ?= intfloat/multilingual-e5-base
BASE_INDEX ?= data/e5-base

.PHONY: help install corpus test testset generate train train-small eval index-ft rag-eval rag-compare lora mlflow

help: ## liste des cibles
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## crée .venv (torch CPU) et installe les dépendances
	python3 -m venv .venv
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements.txt

corpus: ## copie le catalogue d'articles du projet 1
	mkdir -p data
	cp $(RAG_DIR)/data/e5-base/articles.parquet data/articles.parquet

test: ## tests hors-ligne
	$(PY) -m pytest

testset: ## reconstruit eval/test_questions.jsonl depuis les brouillons validés
	$(PY) testset.py build

generate: ## paires d'entraînement + paires held-out (DeepSeek)
	$(PY) generate.py
	$(PY) generate.py --split heldout

train: ## fine-tuning d'e5-base (CPU, plusieurs heures)
	$(PY) train.py

train-small: ## fine-tuning d'e5-small
	$(PY) train.py --base intfloat/multilingual-e5-small --out models/e5-small-legal

eval: ## Recall@k / MRR : BM25, bases, modèles fine-tunés
	$(PY) evaluate.py --models e5-base e5-small bge-m3 e5-base-ft --sets test dev synthetic

index-ft: ## index FAISS du RAG avec le modèle fine-tuné
	cd $(RAG_DIR) && $(RAG_PY) ingest.py --model $(FT_MODEL) --out $(FT_INDEX)

rag-eval: ## RAG du projet 1 : {base, fine-tuné} x {avec, sans reformulation}, jeux dev et test
	for q in dev test; do \
	  qopt=$$( [ $$q = test ] && echo "--questions $(abspath eval/test_questions.jsonl)" ); \
	  for ref in 1 0; do \
	    cd $(RAG_DIR) && REFORMULATE=$$ref RETRIEVER_MODEL=$(BASE_MODEL) RETRIEVER_DATA_DIR=$(BASE_INDEX) \
	      $(RAG_PY) eval/run_eval.py --label p2-$$q-base-ref$$ref $$qopt; cd - >/dev/null; \
	    cd $(RAG_DIR) && REFORMULATE=$$ref RETRIEVER_MODEL=$(FT_MODEL) RETRIEVER_DATA_DIR=$(FT_INDEX) \
	      $(RAG_PY) eval/run_eval.py --label p2-$$q-ft-ref$$ref $$qopt; cd - >/dev/null; \
	  done; \
	done

rag-compare: ## tableau comparatif des runs RAG (results/rag.json)
	$(PY) rag_compare.py

lora: ## bonus : LoRA Qwen2.5-0.5B sur Allociné, contre TF-IDF et zero-shot
	$(PY) bonus/lora_sentiment.py

mlflow: ## interface MLflow sur http://127.0.0.1:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
