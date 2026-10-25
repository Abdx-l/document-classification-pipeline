# Document Classification Pipeline

Four-class document routing with a calibrated TF-IDF/SVM baseline, optional DistilBERT training, MLflow experiment tracking, held-out confusion matrices, and a persistent review queue for low-confidence predictions.

## Run

Install requirements.txt. Supply a CSV with text,label columns using labels claim,id,invoice,medical. `python train.py dataset.csv` trains and evaluates the baseline; `python transformer_train.py dataset.csv` fine-tunes DistilBERT with optional requirements-transformers.txt. Run `uvicorn app:app`.
