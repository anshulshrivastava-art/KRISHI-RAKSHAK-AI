# ⚙️ Krishi Rakshak AI - Backend & ML Engine

This directory houses the machine learning models, REST API server, agronomic knowledge base, database, and background services.

## Sub-modules
- `api.py`: FastAPI REST server with `/predict`, `/diseases`, `/weather`, and `/history` endpoints.
- `config.py`: Central hyperparameters, thresholds (High: 75%, Med: 45%), and paths.
- `disease_info.json`: Pathology database for 10 crop conditions with symptoms, cultural practices, organic/biological remedies, and verified chemical guidance.
- `krishi_rakshak.db`: Local SQLite database storing scan history.
- `src/`:
  - `preprocessing.py`: Image validation and tensor resizing to (224, 224).
  - `predictor.py`: MobileNetV2 cached model inference and confidence thresholding.
  - `recommendations.py`: Structured remedy and diagnosis retrieval.
  - `weather.py`: Open-Meteo API integration and microclimatic disease risk rules.
  - `database.py`: SQLite manager and analytics aggregation.
- `ml_pipeline/`:
  - `download_dataset.py`: PlantVillage downloader and leak-free partitioner.
  - `train_model.py`: MobileNetV2 transfer learning and fine-tuning.
  - `evaluate_model.py`: Test set evaluation and confusion matrix generator.
- `models/`: Saved model weights (`.keras`), class indices, and evaluation reports.
- `data/`: Train, validation, test, and sample leaf datasets.
- `tests/`: 22 automated Pytest unit tests.

## Running Backend Directly
```powershell
python api.py
```
Or double-click `run_backend.bat`. Interactive API documentation will be available at `http://127.0.0.1:8000/docs`.
