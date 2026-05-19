# Singapore TOTO Analytics & ML Predictor

Automated pipeline for Singapore Pools TOTO results – scraping, analytics, and LSTM-based "prediction" (educational). Includes a Streamlit dashboard.

**Disclaimer:** Lottery draws are random. This project is purely educational.

## Quick Start

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Backfill historical data (optional):
   `python toto_engine.py backfill --from 3900 --to 4050`
4. Scrape the latest draw: `python toto_engine.py scrape`
5. Train the LSTM model (needs >10 draws): `python toto_engine.py train`
6. Launch dashboard: `streamlit run app.py`

## Commands
- `scrape` – add newest draw to CSV
- `backfill --from X --to Y` – fetch historical draws
- `train` – train/retrain the LSTM
- `predict` – print next predictions

## Automation
GitHub Actions update the data daily and retrain the model weekly. The Streamlit app can be deployed on Streamlit Community Cloud.
