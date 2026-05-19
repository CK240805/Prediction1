"""
toto_engine.py – Unified core for Singapore TOTO scraper, analytics, ML, and backfill.
"""
import re
import json
import time
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------------
#  Configuration
# -------------------------------
CSV_PATH = Path(__file__).parent / "data" / "toto_results.csv"
MODEL_PATH = Path(__file__).parent / "ml" / "model_weights.pt"
BASE_URL = "https://www.singaporepools.com.sg/en/product/sr/Pages/toto_results.aspx"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SEQ_LENGTH = 10
NUMBERS = 49
HIDDEN_DIM = 128
NUM_LAYERS = 2

# ============================================
#  1. DATA SCRAPING & CSV MANAGEMENT
# ============================================

def fetch_page(draw_no: int = 4044) -> str:
    """Fetch results page HTML."""
    resp = requests.get(
        BASE_URL,
        params={"DrawNo": draw_no},
        headers={"User-Agent": USER_AGENT},
        timeout=15
    )
    resp.raise_for_status()
    return resp.text

def extract_draw_json(html: str) -> dict:
    """Extract the JavaScript drawResult object."""
    pattern = r'var\s+drawResult\s*=\s*(\{.*?\});'
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        raise ValueError("Could not find drawResult JSON")
    raw = match.group(1)
    # Clean possible trailing commas
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)
    return json.loads(raw)

def parse_draw(json_data: dict) -> dict:
    """Convert JSON to a flat dict matching the new CSV schema."""
    draw_no = int(json_data["DrawNumber"])
    draw_date = datetime.strptime(json_data["DrawDate"], "%d/%m/%Y").strftime("%Y-%m-%d")
    winning_numbers = sorted([int(n) for n in json_data["WinningNumbers"]])
    additional = int(json_data["AdditionalNumber"])
    if len(winning_numbers) != 6 or not (1 <= additional <= 49):
        raise ValueError("Invalid numbers")
    return {
        "draw_no": draw_no,          # first column as per new header
        "date": draw_date,           # renamed from draw_date
        "n1": winning_numbers[0],
        "n2": winning_numbers[1],
        "n3": winning_numbers[2],
        "n4": winning_numbers[3],
        "n5": winning_numbers[4],
        "n6": winning_numbers[5],
        "additional": additional,
    }

# The new CSV header is: draw_no,date,n1,n2,n3,n4,n5,n6,additional

def update_csv():
    """Fetch latest draw, append to CSV if new."""
    columns = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "additional"]
    if CSV_PATH.exists():
        try:
            df = pd.read_csv(CSV_PATH, parse_dates=["date"])
            df["draw_no"] = df["draw_no"].astype(int)
        except Exception as e:
            print(f"Warning: failed to read existing CSV ({e}). Starting fresh.")
            df = pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(columns=columns)

    latest_known = df["draw_no"].max() if not df.empty else 4000
    html = fetch_page(latest_known + 1)
    try:
        draw_json = extract_draw_json(html)
    except ValueError:
        print("JSON extraction failed.")
        return
    new_draw = parse_draw(draw_json)
    if new_draw["draw_no"] in df["draw_no"].values:
        print(f"Draw {new_draw['draw_no']} already exists.")
        return
    df = pd.concat([df, pd.DataFrame([new_draw])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Added draw {new_draw['draw_no']}.")

def backfill_draws(start_draw: int, end_draw: int):
    """Fetch and append historical draws from start_draw to end_draw (inclusive)."""
    columns = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "additional"]
    df = load_data() if CSV_PATH.exists() else pd.DataFrame(columns=columns)
    existing_draws = set(df["draw_no"]) if not df.empty else set()

    for draw_no in range(start_draw, end_draw + 1):
        if draw_no in existing_draws:
            print(f"Draw {draw_no} already in CSV. Skipping.")
            continue
        try:
            html = fetch_page(draw_no)
            json_data = extract_draw_json(html)
            row = parse_draw(json_data)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            existing_draws.add(draw_no)
            print(f"Added draw {draw_no}")
        except Exception as e:
            print(f"Failed to fetch draw {draw_no}: {e}")
        time.sleep(1)
    df.to_csv(CSV_PATH, index=False)
    print(f"Backfill complete. Total draws: {len(df)}")


# ============================================
#  2. ANALYTICS
# ============================================

def load_data() -> pd.DataFrame:
    """Load CSV and return DataFrame."""
    if not CSV_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    return df

def number_frequency_chart(df: pd.DataFrame, recent_n: int = None) -> go.Figure:
    """Bar chart of number frequencies."""
    subset = df if recent_n is None else df.tail(recent_n)
    nums = subset[["n1","n2","n3","n4","n5","n6"]].values.flatten()   # <-- changed
    freq = pd.Series(nums).value_counts().reindex(range(1, 50), fill_value=0)
    fig = px.bar(
        x=freq.index.astype(str),
        y=freq.values,
        labels={"x": "Number", "y": "Frequency"},
        title=f"Number Frequency (last {recent_n or len(df)} draws)"
    )
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    return fig

def overdue_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame of how many draws since each number last appeared."""
    if df.empty:
        return pd.DataFrame(columns=["number", "draws_since"])
    results = []
    for num in range(1, 50):
        # Check columns n1..n6
        mask = df[
            (df["n1"]==num) | (df["n2"]==num) | (df["n3"]==num) |
            (df["n4"]==num) | (df["n5"]==num) | (df["n6"]==num)
        ]
        if mask.empty:
            gap = 9999
        else:
            last_app = mask["draw_no"].max()
            gap = df[df["draw_no"] > last_app].shape[0]
        results.append({"number": num, "draws_since": gap})
    return pd.DataFrame(results).sort_values("draws_since", ascending=False)

def pair_heatmap(df: pd.DataFrame, recent_n: int = 200) -> go.Figure:
    """Pair co-occurrence heatmap."""
    subset = df.tail(recent_n)
    pair_count = np.zeros((49, 49), dtype=int)
    for _, row in subset.iterrows():
        drawn = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"], row["n6"]]
        for i in range(len(drawn)):
            for j in range(i+1, len(drawn)):
                a, b = drawn[i]-1, drawn[j]-1
                pair_count[a, b] += 1
                pair_count[b, a] += 1
    labels = [str(i) for i in range(1, 50)]
    fig = go.Figure(data=go.Heatmap(
        z=pair_count,
        x=labels, y=labels,
        colorscale="Viridis",
        hovertemplate="Numbers: %{x} & %{y}<br>Count: %{z}<extra></extra>"
    ))
    fig.update_layout(title=f"Pair Co-occurrence (last {recent_n} draws)")
    return fig

def hot_cold_table(df: pd.DataFrame, top_n: int = 10):
    """Return DataFrame of top hot/cold numbers."""
    if df.empty:
        return pd.DataFrame()
    nums = df[["n1","n2","n3","n4","n5","n6"]].values.flatten()
    freq = pd.Series(nums).value_counts().reindex(range(1,50), fill_value=0)
    hot = freq.nlargest(top_n).reset_index()
    hot.columns = ["number", "frequency"]
    hot["type"] = "Hot"
    cold = freq.nsmallest(top_n).reset_index()
    cold.columns = ["number", "frequency"]
    cold["type"] = "Cold"
    return pd.concat([hot, cold], ignore_index=True)

def weighted_lucky_pick(df: pd.DataFrame):
    """Returns 6 numbers sampled according to empirical frequency."""
    if df.empty:
        return sorted(np.random.choice(range(1,50), size=6, replace=False).tolist())
    nums = df[["n1","n2","n3","n4","n5","n6"]].values.flatten()
    freq = pd.Series(nums).value_counts().reindex(range(1,50), fill_value=0)
    prob = freq / freq.sum()
    pick = np.random.choice(prob.index, size=6, replace=False, p=prob.values)
    return sorted(pick.tolist())


# ============================================
#  3. MACHINE LEARNING (LSTM)
# ============================================

class TotoDataset(Dataset):
    """Converts draws to multi‑hot sequence windows."""
    def __init__(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH):
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length
        self.draw_vectors = []
        for _, row in df.iterrows():
            vec = np.zeros(NUMBERS, dtype=np.float32)
            for col in ["n1","n2","n3","n4","n5","n6"]:   # <-- changed
                vec[int(row[col]) - 1] = 1.0
            self.draw_vectors.append(vec)
        self.draw_vectors = np.array(self.draw_vectors)
        self.valid_starts = list(range(len(self.draw_vectors) - seq_length))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        x = self.draw_vectors[start:start + self.seq_length]
        y = self.draw_vectors[start + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TotoPredictor(nn.Module):
    def __init__(self, input_dim=NUMBERS, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])   # last time step
        return out  # raw logits


def train_model(epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    """Train the LSTM model and save weights."""
    df = load_data()
    if len(df) < SEQ_LENGTH + 2:
        raise ValueError("Not enough data to train.")
    dataset = TotoDataset(df, seq_length=SEQ_LENGTH)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TotoPredictor().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience = 10
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_set)

        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            MODEL_PATH.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            print("  -> Model improved, saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(torch.load(MODEL_PATH))
    return model


def predict_lstm() -> list | None:
    """Predict next draw using trained LSTM weights. Returns 6 numbers or None."""
    if not MODEL_PATH.exists():
        return None
    df = load_data()
    if len(df) < SEQ_LENGTH:
        return None
    recent = df.tail(SEQ_LENGTH)
    seq = np.zeros((1, SEQ_LENGTH, NUMBERS), dtype=np.float32)
    for i, (_, row) in enumerate(recent.iterrows()):
        for col in ["n1","n2","n3","n4","n5","n6"]:   # <-- changed
            seq[0, i, int(row[col]) - 1] = 1.0
    device = torch.device("cpu")
    model = TotoPredictor()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(seq)).squeeze(0)
        probs = torch.sigmoid(logits).numpy()
    top6 = np.argsort(probs)[-6:] + 1
    return sorted(top6.tolist())


def recent_sequence() -> np.ndarray:
    """Return the most recent SEQ_LENGTH draws as multi-hot array."""
    df = load_data()
    if len(df) < SEQ_LENGTH:
        return None
    recent = df.tail(SEQ_LENGTH)
    seq = np.zeros((SEQ_LENGTH, NUMBERS), dtype=np.float32)
    for i, (_, row) in enumerate(recent.iterrows()):
        for col in ["n1","n2","n3","n4","n5","n6"]:
            seq[i, int(row[col]) - 1] = 1.0
    return seq


# ============================================
#  4. MAIN (CLI)
# ============================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TOTO Engine")
    sub = parser.add_subparsers(dest="command")

    scrape_parser = sub.add_parser("scrape", help="Update CSV with latest draw")
    train_parser = sub.add_parser("train", help="Train LSTM model")
    train_parser.add_argument("--epochs", type=int, default=50)
    predict_parser = sub.add_parser("predict", help="Print predictions")

    backfill_parser = sub.add_parser("backfill", help="Fetch a range of historical draws")
    backfill_parser.add_argument("--from", dest="start", type=int, required=True)
    backfill_parser.add_argument("--to", dest="end", type=int, required=True)

    args = parser.parse_args()
    if args.command == "scrape":
        update_csv()
    elif args.command == "train":
        train_model(epochs=args.epochs)
    elif args.command == "predict":
        df = load_data()
        print("LSTM prediction:", predict_lstm())
        print("Baseline (weighted):", weighted_lucky_pick(df))
    elif args.command == "backfill":
        backfill_draws(args.start, args.end)
    else:
        parser.print_help()
