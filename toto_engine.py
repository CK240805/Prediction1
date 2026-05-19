"""
toto_engine.py – Unified core for Singapore TOTO scraper, analytics, ML, and backfill.
Ensemble: LSTM + Transformer predictions combined.
Predicts 6 main numbers + additional number (7 unique numbers total).
CSV header: draw_no,date,n1,n2,n3,n4,n5,n6,additional
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

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
NUM_LAYERS = 2          # for LSTM
NHEAD = 4               # for Transformer
NUM_ENCODER_LAYERS = 2  # for Transformer

# ============================================
#  1. DATA SCRAPING & CSV MANAGEMENT
# ============================================

def fetch_page(draw_no: int = 4044) -> str:
    resp = requests.get(BASE_URL, params={"DrawNo": draw_no},
                        headers={"User-Agent": USER_AGENT}, timeout=15)
    resp.raise_for_status()
    return resp.text

def extract_draw_json(html: str) -> dict:
    pattern = r'var\s+drawResult\s*=\s*(\{.*?\});'
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        raise ValueError("Could not find drawResult JSON")
    raw = match.group(1)
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)
    return json.loads(raw)

def parse_draw(json_data: dict) -> dict:
    draw_no = int(json_data["DrawNumber"])
    draw_date = datetime.strptime(json_data["DrawDate"], "%d/%m/%Y").strftime("%Y-%m-%d")
    winning_numbers = sorted([int(n) for n in json_data["WinningNumbers"]])
    additional = int(json_data["AdditionalNumber"])
    if len(winning_numbers) != 6 or not (1 <= additional <= 49):
        raise ValueError("Invalid numbers")
    return {
        "draw_no": draw_no,
        "date": draw_date,
        "n1": winning_numbers[0], "n2": winning_numbers[1], "n3": winning_numbers[2],
        "n4": winning_numbers[3], "n5": winning_numbers[4], "n6": winning_numbers[5],
        "additional": additional,
    }

def update_csv():
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
    if not CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CSV_PATH, parse_dates=["date"])

def number_frequency_chart(df: pd.DataFrame, recent_n: int = None) -> go.Figure:
    subset = df if recent_n is None else df.tail(recent_n)
    nums = subset[["n1","n2","n3","n4","n5","n6"]].values.flatten()
    freq = pd.Series(nums).value_counts().reindex(range(1, 50), fill_value=0)
    fig = px.bar(x=freq.index.astype(str), y=freq.values,
                 labels={"x": "Number", "y": "Frequency"},
                 title=f"Number Frequency (last {recent_n or len(df)} draws)")
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    return fig

def overdue_analysis(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["number", "draws_since"])
    results = []
    for num in range(1, 50):
        mask = df[(df["n1"]==num)|(df["n2"]==num)|(df["n3"]==num)|
                  (df["n4"]==num)|(df["n5"]==num)|(df["n6"]==num)]
        if mask.empty:
            gap = 9999
        else:
            last_app = mask["draw_no"].max()
            gap = df[df["draw_no"] > last_app].shape[0]
        results.append({"number": num, "draws_since": gap})
    return pd.DataFrame(results).sort_values("draws_since", ascending=False)

def pair_heatmap(df: pd.DataFrame, recent_n: int = 200) -> go.Figure:
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
    fig = go.Figure(data=go.Heatmap(z=pair_count, x=labels, y=labels, colorscale="Viridis",
                                    hovertemplate="Numbers: %{x} & %{y}<br>Count: %{z}<extra></extra>"))
    fig.update_layout(title=f"Pair Co-occurrence (last {recent_n} draws)")
    return fig

def hot_cold_table(df: pd.DataFrame, top_n: int = 10):
    if df.empty:
        return pd.DataFrame()
    nums = df[["n1","n2","n3","n4","n5","n6"]].values.flatten()
    freq = pd.Series(nums).value_counts().reindex(range(1,50), fill_value=0)
    hot = freq.nlargest(top_n).reset_index()
    hot.columns = ["number", "frequency"]; hot["type"] = "Hot"
    cold = freq.nsmallest(top_n).reset_index()
    cold.columns = ["number", "frequency"]; cold["type"] = "Cold"
    return pd.concat([hot, cold], ignore_index=True)

def weighted_lucky_pick(df: pd.DataFrame):
    if df.empty:
        main = sorted(np.random.choice(range(1,50), size=6, replace=False).tolist())
        add = int(np.random.choice(list(set(range(1,50)) - set(main))))
        return main, add
    nums = df[["n1","n2","n3","n4","n5","n6"]].values.flatten()
    freq = pd.Series(nums).value_counts().reindex(range(1,50), fill_value=0)
    prob = freq / freq.sum()
    main = list(np.random.choice(prob.index, size=6, replace=False, p=prob.values))
    main.sort()
    remaining_idx = [i for i in range(1,50) if i not in main]
    remaining_prob = prob[remaining_idx] / prob[remaining_idx].sum()
    add = int(np.random.choice(remaining_idx, size=1, p=remaining_prob.values)[0])
    return main, add


# ============================================
#  3. MACHINE LEARNING MODELS
# ============================================

class TotoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_length: int = SEQ_LENGTH):
        self.df = df.reset_index(drop=True)
        self.seq_length = seq_length
        self.draw_vectors = []
        self.additionals = []
        for _, row in df.iterrows():
            vec = np.zeros(NUMBERS, dtype=np.float32)
            for col in ["n1","n2","n3","n4","n5","n6"]:
                vec[int(row[col]) - 1] = 1.0
            self.draw_vectors.append(vec)
            self.additionals.append(int(row["additional"]) - 1)
        self.draw_vectors = np.array(self.draw_vectors)
        self.additionals = np.array(self.additionals)
        self.valid_starts = list(range(len(self.draw_vectors) - seq_length))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        x = self.draw_vectors[start:start + self.seq_length]
        y_main = self.draw_vectors[start + self.seq_length]
        y_add = self.additionals[start + self.seq_length]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y_main, dtype=torch.float32),
                torch.tensor(y_add, dtype=torch.long))


# LSTM model
class LottoLSTM(nn.Module):
    def __init__(self, input_dim=NUMBERS, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc_main = nn.Linear(hidden_dim, input_dim)
        self.fc_add  = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        main_logits = self.fc_main(last)
        add_logits  = self.fc_add(last)
        return main_logits, add_logits


# Transformer model
class LottoTransformer(nn.Module):
    def __init__(self, input_dim=NUMBERS, hidden_dim=HIDDEN_DIM, nhead=NHEAD,
                 num_encoder_layers=NUM_ENCODER_LAYERS, dropout=0.1):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                   dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_main = nn.Linear(hidden_dim, input_dim)
        self.fc_add  = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.input_fc(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        main_logits = self.fc_main(last)
        add_logits  = self.fc_add(last)
        return main_logits, add_logits


# ============================================
#  4. TRAINING (Ensemble: train both, save combined)
# ============================================

def train_one_epoch(model, loader, optimizer, criterion_main, criterion_add, device, clip=1.0):
    model.train()
    total_loss = 0.0
    for xb, y_main, y_add in loader:
        xb, y_main, y_add = xb.to(device), y_main.to(device), y_add.to(device)
        optimizer.zero_grad()
        main_logits, add_logits = model(xb)
        loss = criterion_main(main_logits, y_main) + criterion_add(add_logits, y_add)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def eval_loss(model, loader, criterion_main, criterion_add, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for xb, y_main, y_add in loader:
            xb, y_main, y_add = xb.to(device), y_main.to(device), y_add.to(device)
            main_logits, add_logits = model(xb)
            loss = criterion_main(main_logits, y_main) + criterion_add(add_logits, y_add)
            total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)


def train_single_model(model, train_loader, test_loader, epochs, lr, device, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
    criterion_main = nn.BCEWithLogitsLoss()
    criterion_add = nn.CrossEntropyLoss()

    best_val = float("inf")
    patience = 20
    no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_main, criterion_add, device)
        val_loss = eval_loss(model, test_loader, criterion_main, criterion_add, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[{model_name}] Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")
        if val_loss < best_val:
            best_val = val_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return best_state, best_val


def train_ensemble(epochs: int = 50, batch_size: int = 32, lr: float = 1e-3):
    """Train both LSTM and Transformer, save combined weights."""
    df = load_data()
    if len(df) < SEQ_LENGTH + 2:
        raise ValueError("Not enough data to train.")

    dataset = TotoDataset(df, seq_length=SEQ_LENGTH)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_test = n_total - n_train
    train_set = Subset(dataset, range(0, n_train))
    test_set  = Subset(dataset, range(n_train, n_total))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Train LSTM
    print("\n=== Training LSTM ===")
    lstm_model = LottoLSTM(dropout=0.3).to(device)
    lstm_state, lstm_val = train_single_model(lstm_model, train_loader, test_loader,
                                              epochs, lr, device, "LSTM")
    # Train Transformer
    print("\n=== Training Transformer ===")
    tf_model = LottoTransformer(dropout=0.1).to(device)
    tf_state, tf_val = train_single_model(tf_model, train_loader, test_loader,
                                          epochs, lr, device, "Transformer")

    # Save combined weights
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({"lstm": lstm_state, "transformer": tf_state}, MODEL_PATH)
    print(f"\nEnsemble weights saved to {MODEL_PATH}")

    # Evaluate ensemble on test set
    print("\n=== Ensemble Evaluation on Test Set ===")
    lstm_model.load_state_dict(lstm_state)
    tf_model.load_state_dict(tf_state)
    lstm_model.eval()
    tf_model.eval()

    total_main_matches = 0
    total_add_correct = 0
    n_samples = 0
    with torch.no_grad():
        for xb, y_main, y_add in test_loader:
            xb = xb.to(device)
            # LSTM
            main_log_l, add_log_l = lstm_model(xb)
            # Transformer
            main_log_t, add_log_t = tf_model(xb)
            # Ensemble: average probabilities
            probs_main = (torch.sigmoid(main_log_l) + torch.sigmoid(main_log_t)) / 2.0
            probs_add = (F.softmax(add_log_l, dim=1) + F.softmax(add_log_t, dim=1)) / 2.0

            _, top6 = torch.topk(probs_main, 6, dim=1)
            # Additional: pick highest prob not in top6
            add_preds = []
            for i in range(xb.size(0)):
                top_set = set(top6[i].tolist())
                masked = probs_add[i].clone()
                masked[list(top_set)] = -1
                add_preds.append(torch.argmax(masked).item())
            add_preds = torch.tensor(add_preds, device=device)

            for i in range(xb.size(0)):
                true_main = set(y_main[i].nonzero(as_tuple=True)[0].tolist())
                pred_main = set(top6[i].tolist())
                total_main_matches += len(true_main.intersection(pred_main))
                if add_preds[i].item() == y_add[i].item():
                    total_add_correct += 1
                n_samples += 1

    avg_main = total_main_matches / n_samples if n_samples > 0 else 0
    add_acc = total_add_correct / n_samples if n_samples > 0 else 0
    print(f"Ensemble Test set ({n_samples} samples):")
    print(f"Average main matches (out of 6): {avg_main:.2f}")
    print(f"Additional number accuracy: {add_acc:.4f} ({total_add_correct}/{n_samples})")
    return


# ============================================
#  5. PREDICTION (Ensemble)
# ============================================

def predict_ensemble() -> tuple | None:
    """Predict next draw using ensemble of LSTM and Transformer."""
    if not MODEL_PATH.exists():
        return None
    df = load_data()
    if len(df) < SEQ_LENGTH:
        return None

    recent = df.tail(SEQ_LENGTH)
    seq = np.zeros((1, SEQ_LENGTH, NUMBERS), dtype=np.float32)
    for i, (_, row) in enumerate(recent.iterrows()):
        for col in ["n1","n2","n3","n4","n5","n6"]:
            seq[0, i, int(row[col]) - 1] = 1.0

    device = torch.device("cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    # Expect dict with keys 'lstm' and 'transformer'
    if "lstm" not in checkpoint or "transformer" not in checkpoint:
        print("Saved model is not an ensemble. Falling back to single model.")
        # If it's an old state dict (single model), try to load as LSTM
        model = LottoLSTM()
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        with torch.no_grad():
            main_logits, add_logits = model(torch.tensor(seq))
    else:
        lstm_model = LottoLSTM()
        lstm_model.load_state_dict(checkpoint["lstm"])
        lstm_model.eval()
        tf_model = LottoTransformer()
        tf_model.load_state_dict(checkpoint["transformer"])
        tf_model.eval()

        with torch.no_grad():
            main_l, add_l = lstm_model(torch.tensor(seq))
            main_t, add_t = tf_model(torch.tensor(seq))
            # Ensemble: average probabilities
            probs_main = (torch.sigmoid(main_l) + torch.sigmoid(main_t)) / 2.0
            probs_add = (F.softmax(add_l, dim=1) + F.softmax(add_t, dim=1)) / 2.0
            main_logits = probs_main.squeeze(0)     # we need logits-style for topk, but we already have probs
            add_logits = probs_add.squeeze(0)
    # At this point we have `main_logits` (49-d tensor of probabilities) and `add_logits`
    # `main_logits` might be from sigmoid or ensemble probs; both work for topk.
    top6 = torch.topk(main_logits, 6).indices.tolist()
    # Additional: highest probability not in top6
    add_logits_copy = add_logits.clone()
    for idx in top6:
        add_logits_copy[idx] = -1
    add_pred = int(torch.argmax(add_logits_copy).item())

    main_numbers = sorted([n + 1 for n in top6])
    additional = add_pred + 1
    return main_numbers, additional


# ============================================
#  6. MAIN (CLI)
# ============================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TOTO Engine")
    sub = parser.add_subparsers(dest="command")

    scrape_parser = sub.add_parser("scrape", help="Update CSV with latest draw")

    train_parser = sub.add_parser("train", help="Train ensemble (LSTM + Transformer)")
    train_parser.add_argument("--epochs", type=int, default=50)

    predict_parser = sub.add_parser("predict", help="Print ensemble predictions")

    backfill_parser = sub.add_parser("backfill", help="Fetch a range of historical draws")
    backfill_parser.add_argument("--from", dest="start", type=int, required=True)
    backfill_parser.add_argument("--to", dest="end", type=int, required=True)

    args = parser.parse_args()
    if args.command == "scrape":
        update_csv()
    elif args.command == "train":
        train_ensemble(epochs=args.epochs)
    elif args.command == "predict":
        df = load_data()
        pred = predict_ensemble()
        base_main, base_add = weighted_lucky_pick(df)
        if pred:
            print(f"Ensemble prediction: Main: {pred[0]}  Additional: {pred[1]}")
        else:
            print("Ensemble model not available.")
        print(f"Baseline (weighted): Main: {base_main}  Additional: {base_add}")
    elif args.command == "backfill":
        backfill_draws(args.start, args.end)
    else:
        parser.print_help()
