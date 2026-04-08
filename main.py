import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

HIDDEN_DIM = 256


def build_vocab(series: pd.Series) -> dict:
    """Map unique values to contiguous ints starting at 1 (0 = padding)."""
    unique = sorted(series.unique())
    return {v: i + 1 for i, v in enumerate(unique)}


def load_trips(csv_path: str = "dataset/train_set.csv"):
    df = pd.read_csv(csv_path)
    df["checkin"] = pd.to_datetime(df["checkin"])
    df["checkout"] = pd.to_datetime(df["checkout"])
    df["stay_nights"] = (df["checkout"] - df["checkin"]).dt.days
    df["checkin_month"] = df["checkin"].dt.month

    vocabs = {
        "city_id": build_vocab(df["city_id"]),
        "hotel_country": build_vocab(df["hotel_country"]),
        "booker_country": build_vocab(df["booker_country"]),
        "device_class": build_vocab(df["device_class"]),
    }

    for col, vocab in vocabs.items():
        df[col + "_enc"] = df[col].map(vocab)

    # Group by trip, sort by checkin, build sequences
    trips = []
    for _, group in df.sort_values("checkin").groupby("utrip_id"):
        # We need at least 2 bookings to predict the next destination
        if len(group) < 2:
            continue
        seq = group[
            [
                "city_id_enc",
                "hotel_country_enc",
                "booker_country_enc",
                "device_class_enc",
                "stay_nights",
                "checkin_month",
            ]
        ].values
        target = seq[-1, 0]
        features = seq[:-1]
        trips.append((features, target))

    return trips, vocabs


class TripDataset(Dataset):
    def __init__(self, trips):
        self.trips = trips

    def __len__(self):
        return len(self.trips)

    def __getitem__(self, idx):
        features, target = self.trips[idx]
        return torch.tensor(features, dtype=torch.long), torch.tensor(
            target, dtype=torch.long
        )


def collate_trips(batch):
    seqs, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded, targets, lengths


class TripGRU(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict,
        embed_dim: int = 32,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 1,
    ):
        super().__init__()
        # +1 everywhere because index 0 is reserved for padding
        self.city_emb = nn.Embedding(
            vocab_sizes["city_id"] + 1, embed_dim, padding_idx=0
        )
        self.country_emb = nn.Embedding(
            vocab_sizes["hotel_country"] + 1, embed_dim, padding_idx=0
        )
        self.booker_emb = nn.Embedding(
            vocab_sizes["booker_country"] + 1, embed_dim, padding_idx=0
        )
        self.device_emb = nn.Embedding(
            vocab_sizes["device_class"] + 1, embed_dim, padding_idx=0
        )

        # 4 embeddings + stay_nights + checkin_month
        # see .github/model_architecture.png
        input_dim = embed_dim * 4 + 2
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_sizes["city_id"] + 1)

    def forward(self, x, lengths):
        # x is (batch, seq_len, 6): 4 categorical cols then 2 numerical
        city = self.city_emb(x[:, :, 0])
        country = self.country_emb(x[:, :, 1])
        booker = self.booker_emb(x[:, :, 2])
        device = self.device_emb(x[:, :, 3])
        numerical = x[:, :, 4:].float()

        combined = torch.cat([city, country, booker, device, numerical], dim=-1)

        packed = pack_padded_sequence(
            combined, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return self.fc(hidden[-1])


def train_model(
    csv_path: str = "dataset/train_set.csv",
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    hidden_dim: int = HIDDEN_DIM,
    embed_dim: int = 32,
):
    # MPS on Apple Silicon gave terrible accuracy (~10%), probably
    # a precision bug with packed RNNs. Sticking with CPU for now.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("loading trips...")
    trips, vocabs = load_trips(csv_path)
    vocab_sizes = {k: len(v) for k, v in vocabs.items()}
    print(f"  {len(trips)} trips, {vocab_sizes['city_id']} unique cities")

    # Train / validation split (90/10)
    # 67 is for Bas-Rhin not that 67 lol
    np.random.seed(67)
    indices = np.random.permutation(len(trips))
    split = int(0.9 * len(trips))
    train_trips = [trips[i] for i in indices[:split]]
    val_trips = [trips[i] for i in indices[split:]]

    train_loader = DataLoader(
        TripDataset(train_trips),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trips,
    )
    val_loader = DataLoader(
        TripDataset(val_trips),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trips,
    )

    model = TripGRU(vocab_sizes, embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y, lengths in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        correct_top4, total = 0, 0
        with torch.no_grad():
            for x, y, lengths in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths)
                top4 = logits.topk(4, dim=1).indices
                correct_top4 += (top4 == y.unsqueeze(1)).any(dim=1).sum().item()
                total += y.size(0)

        # TODO: Stop when val_top4_acc doesn't improve?
        print(
            f"Epoch {epoch}/{epochs}  "
            f"loss={total_loss / n_batches:.4f}  "
            f"val_top4_acc={correct_top4 / total:.4f}"
        )

    return model, vocabs


if __name__ == "__main__":
    model, vocabs = train_model()
    torch.save({"model_state": model.state_dict(), "vocabs": vocabs}, "model.pt")
    print("\nModel saved to model.pt")
