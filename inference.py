import torch
import pandas as pd
from main import TripGRU


def predict_next_cities(
    model: TripGRU,
    vocabs: dict[str, dict],
    trip_features: list[dict],
    device: torch.device | None = None,
    top_k: int = 4,
):
    """
    Predict next city for a partial trip.

    Args:
        trip_features: list of dicts with keys city_id, hotel_country,
                       booker_country, device_class, stay_nights, checkin_month
    Returns:
        List of top_k predicted city_ids.
    """
    if device is None:
        device = next(model.parameters()).device

    inv_city = {v: k for k, v in vocabs["city_id"].items()}
    rows = []
    for step in trip_features:
        rows.append(
            [
                vocabs["city_id"].get(step["city_id"], 0),
                vocabs["hotel_country"].get(step["hotel_country"], 0),
                vocabs["booker_country"].get(step["booker_country"], 0),
                vocabs["device_class"].get(step["device_class"], 0),
                step["stay_nights"],
                step["checkin_month"],
            ]
        )
    x = torch.tensor([rows], dtype=torch.long, device=device)
    lengths = torch.tensor([len(rows)])

    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)
        top_ids = logits.topk(top_k, dim=1).indices[0].tolist()

    return [int(inv_city.get(i, i)) for i in top_ids]


def load_model(path: str = "model.pt"):
    checkpoint = torch.load(path, weights_only=False)
    vocabs = checkpoint["vocabs"]
    vocab_sizes = {k: len(v) for k, v in vocabs.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TripGRU(vocab_sizes).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, vocabs


def get_random_trip(df: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    df["checkin"] = pd.to_datetime(df["checkin"])
    df["checkout"] = pd.to_datetime(df["checkout"])
    df["stay_nights"] = (df["checkout"] - df["checkin"]).dt.days
    df["checkin_month"] = df["checkin"].dt.month

    # Pick a random trip with at least 3 bookings
    id = (
        df.groupby("utrip_id")
        .filter(lambda trip: len(trip) >= 3)["utrip_id"]
        .drop_duplicates()
        .sample(1)
        .iloc[0]
    )
    trip = df[df["utrip_id"] == id].sort_values("checkin")
    return id, trip


if __name__ == "__main__":
    model, vocabs = load_model()

    df = pd.read_csv("dataset/train_set.csv")
    id, trip = get_random_trip(df)
    input_bookings = trip.iloc[:-1]
    actual_next = trip.iloc[-1]["city_id"]

    trip_features = input_bookings[
        [
            "city_id",
            "hotel_country",
            "booker_country",
            "device_class",
            "stay_nights",
            "checkin_month",
        ]
    ].to_dict(orient="records")

    predictions = predict_next_cities(model, vocabs, trip_features)
    print(f"Trip: {id}")
    print(f"Input cities: {input_bookings['city_id'].tolist()}")
    print(f"Actual next city: {actual_next}")
    print(f"Top 4 predicted:  {predictions}")
