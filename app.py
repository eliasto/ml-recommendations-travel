import gradio as gr
import pandas as pd
from datetime import date, datetime
from inference import load_model, predict_next_cities, get_random_trip

model, vocabs = load_model()
df = pd.read_csv("dataset/train_set.csv")

hotel_countries = sorted(vocabs["hotel_country"].keys())
booker_countries = sorted(vocabs["booker_country"].keys())
device_classes = sorted(vocabs["device_class"].keys())

trip_steps = []


TABLE_COLS = [
    "city_id",
    "hotel_country",
    "booker_country",
    "device_class",
    "checkin",
    "checkout",
    "stay_nights",
    "checkin_month",
]


def make_table():
    rows = [
        [
            s["city_id"],
            s["hotel_country"],
            s["booker_country"],
            s["device_class"],
            s["checkin"],
            s["checkout"],
            s["stay_nights"],
            s["checkin_month"],
        ]
        for s in trip_steps
    ]
    return pd.DataFrame(rows, columns=TABLE_COLS)


def add_step(city_id, hotel_country, booker_country, device_class, checkin, checkout):
    checkin_dt = (
        datetime.strptime(checkin, "%Y-%m-%d") if isinstance(checkin, str) else checkin
    )
    checkout_dt = (
        datetime.strptime(checkout, "%Y-%m-%d")
        if isinstance(checkout, str)
        else checkout
    )
    stay_nights = (checkout_dt - checkin_dt).days
    if stay_nights < 0:
        return make_table(), "checkout must be after checkin", ""

    trip_steps.append(
        {
            "city_id": int(city_id),
            "hotel_country": hotel_country,
            "booker_country": booker_country,
            "device_class": device_class,
            "checkin": (
                str(checkin_dt.date()) if hasattr(checkin_dt, "date") else str(checkin)
            ),
            "checkout": (
                str(checkout_dt.date())
                if hasattr(checkout_dt, "date")
                else str(checkout)
            ),
            "stay_nights": stay_nights,
            "checkin_month": checkin_dt.month,
        }
    )
    return (
        make_table(),
        f"{len(trip_steps)} step(s) added",
        "",
    )


def clear_trip():
    trip_steps.clear()
    return pd.DataFrame(columns=TABLE_COLS), "", ""


def run_prediction():
    if not trip_steps:
        return "Add at least one booking step first."
    preds = predict_next_cities(model, vocabs, trip_steps)
    lines = [f"  #{i+1}  city_id {c}" for i, c in enumerate(preds)]
    return "Top 4 predicted next destinations:\n" + "\n".join(lines)


def random_trip_demo():
    trip_steps.clear()
    trip_id, trip = get_random_trip(df)

    input_bookings = trip.iloc[:-1]
    actual_next = trip.iloc[-1]["city_id"]

    for _, row in input_bookings.iterrows():
        trip_steps.append(
            {
                "city_id": int(row["city_id"]),
                "hotel_country": row["hotel_country"],
                "booker_country": row["booker_country"],
                "device_class": row["device_class"],
                "checkin": (
                    str(row["checkin"].date())
                    if hasattr(row["checkin"], "date")
                    else str(row["checkin"])
                ),
                "checkout": (
                    str(row["checkout"].date())
                    if hasattr(row["checkout"], "date")
                    else str(row["checkout"])
                ),
                "stay_nights": int(row["stay_nights"]),
                "checkin_month": int(row["checkin_month"]),
            }
        )

    table = make_table()

    preds = predict_next_cities(model, vocabs, trip_steps)
    hit = "yes" if actual_next in preds else "no"
    result = (
        f"Trip: {trip_id}\n"
        f"Actual next city: {actual_next}\n"
        f"Top 4 predicted:  {preds}\n"
        f"Correct in top 4: {hit}"
    )
    return table, f"Loaded {len(trip_steps)} steps from random trip", result


with gr.Blocks(title="TripGRU") as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# TripGRU - Prediction of the next destination")
            gr.Markdown(
                "Build a trip step by step or load a random one from the dataset, "
                "then predict the next destination."
            )
        random_btn = gr.Button("Random trip from dataset", variant="huggingface")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Add a booking step")
            city_id = gr.Number(label="city_id", precision=0)
            hotel_country = gr.Dropdown(hotel_countries, label="hotel_country")
            booker_country = gr.Dropdown(booker_countries, label="booker_country")
            device_class = gr.Dropdown(device_classes, label="device_class")
            with gr.Row():
                checkin = gr.DateTime(label="Check-in", include_time=False)
                checkout = gr.DateTime(label="Check-out", include_time=False)

            with gr.Row():
                add_btn = gr.Button("Add step")
                clear_btn = gr.Button("Clear trip")

        with gr.Column(scale=2):
            gr.Markdown("### Current trip")
            trip_table = gr.Dataframe(
                headers=TABLE_COLS,
                interactive=False,
            )
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Prediction")
            predict_btn = gr.Button(
                "Predict next destination", variant="primary", size="lg"
            )
            result = gr.Textbox(label="Result", interactive=False, lines=5)

    add_btn.click(
        add_step,
        inputs=[
            city_id,
            hotel_country,
            booker_country,
            device_class,
            checkin,
            checkout,
        ],
        outputs=[trip_table, status, result],
    )
    clear_btn.click(clear_trip, outputs=[trip_table, status, result])
    random_btn.click(random_trip_demo, outputs=[trip_table, status, result])
    predict_btn.click(run_prediction, outputs=[result])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Ocean())
