# Machine Learning Model for predicting a next destination

[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Booking.com%20Multi--Destination%20Trips-yellow)](https://huggingface.co/datasets/Booking-com/multi-destination-trip-dataset)

## ⚠️ WIP
This project is currently in progress. It is not ready yet and the model is very bad at the moment. I'm working on it during my free-time. :)

![Architecture](.github/model_architecture.png)

This project provides a machine learning model built on data pipelines using the [Booking.com Multi-Destination Trip Dataset](https://huggingface.co/datasets/Booking-com/multi-destination-trip-dataset) to predict the four most likely next destinations for a customer.

![Gradio](.github/gradio.png)

# Quick start with Docker

```bash
docker build -t tripgru .
docker run -p 7860:7860 -v tripgru-data:/app/dataset tripgru
```

On the first run, it downloads the dataset and trains the model (this can take a while). After that, the Gradio app is live at http://localhost:7860.

# Local setup

This project uses `uv` as its dependency manager.

```bash
uv sync
python download.py
python main.py
python app.py
```

It also uses `black` for formatting, which runs before every commit with `pre-commit`.

# Project structure

- `main.py`: Trains a new model
- `inference.py`: Run the trained model on a random trip from the dataset
- `app.py`: Gradio app to play with the model
- `notebook.ipynb`: Explains and dives a bit in the dataset
- `report/report.pdf`: Report for this project