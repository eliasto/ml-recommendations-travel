#!/bin/sh
set -e

. /app/.venv/bin/activate

if [ ! -f model.pt ]; then
    echo "First run: downloading dataset..."
    python download.py

    echo "Training model..."
    python main.py
fi

echo "Starting Gradio app..."
exec python app.py
