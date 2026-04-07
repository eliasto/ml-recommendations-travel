from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Booking-com/multi-destination-trip-dataset",
    repo_type="dataset",
    local_dir="./dataset"
)