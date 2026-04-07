from datasets import load_dataset

dataset = load_dataset("Booking-com/multi-destination-trip-dataset")

print(dataset)
print(dataset["train"][0])