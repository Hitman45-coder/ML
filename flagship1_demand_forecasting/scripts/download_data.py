from pathlib import Path

from demand_forecasting.data import download_dataset

path = download_dataset(Path("data/raw/hour.csv"))
print(f"Downloaded {path.resolve()}")
