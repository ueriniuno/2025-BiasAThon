#data_loader.py
import pandas as pd

def load_data(file_path: str, sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    if sample_size:
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    return df