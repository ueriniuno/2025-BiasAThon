import pandas as pd

def load_data(file_path: str, sample_size: int = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    if sample_size is not None:
        df = df.sample(n=min(sample_size, len(df)), random_state=seed)

    return df