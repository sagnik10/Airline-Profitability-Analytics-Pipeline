import pandas as pd
import time

def load_csv_chunked(path, chunksize):
    start = time.time()
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"{path} loaded in {round(time.time() - start, 2)}s | rows={df.shape[0]}")
    return df