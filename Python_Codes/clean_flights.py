import time
import pandas as pd
pd.options.mode.chained_assignment = "raise"

def clean_flights(df):
    start = time.time()

    df = df.loc[df["CANCELLED"] == 0].copy()

    df.loc[:, "DEP_DELAY"] = df.loc[:, "DEP_DELAY"].clip(lower=0)
    df.loc[:, "ARR_DELAY"] = df.loc[:, "ARR_DELAY"].clip(lower=0)

    df = df.loc[df["DISTANCE"].notna() & df["OCCUPANCY_RATE"].notna()].copy()

    print(f"Flights cleaned in {round(time.time() - start, 2)}s | rows={df.shape[0]}")
    return df