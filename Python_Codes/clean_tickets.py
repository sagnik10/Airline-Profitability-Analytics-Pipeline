import time

def clean_tickets(df):
    start = time.time()
    df = df[df["ROUNDTRIP"] == 1]
    df = df[(df["YEAR"] == 2019) & (df["QUARTER"] == 1)]
    df = df[df["ITIN_FARE"].notna() & df["PASSENGERS"].notna()]
    print(f"Tickets cleaned in {round(time.time() - start, 2)}s | rows={df.shape[0]}")
    return df