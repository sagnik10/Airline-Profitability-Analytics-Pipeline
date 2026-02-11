import time

def clean_airports(df):
    start = time.time()
    df = df[df["ISO_COUNTRY"] == "US"]
    df = df[df["TYPE"].isin(["medium_airport", "large_airport"])]
    df = df[df["IATA_CODE"].notna()]
    df.columns = df.columns.str.lower()
    print(f"Airports cleaned in {round(time.time() - start, 2)}s | rows={df.shape[0]}")
    return df