import os
import time
from load_data import load_csv_chunked
from clean_airports import clean_airports
from clean_flights import clean_flights
from clean_tickets import clean_tickets

start_total = time.time()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "cleaned")

os.makedirs(OUTPUT_DIR, exist_ok=True)

airports = load_csv_chunked(
    os.path.join(DATA_DIR, "Airport_Codes.csv"),
    chunksize=200_000
)

flights = load_csv_chunked(
    os.path.join(DATA_DIR, "Flights.csv"),
    chunksize=500_000
)

tickets = load_csv_chunked(
    os.path.join(DATA_DIR, "Tickets.csv"),
    chunksize=300_000
)

airports_clean = clean_airports(airports)
flights_clean = clean_flights(flights)
tickets_clean = clean_tickets(tickets)

airports_clean.to_csv(
    os.path.join(OUTPUT_DIR, "airports_clean.csv"),
    index=False
)

flights_clean.to_csv(
    os.path.join(OUTPUT_DIR, "flights_clean.csv"),
    index=False
)

tickets_clean.to_csv(
    os.path.join(OUTPUT_DIR, "tickets_clean.csv"),
    index=False
)

print(f"TOTAL PIPELINE TIME: {round(time.time() - start_total, 2)}s")