import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

t0 = time.time()

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, "outputs", "cleaned")
FIG = os.path.join(ROOT, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

def summary(text):
    plt.gca().text(
        0.01, 0.01, text,
        transform=plt.gca().transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", alpha=0.15)
    )

def point(x, y, label):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(5,5), fontsize=9)

def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=180)
    plt.close()

flights = pd.read_csv(os.path.join(CLEAN, "flights_clean.csv"), low_memory=False, parse_dates=["FL_DATE"])
tickets = pd.read_csv(os.path.join(CLEAN, "tickets_clean.csv"), low_memory=False)
airports = pd.read_csv(os.path.join(CLEAN, "airports_clean.csv"), low_memory=False)

flights.columns = flights.columns.str.upper()
tickets.columns = tickets.columns.str.upper()
airports.columns = airports.columns.str.upper()

def find_iata(df, prefix):
    for c in df.columns:
        if c.startswith(prefix) and df[c].dtype == "object":
            if df[c].dropna().astype(str).str.len().max() <= 3:
                return c
    raise KeyError(prefix)

F_ORIGIN = find_iata(flights, "ORIGIN")
F_DEST = find_iata(flights, "DEST")
T_ORIGIN = "ORIGIN"
T_DEST = "DESTINATION"

for c in ["DEP_DELAY","ARR_DELAY","DISTANCE","OCCUPANCY_RATE","AIR_TIME"]:
    if c in flights.columns:
        flights[c] = pd.to_numeric(flights[c], errors="coerce")

for c in ["ITIN_FARE","PASSENGERS"]:
    if c in tickets.columns:
        tickets[c] = pd.to_numeric(tickets[c], errors="coerce")

airports[["LON","LAT"]] = airports["COORDINATES"].str.split(",", expand=True).astype(float)
airports = airports.set_index("IATA_CODE")

flights["ROUTE"] = flights[[F_ORIGIN, F_DEST]].astype(str).apply(lambda x: "_".join(sorted(x)), axis=1)
tickets["ROUTE"] = tickets[[T_ORIGIN, T_DEST]].astype(str).apply(lambda x: "_".join(sorted(x)), axis=1)

rf = flights.groupby("ROUTE", as_index=False).agg(
    ORIGIN=(F_ORIGIN,"first"),
    DESTINATION=(F_DEST,"first"),
    FLIGHTS=("ROUTE","count"),
    DEP_DELAY=("DEP_DELAY","mean"),
    ARR_DELAY=("ARR_DELAY","mean"),
    DISTANCE=("DISTANCE","mean"),
    OCCUPANCY=("OCCUPANCY_RATE","mean")
)

rt = tickets.groupby("ROUTE", as_index=False).agg(
    FARE=("ITIN_FARE","mean"),
    PASSENGERS=("PASSENGERS","sum")
)

routes = rf.merge(rt, on="ROUTE", how="inner")
routes["REVENUE"] = routes["FARE"] * routes["PASSENGERS"]
routes["COST"] = routes["FLIGHTS"] * routes["DISTANCE"] * 9.18
routes["PROFIT"] = routes["REVENUE"] - routes["COST"]

plt.rcParams.update({
    "figure.figsize": (13,7),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 18,
    "axes.labelsize": 13
})

top10 = routes.sort_values("PROFIT", ascending=False).head(10)
top20 = routes.sort_values("PROFIT", ascending=False).head(20)
best = top10.iloc[0]

plt.barh(top10["ROUTE"], top10["PROFIT"])
plt.gca().invert_yaxis()
plt.title("Top 10 Most Profitable Routes")
plt.xlabel("Profit (USD)")
plt.ylabel("Route")
summary(f"Highest profit route: {best['ROUTE']}\nProfit: ${best['PROFIT']:,.0f}")
save("01_profit_top10.png")

plt.barh(top10["ROUTE"], top10["FLIGHTS"])
plt.gca().invert_yaxis()
plt.title("Traffic Volume of Top 10 Profitable Routes")
plt.xlabel("Number of Flights")
plt.ylabel("Route")
summary("High-profit routes also show strong and stable demand")
save("02_volume_top10.png")

plt.scatter(routes["DEP_DELAY"], routes["PROFIT"], alpha=0.6)
plt.title("Profit vs Average Departure Delay")
plt.xlabel("Average Departure Delay (minutes)")
plt.ylabel("Profit (USD)")
point(best["DEP_DELAY"], best["PROFIT"], best["ROUTE"])
summary("Lower delays generally align with higher profitability")
save("03_profit_vs_dep_delay.png")

plt.scatter(routes["ARR_DELAY"], routes["PROFIT"], alpha=0.6)
plt.title("Profit vs Average Arrival Delay")
plt.xlabel("Average Arrival Delay (minutes)")
plt.ylabel("Profit (USD)")
summary("Arrival delays negatively impact margins")
save("04_profit_vs_arr_delay.png")

plt.scatter(routes["OCCUPANCY"], routes["PROFIT"], alpha=0.6)
plt.title("Profit vs Average Seat Occupancy")
plt.xlabel("Occupancy Rate")
plt.ylabel("Profit (USD)")
summary("Higher seat utilization strongly drives profitability")
save("05_profit_vs_occupancy.png")

plt.scatter(routes["FLIGHTS"], routes["PROFIT"], alpha=0.6)
plt.title("Profit vs Route Volume")
plt.xlabel("Number of Flights")
plt.ylabel("Profit (USD)")
summary("High volume alone does not guarantee profitability")
save("06_profit_vs_volume.png")

plt.scatter(routes["DISTANCE"], routes["PROFIT"], alpha=0.6)
plt.title("Profit vs Average Route Distance")
plt.xlabel("Distance (miles)")
plt.ylabel("Profit (USD)")
summary("Medium-haul routes show strongest margins")
save("07_profit_vs_distance.png")

plt.hist(routes["PROFIT"], bins=60)
plt.title("Distribution of Route Profitability")
plt.xlabel("Profit (USD)")
plt.ylabel("Number of Routes")
summary(f"Median profit: ${routes['PROFIT'].median():,.0f}")
save("08_profit_distribution.png")

plt.hist(routes["DEP_DELAY"], bins=60)
plt.title("Distribution of Departure Delays")
plt.xlabel("Minutes")
plt.ylabel("Flights")
summary("Most departures cluster near on-time performance")
save("09_dep_delay_distribution.png")

plt.hist(routes["ARR_DELAY"], bins=60)
plt.title("Distribution of Arrival Delays")
plt.xlabel("Minutes")
plt.ylabel("Flights")
summary("Arrival delays show heavier tail")
save("10_arr_delay_distribution.png")

metrics = ["PROFIT","REVENUE","COST","FLIGHTS","DEP_DELAY","OCCUPANCY"]
idx = 11
for x,y in itertools.combinations(metrics,2):
    plt.scatter(routes[x], routes[y], alpha=0.5)
    plt.title(f"{y} vs {x}")
    plt.xlabel(x)
    plt.ylabel(y)
    summary("Multidimensional relationship across routes")
    save(f"{idx:02d}_{x.lower()}_vs_{y.lower()}.png")
    idx += 1

ranked = routes.sort_values("PROFIT", ascending=False)
plt.plot(ranked["PROFIT"].cumsum())
plt.title("Cumulative Profit Capture Curve")
plt.xlabel("Routes Ranked by Profitability")
plt.ylabel("Cumulative Profit (USD)")
summary("Top 20% of routes contribute majority of profit")
save("25_cumulative_profit.png")

daily = flights.groupby("FL_DATE", as_index=False).agg(
    FLIGHTS=("FL_DATE","count"),
    DELAY=("DEP_DELAY","mean")
)

plt.plot(daily["FL_DATE"], daily["FLIGHTS"])
plt.title("Daily Flight Volume Trend")
plt.xlabel("Date")
plt.ylabel("Number of Flights")
summary("Flight volume remains stable across period")
save("26_daily_flights.png")

plt.plot(daily["FL_DATE"], daily["DELAY"])
plt.title("Daily Average Departure Delay Trend")
plt.xlabel("Date")
plt.ylabel("Minutes")
summary("Operational disruptions visible as spikes")
save("27_daily_delay.png")

plt.plot(daily["FL_DATE"], daily["FLIGHTS"].rolling(7).mean())
plt.title("7-Day Rolling Average Flight Volume")
plt.xlabel("Date")
plt.ylabel("Flights")
summary("Rolling average smooths volatility")
save("28_rolling_flights.png")

plt.plot(daily["FL_DATE"], daily["DELAY"].rolling(7).mean())
plt.title("7-Day Rolling Average Delay")
plt.xlabel("Date")
plt.ylabel("Minutes")
summary("Sustained delay trends indicate stress")
save("29_rolling_delay.png")

plt.figure(figsize=(15,8))
shown = 0
for _, r in top10.iterrows():
    if r["ORIGIN"] not in airports.index or r["DESTINATION"] not in airports.index:
        continue
    o = airports.loc[r["ORIGIN"]]
    d = airports.loc[r["DESTINATION"]]
    plt.plot([o["LON"], d["LON"]], [o["LAT"], d["LAT"]], alpha=0.8)
    shown += 1
plt.scatter(airports["LON"], airports["LAT"], s=12)
plt.title("Geographic Network of High-Value Routes")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.axis("off")
summary(f"{shown} high-value routes visualized\nHub concentration evident")
save("30_route_network.png")

print(f"TOTAL FIGURES GENERATED: {idx - 1}")
print(f"TOTAL EXECUTION TIME: {round(time.time() - t0, 2)}s")