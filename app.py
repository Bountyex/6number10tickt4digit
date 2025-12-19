import streamlit as st
import pandas as pd
import random
import time
from collections import Counter
from io import BytesIO
import matplotlib.pyplot as plt

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Lottery Optimizer", layout="wide")
st.title("🎯 Lottery Lowest Payout Optimizer")

# -------------------------------
# LOGIN
# -------------------------------
ADMIN_PASSWORD = "admin123"

pwd = st.sidebar.text_input("🔐 Admin Password", type="password")
if pwd != ADMIN_PASSWORD:
    st.warning("Enter admin password")
    st.stop()

# -------------------------------
# SETTINGS
# -------------------------------
TIME_LIMIT = st.sidebar.slider("Time Limit (sec)", 30, 300, 120)
MAX_RESULTS = st.sidebar.slider("Max Results", 5, 30, 15)

PAYOUT = [0, 0, 0, 15, 400, 1850, 50000]
NUMBERS = list(range(1, 26))

# -------------------------------
# FILE UPLOAD
# -------------------------------
file = st.file_uploader("📂 Upload Excel file (Column A: 6 numbers comma-separated)", type=["xlsx"])
if not file:
    st.stop()

df = pd.read_excel(file)

# -------------------------------
# LOAD TICKETS
# -------------------------------
tickets = []
freq = Counter()

for v in df.iloc[:, 0]:
    try:
        nums = list(map(int, str(v).split(",")))
        if len(nums) == 6 and len(set(nums)) == 6:
            tickets.append(set(nums))
            for n in nums:
                freq[n] += 1
    except:
        pass

if not tickets:
    st.error("No valid tickets found")
    st.stop()

st.success(f"🎟️ Tickets Loaded: {len(tickets)}")

# -------------------------------
# STATS
# -------------------------------
st.subheader("📊 Number Frequency")
freq_df = pd.DataFrame(
    {"Number": NUMBERS, "Frequency": [freq[n] for n in NUMBERS]}
)
st.bar_chart(freq_df.set_index("Number"))

# -------------------------------
# EVALUATION
# -------------------------------
def evaluate(combo, cutoff):
    total = 0
    count4 = 0

    for t in tickets:
        m = len(combo & t)

        if m >= 5:
            return None

        if m == 4:
            count4 += 1
            if count4 > 10:
                return None

        total += PAYOUT[m]
        if total > cutoff:
            return None

    if count4 != 10:
        return None

    return total

# -------------------------------
# RUN OPTIMIZER
# -------------------------------
if st.button("🚀 Run Optimizer"):
    start = time.time()
    best = []

    cold = sorted(NUMBERS, key=lambda x: freq[x])

    progress = st.progress(0.0)

    while time.time() - start < TIME_LIMIT:
        combo = set(random.sample(cold[:15], 6))
        cutoff = best[-1][0] if len(best) == MAX_RESULTS else float("inf")

        score = evaluate(combo, cutoff)
        if score is None:
            continue

        best.append((score, combo))
        best = sorted(best)[:MAX_RESULTS]

        progress.progress(min((time.time() - start) / TIME_LIMIT, 1.0))

    # -------------------------------
    # RESULTS
    # -------------------------------
    st.subheader("🏆 Lowest Payout Results")

    results = []
    payouts = []

    for score, combo in best:
        results.append({"Numbers": sorted(combo), "Payout": score})
        payouts.append(score)

    res_df = pd.DataFrame(results)
    st.dataframe(res_df)

    # -------------------------------
    # CHART
    # -------------------------------
    fig, ax = plt.subplots()
    ax.plot(sorted(payouts))
    ax.set_ylabel("Payout")
    ax.set_xlabel("Rank")
    st.pyplot(fig)

    # -------------------------------
    # EXPORT
    # -------------------------------
    buffer = BytesIO()
    res_df.to_excel(buffer, index=False)
    buffer.seek(0)

    st.download_button(
        "⬇ Download Excel",
        buffer,
        "lowest_payout_results.xlsx"
    )

    st.success("✅ Done")
