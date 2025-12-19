# =============================
# STREAMLIT CLOUD SAFE VERSION
# =============================

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from collections import Counter
from io import BytesIO
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Lottery Lowest Payout Optimizer",
    layout="wide"
)

st.title("🎯 Lottery Lowest Payout Optimizer (Pro)")

# =============================
# LOGIN / ADMIN MODE
# =============================
ADMIN_PASSWORD = "admin123"   # 🔐 change this

with st.sidebar:
    st.header("🔐 Admin Login")
    password = st.text_input("Password", type="password")

if password != ADMIN_PASSWORD:
    st.warning("Enter admin password to continue")
    st.stop()

# =============================
# ADMIN SETTINGS
# =============================
st.sidebar.header("⚙️ Settings")

TIME_LIMIT = st.sidebar.slider("Time Limit (seconds)", 30, 600, 240)
MAX_RESULTS = st.sidebar.slider("Max Results", 5, 50, 20)
POP_SIZE = st.sidebar.slider("GA Population Size", 20, 100, 40)
GENERATIONS = st.sidebar.slider("GA Generations", 10, 200, 60)

NUMBERS = list(range(1, 26))
PAYOUT = [0, 0, 0, 15, 400, 1850, 50000]

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader("📂 Upload Excel File (tickets in column A)", type=["xlsx"])

if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# =============================
# LOAD TICKETS
# =============================
tickets = []
freq = Counter()

for v in df.iloc[:, 0]:
    try:
        nums = list(map(int, str(v).split(',')))
        if len(nums) == 6 and len(set(nums)) == 6:
            mask = 0
            for n in nums:
                mask |= 1 << (n - 1)
                freq[n] += 1
            tickets.append(mask)
    except:
        continue

if not tickets:
    st.error("No valid tickets found in file.")
    st.stop()

ticket_masks = np.array(tickets, dtype=np.uint32)

st.success(f"🎟️ Tickets Loaded: {len(ticket_masks)}")

# =============================
# STATS
# =============================
st.subheader("📊 Number Frequency")
freq_df = pd.DataFrame(
    [{"Number": i, "Frequency": freq[i]} for i in NUMBERS]
)
st.bar_chart(freq_df.set_index("Number"))

# =============================
# NUMBER PRIORITY
# =============================
cold = sorted(NUMBERS, key=lambda x: freq[x])
cold_10 = cold[:10]
mid_10 = cold[10:20]

bit_map = {n: 1 << (n - 1) for n in NUMBERS}

# =============================
# SAFE BITCOUNT (CLOUD)
# =============================
def bitcount_arr(arr):
    return np.array([int(x).bit_count() for x in arr], dtype=np.uint8)

# =============================
# SAFE EVALUATION
# =============================
def evaluate(mask, cutoff):
    matches = bitcount_arr(ticket_masks & mask)

    if np.any(matches >= 5):
        return None

    if np.sum(matches == 4) != 10:
        return None

    total = sum(PAYOUT[m] for m in matches)
    return total if total <= cutoff else None

# =============================
# GENETIC ALGORITHM HELPERS
# =============================
def random_combo():
    nums = random.sample(cold_10, 4) + random.sample(mid_10, 2)
    mask = 0
    for n in nums:
        mask |= bit_map[n]
    return mask

def mask_to_nums(mask):
    return [i + 1 for i in range(25) if mask & (1 << i)]

def mutate(mask):
    nums = mask_to_nums(mask)
    nums[random.randint(0, 5)] = random.choice(cold_10)
    nums = list(dict.fromkeys(nums))
    while len(nums) < 6:
        nums.append(random.choice(mid_10))
    mask = 0
    for n in nums[:6]:
        mask |= bit_map[n]
    return mask

def crossover(a, b):
    na = mask_to_nums(a)
    nb = mask_to_nums(b)
    nums = list(dict.fromkeys(na[:3] + nb[3:]))
    while len(nums) < 6:
        nums.append(random.choice(mid_10))
    mask = 0
    for n in nums[:6]:
        mask |= bit_map[n]
    return mask

# =============================
# RUN OPTIMIZER
# =============================
if st.button("🚀 Run Optimizer"):
    start = time.time()
    best_results = []
    population = [random_combo() for _ in range(POP_SIZE)]
    progress = st.progress(0.0)
    status = st.empty()

    def worst_score():
        return best_results[-1][0] if len(best_results) == MAX_RESULTS else float("inf")

    for gen in range(GENERATIONS):
        if time.time() - start > TIME_LIMIT:
            break

        for mask in population:
            score = evaluate(mask, worst_score())
            if score is not None:
                best_results.append((score, mask))
                best_results = sorted(best_results)[:MAX_RESULTS]

        elites = [m for _, m in best_results[:10]] or population
        new_population = elites.copy()

        while len(new_population) < POP_SIZE:
            a, b = random.sample(elites, 2)
            child = crossover(a, b)
            if random.random() < 0.3:
                child = mutate(child)
            new_population.append(child)

        population = new_population

        progress.progress((gen + 1) / GENERATIONS)
        status.write(
            f"🧬 Generation {gen+1} | "
            f"Best Payout: {best_results[0][0] if best_results else '—'}"
        )

    # =============================
    # RESULTS
    # =============================
    st.subheader("🏆 Lowest Payout Results")

    results = []
    payouts = []

    for score, mask in best_results:
        nums = mask_to_nums(mask)
        results.append({"Numbers": nums, "Payout": score})
        payouts.append(score)

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # =============================
    # PAYOUT CHART
    # =============================
    st.subheader("📉 Payout Distribution")
    fig, ax = plt.subplots()
    ax.plot(sorted(payouts))
    ax.set_xlabel("Rank")
    ax.set_ylabel("Payout")
    st.pyplot(fig)

    # =============================
    # EXPORT
    # =============================
    def export_excel(df):
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer

    st.download_button(
        "⬇ Download Results (Excel)",
        export_excel(df_results),
        "lowest_payout_results.xlsx"
    )

    st.success("✅ Optimization Complete")
