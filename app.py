import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from collections import Counter
from io import BytesIO
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Lottery Lowest Payout Optimizer",
    layout="wide"
)

st.title("🎯 Lottery Lowest Payout Optimizer (Pro)")

# =====================================================
# LOGIN / ADMIN MODE
# =====================================================
ADMIN_PASSWORD = "admin123"   # 🔐 change this

with st.sidebar:
    st.header("🔐 Admin Login")
    password = st.text_input("Password", type="password")

if password != ADMIN_PASSWORD:
    st.warning("Enter admin password to continue")
    st.stop()

# =====================================================
# CONFIG (ADMIN CONTROLS)
# =====================================================
st.sidebar.header("⚙️ Settings")

TIME_LIMIT = st.sidebar.slider("Time Limit (seconds)", 30, 600, 240)
MAX_RESULTS = st.sidebar.slider("Max Results", 5, 50, 20)
POP_SIZE = st.sidebar.slider("GA Population Size", 20, 100, 50)
GENERATIONS = st.sidebar.slider("GA Generations", 10, 200, 80)

NUMBERS = list(range(1, 26))
PAYOUT = np.array([0, 0, 0, 15, 400, 1850, 50000])

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader("📂 Upload Excel File (tickets in column A)", type=["xlsx"])

if not uploaded_file:
    st.stop()

df = pd.read_excel(uploaded_file)

# =====================================================
# LOAD TICKETS
# =====================================================
tickets = []
freq = Counter()

for v in df.iloc[:, 0]:
    try:
        nums = list(map(int, str(v).split(',')))
        if len(nums) == 6:
            mask = 0
            for n in nums:
                mask |= 1 << (n - 1)
                freq[n] += 1
            tickets.append(mask)
    except:
        pass

ticket_masks = np.array(tickets, dtype=np.uint32)
ticket_count = len(ticket_masks)

st.success(f"🎟️ Tickets Loaded: {ticket_count}")

# =====================================================
# STATS & CHARTS
# =====================================================
st.subheader("📊 Ticket Number Frequency")
freq_df = pd.DataFrame(freq.items(), columns=["Number", "Frequency"]).sort_values("Number")
st.bar_chart(freq_df.set_index("Number"))

# =====================================================
# ANTI-HOT NUMBER PRIORITY
# =====================================================
cold = sorted(NUMBERS, key=lambda x: freq[x])
cold_10 = cold[:10]
mid_10 = cold[10:20]

bit_map = {n: 1 << (n - 1) for n in NUMBERS}

# =====================================================
# NUMPY BITCOUNT
# =====================================================
def bitcount_arr(x):
    return np.unpackbits(x.view(np.uint8), axis=1).sum(axis=1)

# =====================================================
# FAST NUMPY EVALUATION
# =====================================================
def evaluate(mask, cutoff):
    matches = bitcount_arr(ticket_masks & mask)

    if np.any(matches >= 5):
        return None

    if np.sum(matches == 4) != 10:
        return None

    payout = PAYOUT[matches].sum()
    return payout if payout <= cutoff else None

# =====================================================
# GENETIC ALGORITHM
# =====================================================
def random_combo():
    combo = random.sample(cold_10, 4) + random.sample(mid_10, 2)
    mask = 0
    for n in combo:
        mask |= bit_map[n]
    return mask

def mutate(mask):
    nums = [i + 1 for i in range(25) if mask & (1 << i)]
    idx = random.randint(0, 5)
    nums[idx] = random.choice(cold_10)
    nums = list(set(nums))
    if len(nums) < 6:
        nums += random.sample(mid_10, 6 - len(nums))
    mask = 0
    for n in nums[:6]:
        mask |= bit_map[n]
    return mask

def crossover(a, b):
    na = [i + 1 for i in range(25) if a & (1 << i)]
    nb = [i + 1 for i in range(25) if b & (1 << i)]
    nums = list(set(na[:3] + nb[3:]))
    if len(nums) < 6:
        nums += random.sample(mid_10, 6 - len(nums))
    mask = 0
    for n in nums[:6]:
        mask |= bit_map[n]
    return mask

# =====================================================
# RUN OPTIMIZER
# =====================================================
if st.button("🚀 Run Optimizer"):
    start = time.time()
    best_results = []
    population = [random_combo() for _ in range(POP_SIZE)]
    progress = st.progress(0)
    status = st.empty()

    def worst_score():
        return best_results[-1][0] if len(best_results) == MAX_RESULTS else float("inf")

    for gen in range(GENERATIONS):
        if time.time() - start > TIME_LIMIT:
            break

        new_population = []

        for mask in population:
            score = evaluate(mask, worst_score())
            if score is not None:
                best_results.append((score, mask))
                best_results.sort()
                best_results = best_results[:MAX_RESULTS]

        elites = [m for _, m in best_results[:10]]

        while len(new_population) < POP_SIZE:
            a, b = random.sample(elites, 2)
            child = crossover(a, b)
            if random.random() < 0.3:
                child = mutate(child)
            new_population.append(child)

        population = new_population

        status.write(f"🧬 Generation {gen+1} | Best Payout: {best_results[0][0] if best_results else '—'}")
        progress.progress((gen + 1) / GENERATIONS)

    # =====================================================
    # RESULTS
    # =====================================================
    st.subheader("🏆 Top Lowest Payout Results")

    results = []
    payouts = []

    for score, mask in best_results:
        nums = [i + 1 for i in range(25) if mask & (1 << i)]
        results.append({"Numbers": nums, "Payout": score})
        payouts.append(score)

    results_df = pd.DataFrame(results)
    st.dataframe(results_df)

    # =====================================================
    # PAYOUT CHART
    # =====================================================
    st.subheader("📉 Payout Distribution")
    fig, ax = plt.subplots()
    ax.plot(sorted(payouts))
    ax.set_ylabel("Payout")
    ax.set_xlabel("Rank")
    st.pyplot(fig)

    # =====================================================
    # EXPORT TO EXCEL
    # =====================================================
    def export_excel(df):
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer

    st.download_button(
        "⬇ Download Results (Excel)",
        export_excel(results_df),
        "lowest_payout_results.xlsx"
    )

    st.success("✅ Optimization Complete")
