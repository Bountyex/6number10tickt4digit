import streamlit as st
import pandas as pd
import random
import time
from collections import Counter

# =============================
# CONFIG
# =============================
COMBO_SIZE = 6
NUMBERS = list(range(1, 26))
MAX_ITERATIONS = 800000
TIME_LIMIT = 240    # seconds
MAX_RESULTS = 20

# =============================
# PAYOUT RULES
# =============================
PAYOUT = [0, 0, 0, 15, 400, 1850, 50000]

st.title("🎯 LOWEST TOTAL PAYOUT OPTIMIZER (6 numbers, 1–25)")

uploaded = st.file_uploader("Upload Excel with tickets", type=["xlsx"])

if uploaded:

    df = pd.read_excel(uploaded)
    st.success("File loaded successfully")

    tickets_bits = []
    freq = Counter()

    for val in df.iloc[:, 0]:
        if isinstance(val, str):
            try:
                nums = [int(x.strip()) for x in val.split(",")]
                if len(set(nums)) == 6:
                    mask = 0
                    for n in nums:
                        mask |= 1 << (n - 1)
                        freq[n] += 1
                    tickets_bits.append(mask)
            except:
                continue

    st.write(f"🎟️ Tickets Loaded:", len(tickets_bits))

    numbers_weighted = sorted(NUMBERS, key=lambda x: freq[x])

    best_results = []
    seen = set()

    def worst_payout():
        return best_results[-1][0] if best_results else float("inf")

    def evaluate(mask):
        total = 0
        count_4 = 0

        for t in tickets_bits:
            m = (mask & t).bit_count()

            if m == 4:
                count_4 += 1
                if count_4 > 10:
                    return None

            total += PAYOUT[m]

            if total > worst_payout():
                return None

        if count_4 != 10:
            return None

        return total

    start = time.time()

    progress_bar = st.progress(0)
    status = st.empty()

    for i in range(MAX_ITERATIONS):

        # TIME CHECK
        if time.time() - start > TIME_LIMIT:
            break

        combo = random.sample(numbers_weighted[:15], 4) + random.sample(numbers_weighted[15:], 2)

        mask = 0
        for n in combo:
            mask |= 1 << (n - 1)

        if mask in seen:
            continue

        seen.add(mask)

        score = evaluate(mask)
        if score is None:
            continue

        best_results.append((score, mask))
        best_results = sorted(best_results)[:MAX_RESULTS]

        nums = [i + 1 for i in range(25) if mask & (1 << i)]

        status.write(f"🔥 FOUND → {nums} | ₹{score}")

        progress_bar.progress(i / MAX_ITERATIONS)

    st.subheader("🏆 TOP 20 LOWEST TOTAL PAYOUT RESULTS")

    for i, (score, mask) in enumerate(best_results, 1):
        nums = [i + 1 for i in range(25) if mask & (1 << i)]
        st.write(f"{i:02d}. {nums} → ₹{score}")

    st.success("✔ DONE")
