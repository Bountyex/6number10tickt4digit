import streamlit as st
import pandas as pd
import random
import time
from collections import Counter

st.set_page_config(page_title="Lowest Payout Finder", layout="wide")
st.title("🎯 Lottery Lowest Payout Optimizer")

# =============================
# CONFIG
# =============================
COMBO_SIZE = 6
NUMBERS = list(range(1, 26))
TIME_LIMIT = 240
MAX_RESULTS = 20

PAYOUT = [0, 0, 0, 15, 400, 1850, 50000]

# =============================
# FILE UPLOAD
# =============================
uploaded_file = st.file_uploader("📂 Upload Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    tickets = []
    freq = Counter()

    for v in df.iloc[:, 0]:
        try:
            nums = list(map(int, v.split(',')))
            if len(nums) == 6:
                mask = 0
                for n in nums:
                    mask |= 1 << (n - 1)
                    freq[n] += 1
                tickets.append(mask)
        except:
            pass

    st.success(f"🎟️ Tickets Loaded: {len(tickets)}")

    # =============================
    # NUMBER PRIORITY
    # =============================
    cold = sorted(NUMBERS, key=lambda x: freq[x])
    cold_10 = cold[:10]
    mid_10 = cold[10:20]

    bit_map = {n: 1 << (n - 1) for n in NUMBERS}

    # =============================
    # SEARCH BUTTON
    # =============================
    if st.button("🚀 Start Search"):
        best_results = []
        seen = set()
        start = time.time()
        progress = st.progress(0)
        status = st.empty()

        def worst_score():
            return best_results[-1][0] if len(best_results) == MAX_RESULTS else float("inf")

        def evaluate(mask, cutoff):
            total = 0
            count4 = 0

            for t in tickets:
                m = (mask & t).bit_count()

                if m >= 5:
                    return None

                if m == 4:
                    count4 += 1
                    if count4 > 10:
                        return None

                total += PAYOUT[m]
                if total > cutoff:
                    return None

            return total if count4 == 10 else None

        iterations = 0

        while time.time() - start < TIME_LIMIT:
            iterations += 1

            combo = random.sample(cold_10, 4) + random.sample(mid_10, 2)
            mask = 0
            for n in combo:
                mask |= bit_map[n]

            if mask in seen:
                continue
            seen.add(mask)

            score = evaluate(mask, worst_score())
            if score is None:
                continue

            best_results.append((score, mask))
            best_results.sort()
            best_results = best_results[:MAX_RESULTS]

            status.write(f"🔥 Found payout ₹{score}")

            progress.progress(min((time.time() - start) / TIME_LIMIT, 1.0))

        st.subheader("🏆 Top 20 Lowest Payout Results")

        results = []
        for score, mask in best_results:
            nums = [n + 1 for n in range(25) if mask & (1 << n)]
            results.append({"Numbers": nums, "Payout": score})

        st.dataframe(pd.DataFrame(results))
