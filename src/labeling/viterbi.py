from __future__ import annotations

import numpy as np


def viterbi_decode(emissions: np.ndarray, transition: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Max-score Viterbi decoding. emissions: [T, S], transition: [S, S]."""
    T, S = emissions.shape
    if transition.shape != (S, S):
        raise ValueError(f"transition shape {transition.shape} does not match emissions state size {S}")

    dp = np.full((T, S), -1e18, dtype=float)
    backptr = np.full((T, S), -1, dtype=int)
    dp[0] = emissions[0]

    for t in range(1, T):
        prev_scores = dp[t - 1]
        for curr in range(S):
            scores = prev_scores + transition[:, curr]
            best_prev = int(np.argmax(scores))
            dp[t, curr] = scores[best_prev] + emissions[t, curr]
            backptr[t, curr] = best_prev

    path = np.zeros(T, dtype=int)
    path[T - 1] = int(np.argmax(dp[T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]]
    return path, dp
