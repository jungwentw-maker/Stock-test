import numpy as np
import pandas as pd


def compute_raff_channels(
    df: pd.DataFrame,
    min_length: int = 20,
    max_length: int = 80,
    min_r2: float = 0.6,
    max_half_width_ratio: float = 0.15,
    min_trend_ratio: float = 0.03,
):
    """
    計算不重疊的 Raff Regression Channels：
    - 逐一嘗試每個起點，但通道彼此不可重疊（先收集所有候選，再以 R^2 高→低的順序挑選）
    - 視窗長度限制在 min_length～max_length 之間，線性回歸的 R^2 需達 min_r2
    - 以 R^2 為主要評分（越高越好），同分再取通道半寬較小者，長度較長者優先
    - 半寬不得超過通道中點的 max_half_width_ratio（預設 ±15%）
    - 通道上、下緣各至少一個資料點觸碰才成立，避免通道偏離 K 線
    - 窗口首尾的價差占平均價的比例需大於等於 min_trend_ratio 且至少與半寬同級，避免側向盤整也被畫通道
    - 回傳 list[dict]：每個通道包含 {"x", "upper", "lower", "center", "window", "start", "r2"}
    若資料不足或無法計算則回傳空 list。
    """

    if len(df) < min_length or "Close" not in df.columns or "DateStr" not in df.columns:
        return []

    candidates = []

    for start in range(0, len(df) - min_length + 1):
        max_window = min(max_length, len(df) - start)

        for window in range(min_length, max_window + 1):
            closes = df["Close"].iloc[start:start + window]
            dates = df["DateStr"].iloc[start:start + window]

            if closes.isna().any():
                continue

            x = np.arange(window)
            slope, intercept = np.polyfit(x, closes, 1)
            fitted = intercept + slope * x

            # R^2 計算，確保有變異量再評估
            ss_res = np.sum((closes.to_numpy() - fitted) ** 2)
            ss_tot = np.sum((closes.to_numpy() - np.mean(closes)) ** 2)
            if ss_tot == 0:
                continue

            r2 = 1 - ss_res / ss_tot
            if r2 < min_r2:
                continue

            residuals = closes.to_numpy() - fitted
            max_abs = np.max(np.abs(residuals))

            mean_price = np.mean(closes)
            normalized_width = max_abs / mean_price if mean_price else np.inf

            if normalized_width > max_half_width_ratio:
                continue

            trend_ratio = abs(closes.iloc[-1] - closes.iloc[0]) / mean_price if mean_price else 0
            if trend_ratio < min_trend_ratio or trend_ratio < normalized_width:
                continue

            touch_tolerance = max(1e-6, max_abs * 1e-3)
            upper_touches = np.sum(
                np.isclose(residuals, max_abs, rtol=1e-3, atol=touch_tolerance)
            )
            lower_touches = np.sum(
                np.isclose(residuals, -max_abs, rtol=1e-3, atol=touch_tolerance)
            )

            if upper_touches < 1 or lower_touches < 1:
                continue

            candidates.append({
                "x": dates.tolist(),
                "center": fitted.tolist(),
                "upper": (fitted + max_abs).tolist(),
                "lower": (fitted - max_abs).tolist(),
                "window": window,
                "start": start,
                "end": start + window - 1,
                "r2": r2,
                "score_width": normalized_width,
            })

    # 依 R^2 高→低、半寬窄→寬、視窗長→短 排序，從高分挑選不重疊區段
    candidates.sort(key=lambda c: (-c["r2"], c["score_width"], -c["window"]))
    selected = []

    for cand in candidates:
        if any(not (cand["end"] < sel["start"] or cand["start"] > sel["end"]) for sel in selected):
            continue
        selected.append(cand)

    selected.sort(key=lambda c: c["start"])
    for cand in selected:
        cand.pop("score_width", None)
        cand.pop("end", None)

    return selected