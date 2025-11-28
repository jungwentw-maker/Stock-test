"""
indicators.py

技術指標相關函式集中放在這個模組。
目前包含：
- 移動平均線（MA）
- KD 指標
"""

from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd


Number = Union[int, float]


def add_moving_averages(
    df: pd.DataFrame,
    windows: Sequence[int] = (5, 20, 60, 120),
    price_col: str = "Close",
    prefix: str = "MA",
) -> pd.DataFrame:
    """
    在 df 上計算多組移動平均線，直接新增欄位，並回傳同一個 df。

    參數
    ------
    df : pd.DataFrame
        需要包含 price_col 欄位（預設 "Close"）。
    windows : 序列[int]
        要計算的期數，例如 (5, 20, 60, 120)。
    price_col : str
        用來計算 MA 的價位欄位名稱。
    prefix : str
        新欄位的前綴，預設 "MA"，例如 MA5, MA20。

    回傳
    ------
    df : pd.DataFrame
        原 df，增加 MA 欄位後的結果（同一物件）。
    """
    if price_col not in df.columns:
        raise KeyError(f"資料中找不到價位欄位：{price_col!r}")

    for w in windows:
        col_name = f"{prefix}{w}"
        df[col_name] = df[price_col].rolling(w).mean()

    return df


def add_kd(
    df: pd.DataFrame,
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    n: int = 9,
    k_init: Number = 50,
    d_init: Number = 50,
    k_col: str = "K",
    d_col: str = "D",
    rsv_col: str = "RSV",
) -> pd.DataFrame:
    """
    在 df 上計算 KD 指標（隨機指標），欄位名稱預設為 K / D / RSV。

    演算法：
    - RSV = (Close - low_n) / (high_n - low_n) * 100
      若 high_n == low_n 則 RSV = 50
    - K、D 為指數平滑：
        K_t = 2/3 * K_{t-1} + 1/3 * RSV_t
        D_t = 2/3 * D_{t-1} + 1/3 * K_t
      初始值 K_0 = D_0 = 50（可調整為 k_init, d_init）。

    參數
    ------
    df : pd.DataFrame
        需要包含 high_col / low_col / close_col 欄位。
    high_col, low_col, close_col : str
        最高價 / 最低價 / 收盤價欄位名稱。
    n : int
        回看天數，預設 9。
    k_init, d_init : Number
        K / D 的初始值。
    k_col, d_col, rsv_col : str
        在 df 中要使用的欄位名稱。

    回傳
    ------
    df : pd.DataFrame
        原 df，增加 RSV / K / D 欄位後的結果（同一物件）。
    """
    for col in (high_col, low_col, close_col):
        if col not in df.columns:
            raise KeyError(f"資料中找不到欄位：{col!r}")

    # n 日最高 / 最低
    low_n = df[low_col].rolling(n).min()
    high_n = df[high_col].rolling(n).max()

    # RSV
    df[rsv_col] = np.where(
        (high_n - low_n) == 0,
        50.0,
        (df[close_col] - low_n) / (high_n - low_n) * 100.0,
    )

    # 初始化 K, D 欄位
    df[k_col] = np.nan
    df[d_col] = np.nan

    first_valid = df[rsv_col].first_valid_index()
    if first_valid is None:
        # 全部都是 NaN，直接回傳
        return df

    # 設定初始值
    df.loc[first_valid, k_col] = float(k_init)
    df.loc[first_valid, d_col] = float(d_init)

    # 依照既有程式邏輯的迴圈計算 K / D
    for i in range(first_valid + 1, len(df)):
        prev_k = df.iloc[i - 1][k_col]
        prev_d = df.iloc[i - 1][d_col]
        rsv = df.iloc[i][rsv_col]

        k_today = prev_k * 2.0 / 3.0 + rsv * 1.0 / 3.0
        d_today = prev_d * 2.0 / 3.0 + k_today * 1.0 / 3.0

        df.iat[i, df.columns.get_loc(k_col)] = k_today
        df.iat[i, df.columns.get_loc(d_col)] = d_today

    return df


# 為了相容舊有程式，如果你之前是用 compute_kd(df) 的話，這裡給一個別名。
def compute_kd(df: pd.DataFrame) -> pd.DataFrame:
    """
    相容舊版名稱的 KD 計算函式：
    - 使用 High / Low / Close
    - n = 9
    - 初始 K, D = 50
    - 欄位名稱為 RSV / K / D
    """
    return add_kd(df)
