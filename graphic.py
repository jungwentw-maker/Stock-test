import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# === 路徑設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(BASE_DIR, "historical data")
STOCK_MAP_PATH = os.path.join(BASE_DIR, "content", "stock_id.csv")


def load_stock_map():
    """
    讀取 content/stock_id.csv
    預期格式：stock_id, name
    回傳 dict：{ '2330': '台積電', ... }
    """
    mapping = {}
    if not os.path.exists(STOCK_MAP_PATH):
        return mapping

    # 嘗試 utf-8-sig，失敗再用 utf-8
    try:
        df = pd.read_csv(STOCK_MAP_PATH, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(STOCK_MAP_PATH, encoding="utf-8")

    if df.shape[1] < 2:
        return mapping

    df = df.iloc[:, :2]
    df.columns = ["stock_id", "name"]

    for _, row in df.iterrows():
        sid = str(row["stock_id"]).strip()
        name = str(row["name"]).strip()
        if not sid:
            continue

        # 兩種 key 都放，避免 0050 / 50 的差異
        mapping[sid] = name
        mapping[sid.lstrip("0")] = name

    return mapping


def scan_csv_files():
    """
    掃描 historical data 底下所有 CSV
    檔名預期格式：{stock_id} {period}.csv 例如：2330 1y.csv
    回傳 list[dict]：每個 dict 包含 stock_id, period, path, filename
    """
    files = []
    if not os.path.isdir(HIST_DIR):
        return files

    for fname in os.listdir(HIST_DIR):
        if not fname.lower().endswith(".csv"):
            continue

        base = fname[:-4]  # 去掉 .csv
        parts = base.split()
        if len(parts) >= 2:
            stock_id = parts[0]
            period = " ".join(parts[1:])
        else:
            stock_id = parts[0]
            period = ""

        files.append({
            "stock_id": stock_id,
            "period": period,
            "filename": fname,
            "path": os.path.join(HIST_DIR, fname),
        })

    return sorted(files, key=lambda x: (x["stock_id"], x["period"]))


def compute_kd(df: pd.DataFrame) -> pd.DataFrame:
    """
    在 df 上新增 KD 指標欄位：K, D
    公式：9 日 RSV → K, D 指數平滑
    """
    low_9 = df["Low"].rolling(9).min()
    high_9 = df["High"].rolling(9).max()

    df["RSV"] = np.where(
        (high_9 - low_9) == 0,
        50,
        (df["Close"] - low_9) / (high_9 - low_9) * 100
    )

    df["K"] = np.nan
    df["D"] = np.nan

    first_valid = df["RSV"].first_valid_index()
    if first_valid is None:
        return df

    df.loc[first_valid, "K"] = 50
    df.loc[first_valid, "D"] = 50

    for i in range(first_valid + 1, len(df)):
        df.loc[i, "K"] = df.loc[i - 1, "K"] * 2 / 3 + df.loc[i, "RSV"] * 1 / 3
        df.loc[i, "D"] = df.loc[i - 1, "D"] * 2 / 3 + df.loc[i, "K"] * 1 / 3

    return df



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


def plot_stock(file_info, stock_name_map):
    """
    根據選到的 CSV 畫圖：
    - 上：K 線 + MA (y)
    - 中：成交量 (y2)
    - 下：KD (y3)
    X 軸改為類別軸（DateStr），沒有週末空格
    """
    csv_path = file_info["path"]
    stock_id = file_info["stock_id"]
    period = file_info["period"]

    # 讀取 csv
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        messagebox.showerror("欄位錯誤", f"{csv_path}\n缺少必要欄位：{required_cols - set(df.columns)}")
        return

    # 日期與排序
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")      # 先把字串轉成「有時區的 datetime」，全部統一成 UTC
    df["Date"] = df["Date"].dt.tz_localize(None)      # 再把時區拿掉，變成乾淨的 datetime64[ns]
    df.sort_values("Date", inplace=True) # 照日期排序

    # 類別軸用的字串日期（只要有資料的日期才會出現一次）
    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")

    # 移動平均線
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # KD
    df = compute_kd(df)

    # Raff Regression Channels（20～80 個交易日，列出圖形範圍內所有通道）
    raff_channels = compute_raff_channels(df, min_length=20, max_length=80)

    # 成交量顏色（漲紅跌綠）
    volume_colors = [
        "red" if c >= o else "green"
        for c, o in zip(df["Close"], df["Open"])
    ]

    # 股票名稱
    name = stock_name_map.get(stock_id) or stock_name_map.get(stock_id.lstrip("0")) or ""
    title_text = f"{stock_id} {name}（{period}）" if period else f"{stock_id} {name}"

    # === 建立 traces（X 軸改用 DateStr 類別） ===
    traces = []

    # ① 最上：K 線 + MA（y）
    traces.append(go.Candlestick(
        x=df["DateStr"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="red",
        decreasing_line_color="green",
        name="K線",
        xaxis="x",
        yaxis="y",
    ))

    traces.append(go.Scatter(
        x=df["DateStr"], y=df["MA5"],
        mode="lines", name="MA5",
        line=dict(color="orange"),
        xaxis="x", yaxis="y",
    ))
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["MA20"],
        mode="lines", name="MA20",
        line=dict(color="blue"),
        xaxis="x", yaxis="y",
    ))
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["MA60"],
        mode="lines", name="MA60",
        line=dict(color="purple"),
        xaxis="x", yaxis="y",
    ))
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["MA120"],
        mode="lines", name="MA120",
        line=dict(color="brown"),
        xaxis="x", yaxis="y",
    ))

    if raff_channels:
        legend_added = False
        for channel in raff_channels:
            showlegend = not legend_added
            name_suffix = f"（{channel['window']}日）"
            traces.append(go.Scatter(
                x=channel["x"], y=channel["upper"],
                mode="lines", name=f"Raff 上緣{name_suffix}",
                line=dict(color="yellow", width=2),
                xaxis="x", yaxis="y", showlegend=showlegend,
            ))
            traces.append(go.Scatter(
                x=channel["x"], y=channel["lower"],
                mode="lines", name=f"Raff 下緣{name_suffix}",
                line=dict(color="yellow", width=2),
                xaxis="x", yaxis="y", showlegend=False,
            ))
            legend_added = True

    # ② 中：成交量 Volume（y2）
    traces.append(go.Bar(
        x=df["DateStr"],
        y=df["Volume"],
        name="成交量",
        marker=dict(color=volume_colors),
        xaxis="x",
        yaxis="y2",
    ))

    # ③ 下：KD（y3）
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["K"],
        mode="lines", name="K（快線）",
        line=dict(color="green"),
        xaxis="x", yaxis="y3",
    ))
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["D"],
        mode="lines", name="D（慢線）",
        line=dict(color="red"),
        xaxis="x", yaxis="y3",
    ))

    # === Layout：X 軸為 category，三層 Y 軸堆疊 ===
    layout = go.Layout(
        title=dict(text=title_text),
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),

        # 一條共用 X 軸（類別軸）
        xaxis=dict(
            type="category",      # ★ 關鍵：類別軸，只顯示有資料的日期
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="dot",
            rangeslider=dict(visible=False),
            showticklabels=False,   # ★ 不要顯示日期
            color="white",            
        ),

        # 上：價格
        yaxis=dict(
            title="價格",
            domain=[0.3, 1.0],
            color="white",
        ),

        # 中：成交量
        yaxis2=dict(
            title="成交量",
            domain=[0.12, 0.3],
            color="white",
        ),

        # 下：KD
        yaxis3=dict(
            title="KD",
            domain=[0.0, 0.1],
            showticklabels=True, # ★ 顯示 X 軸 tick labels（日期）
            color="white",
        ),

        hovermode="x",
        hoversubplots="axis",  # 使用同一個 x 的所有 subplot 都吃 hover / spike
        height=1250,
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def main():
    # 檢查 historical data 是否有檔案
    file_list = scan_csv_files()
    if not file_list:
        msg = f"在資料夾：\n{HIST_DIR}\n找不到任何 CSV 檔，請先用下載工具取得歷史資料。"
        messagebox.showerror("找不到資料", msg)
        return

    stock_map = load_stock_map()

    root = tk.Tk()
    root.title("歷史股價繪圖工具")
    root.geometry("650x400")
    root.resizable(False, False)

    tk.Label(root, text="選擇要繪圖的股票：", font=("Arial", 11)).pack(pady=5)

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=10, pady=5)

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        frame,
        font=("Consolas", 10),
        yscrollcommand=scrollbar.set,
        width=80,
        height=15,
    )
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)

    # 把檔案列表顯示出來
    for info in file_list:
        sid = info["stock_id"]
        period = info["period"]
        name = stock_map.get(sid) or stock_map.get(sid.lstrip("0")) or ""
        display = f"{sid:<6} {name:<12} ({period})" if period else f"{sid:<6} {name}"
        listbox.insert(tk.END, display)

    def on_plot():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("未選擇", "請先從列表中選擇一檔股票。")
            return
        idx = sel[0]
        info = file_list[idx]
        plot_stock(info, stock_map)

    btn = tk.Button(root, text="繪製圖表", font=("Arial", 12), command=on_plot)
    btn.pack(pady=10)

    root.mainloop()





def plot_backtest_figure(df: pd.DataFrame,
                         stock_id: str,
                         stock_name: str = "",
                         period: str | None = None) -> go.Figure:
    """
    從已包含技術指標與 buy_signal 欄位的 DataFrame 繪製三層圖：
    - 上：K 線 + MA (y)
    - 中：成交量 (y2)
    - 下：KD (y3)
    並用黑色三角形標出買點。
    """
    # 確保有 DateStr 欄位
    if "DateStr" not in df.columns:
        df = df.copy()
        df["DateStr"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # 成交量顏色：收紅為紅，收黑為綠
    volume_colors = ["red" if c >= o else "green"
                     for c, o in zip(df["Close"], df["Open"])]

    title_text = f"{stock_id} {stock_name}（{period}）" if period else f"{stock_id} {stock_name}"

    traces = []

    # ① 上：K 線 + MA (y)
    traces.append(go.Candlestick(
        x=df["DateStr"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="red",
        decreasing_line_color="green",
        name="K線",
        xaxis="x",
        yaxis="y",
    ))

    if "MA5" in df.columns:
        traces.append(go.Scatter(
            x=df["DateStr"], y=df["MA5"],
            mode="lines", name="MA5",
            line=dict(color="orange"),
            xaxis="x", yaxis="y",
        ))
    if "MA20" in df.columns:
        traces.append(go.Scatter(
            x=df["DateStr"], y=df["MA20"],
            mode="lines", name="MA20",
            line=dict(color="blue"),
            xaxis="x", yaxis="y",
        ))
    if "MA60" in df.columns:
        traces.append(go.Scatter(
            x=df["DateStr"], y=df["MA60"],
            mode="lines", name="MA60",
            line=dict(color="purple"),
            xaxis="x", yaxis="y",
        ))
    if "MA120" in df.columns:
        traces.append(go.Scatter(
            x=df["DateStr"], y=df["MA120"],
            mode="lines", name="MA120",
            line=dict(color="brown"),
            xaxis="x", yaxis="y",
        ))

    raff_channels = compute_raff_channels(df, min_length=20, max_length=80)

    if raff_channels:
        legend_added = False
        for channel in raff_channels:
            showlegend = not legend_added
            name_suffix = f"（{channel['window']}日）"
            traces.append(go.Scatter(
                x=channel["x"], y=channel["upper"],
                mode="lines", name=f"Raff 上緣{name_suffix}",
                line=dict(color="yellow", width=2),
                xaxis="x", yaxis="y", showlegend=showlegend,
            ))
            traces.append(go.Scatter(
                x=channel["x"], y=channel["lower"],
                mode="lines", name=f"Raff 下緣{name_suffix}",
                line=dict(color="yellow", width=2),
                xaxis="x", yaxis="y", showlegend=False,
            ))
            legend_added = True

    # ② 中：成交量 Volume（y2）
    traces.append(go.Bar(
        x=df["DateStr"],
        y=df["Volume"],
        name="成交量",
        marker=dict(color=volume_colors),
        xaxis="x",
        yaxis="y2",
    ))

    # ③ 下：KD（y3）
    if "K" in df.columns and "D" in df.columns:
        traces.append(go.Scatter(
            x=df["DateStr"], y=df["K"],
            mode="lines", name="K（快線）",
            line=dict(color="green"),
            xaxis="x", yaxis="y3",
        ))
        traces.append(go.Scatter(
            x=df["DateStr"], y=df["D"],
            mode="lines", name="D（慢線）",
            line=dict(color="red"),
            xaxis="x", yaxis="y3",
        ))

    # 買點標記（白色三角形）
    if "buy_signal" in df.columns and df["buy_signal"].any():
        buy_mask = df["buy_signal"].astype(bool)
        buy_prices = df.loc[buy_mask, "Close"]
        traces.append(go.Scatter(
            x=df.loc[buy_mask, "DateStr"],
            y=buy_prices * 0.95,
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="white", line=dict(color="white", width=1)),
            name="買點",
            customdata=buy_prices,
            hovertemplate="買點<br>收盤：%{customdata}<extra></extra>",
            xaxis="x",
            yaxis="y",
        ))

    # 賣點標記（白色倒三角形）
    if "sell_signal" in df.columns and df["sell_signal"].any():
        sell_mask = df["sell_signal"].astype(bool)
        sell_prices = df.loc[sell_mask, "Close"]
        traces.append(go.Scatter(
            x=df.loc[sell_mask, "DateStr"],
            y=sell_prices * 1.05,
            mode="markers",
            marker=dict(symbol="triangle-down-open", size=14, color="white", line=dict(color="white", width=1)),
            name="賣點",
            customdata=sell_prices,
            hovertemplate="賣點<br>收盤：%{customdata}<extra></extra>",
            xaxis="x",
            yaxis="y",
        ))

    # 移動停損賣出標記（紅色倒三角形）
    if "trailing_stop_sell" in df.columns and df["trailing_stop_sell"].any():
        ts_mask = df["trailing_stop_sell"].astype(bool)
        ts_prices = df.loc[ts_mask, "Close"]
        traces.append(go.Scatter(
            x=df.loc[ts_mask, "DateStr"],
            y=ts_prices * 1.05,
            mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="red"),
            name="移動停損賣出",
            customdata=ts_prices,
            hovertemplate="移動停損<br>收盤：%{customdata}<extra></extra>",
            xaxis="x",
            yaxis="y",
        ))

    # 移動停利賣出標記（綠色倒三角形）
    if "trailing_profit_sell" in df.columns and df["trailing_profit_sell"].any():
        tp_mask = df["trailing_profit_sell"].astype(bool)
        tp_prices = df.loc[tp_mask, "Close"]
        traces.append(go.Scatter(
            x=df.loc[tp_mask, "DateStr"],
            y=tp_prices * 1.05,
            mode="markers",
            marker=dict(symbol="triangle-down", size=14, color="green"),
            name="移動停利賣出",
            customdata=tp_prices,
            hovertemplate="移動停利<br>收盤：%{customdata}<extra></extra>",
            xaxis="x",
            yaxis="y",
        ))

    layout = go.Layout(
        title=dict(text=title_text),
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white"),
        xaxis=dict(
            type="category",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="dot",
            rangeslider=dict(visible=False),
            showticklabels=False,
            color="white",
        ),
        yaxis=dict(title="價格", domain=[0.3, 1.0], color="white"),
        yaxis2=dict(title="成交量", domain=[0.12, 0.3], color="white"),
        yaxis3=dict(title="KD", domain=[0.0, 0.1], showticklabels=True, color="white"),
        hovermode="x",
        hoversubplots="axis",
        height=1250,
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    return fig


if __name__ == "__main__":
    main()
