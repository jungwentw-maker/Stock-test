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

# === 回測設定（之後你可以改） ===
BUY_MODE = "shares"        # "shares"：固定股數；"amount"：固定金額
SHARES_PER_TRADE = 1       # 每次買幾股（BUY_MODE="shares" 時有效）
AMOUNT_PER_TRADE = 20000   # 每次買入金額（BUY_MODE="amount" 時有效）


def load_stock_map():
    mapping = {}
    if not os.path.exists(STOCK_MAP_PATH):
        return mapping

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

        mapping[sid] = name
        mapping[sid.lstrip("0")] = name

    return mapping


def scan_csv_files():
    files = []
    if not os.path.isdir(HIST_DIR):
        return files

    for fname in os.listdir(HIST_DIR):
        if not fname.lower().endswith(".csv"):
            continue

        base = fname[:-4]
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
        df.loc[i, "K"] = df.loc[i - 1, "K"] * 2/3 + df.loc[i, "RSV"] * 1/3
        df.loc[i, "D"] = df.loc[i - 1, "D"] * 2/3 + df.loc[i, "K"] * 1/3

    return df


def plot_stock(file_info, stock_name_map):

    csv_path = file_info["path"]
    stock_id = file_info["stock_id"]
    period = file_info["period"]

    # 讀取 CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        messagebox.showerror("欄位錯誤",
                             f"{csv_path}\n缺少必要欄位：{required_cols - set(df.columns)}")
        return

    # 日期處理
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df.sort_values("Date", inplace=True)

    # ======== ★ 僅保留最近五年資料 ========
    last_date = df["Date"].max()
    df = df[df["Date"] >= last_date - pd.Timedelta(days=1826)].copy()
    df.reset_index(drop=True, inplace=True)

    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")

    # 移動平均線
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # KD
    df = compute_kd(df)

    # ======== ★ 回測：K < 20 ========
    df["buy_signal"] = df["K"] < 20

    trades = []
    total_shares = 0
    final_close = df["Close"].iloc[-1]

    for _, row in df[df["buy_signal"]].iterrows():
        date = row["Date"]
        close = row["Close"]

        if BUY_MODE == "shares":
            shares = SHARES_PER_TRADE
        else:
            shares = int(AMOUNT_PER_TRADE // close)
            if shares <= 0:
                continue

        total_shares += shares
        cost = close * shares

        trades.append({
            "日期": date.strftime("%Y-%m-%d"),
            "買入價": close,
            "股數": shares,
            "成本": cost
        })

    trade_df = pd.DataFrame(trades)

    # ========== ★★ 正確縮排 — 回測摘要 + xlsx ★★ ==========
    if len(trade_df) > 0:

        trade_df["期末價"] = final_close
        trade_df["最終市值"] = trade_df["股數"] * final_close
        trade_df["獲利"] = trade_df["最終市值"] - trade_df["成本"]
        trade_df["報酬率%"] = trade_df["獲利"] / trade_df["成本"] * 100

        # 輸出 Excel
        out_path = os.path.join(BASE_DIR, "result.xlsx")
        trade_df.to_excel(out_path, index=False)

        # 摘要
        total_cost = trade_df["成本"].sum()
        final_value = total_shares * final_close
        total_profit = final_value - total_cost
        profit_pct = (total_profit / total_cost) * 100 if total_cost > 0 else 0

        print("========== 回測摘要（最近五年） ==========")
        print(f"總買入次數：{len(trade_df)}")
        print(f"累積買進股數：{total_shares}")
        print(f"總成本：{total_cost:,.2f}")
        print(f"最後一天收盤：{final_close}")
        print(f"最終市值：{final_value:,.2f}")
        print(f"總報酬：{total_profit:,.2f}")

        sign = "+" if profit_pct >= 0 else "-"
        print(f"總報酬百分比：{sign}{abs(profit_pct):.2f}%")
        print("=========================================")

    else:
        print("⚠ 近五年內沒有任何 K<20 的買進訊號（不會產生 result.xlsx）。")

    # ============= 原本的繪圖區域（完全不動） =============
    volume_colors = ["red" if c >= o else "green"
                     for c, o in zip(df["Close"], df["Open"])]

    name = stock_name_map.get(stock_id) or stock_name_map.get(stock_id.lstrip("0")) or ""
    title_text = f"{stock_id} {name}（{period}）" if period else f"{stock_id} {name}"

    traces = []

    traces.append(go.Candlestick(
        x=df["DateStr"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="red", decreasing_line_color="green",
        name="K線", xaxis="x", yaxis="y"
    ))

    traces.append(go.Scatter(x=df["DateStr"], y=df["MA5"], mode="lines",
                             name="MA5", line=dict(color="orange"), xaxis="x", yaxis="y"))
    traces.append(go.Scatter(x=df["DateStr"], y=df["MA20"], mode="lines",
                             name="MA20", line=dict(color="blue"), xaxis="x", yaxis="y"))
    traces.append(go.Scatter(x=df["DateStr"], y=df["MA60"], mode="lines",
                             name="MA60", line=dict(color="purple"), xaxis="x", yaxis="y"))
    traces.append(go.Scatter(x=df["DateStr"], y=df["MA120"], mode="lines",
                             name="MA120", line=dict(color="brown"), xaxis="x", yaxis="y"))

    traces.append(go.Bar(
        x=df["DateStr"], y=df["Volume"], name="成交量",
        marker=dict(color=volume_colors), xaxis="x", yaxis="y2"
    ))

    traces.append(go.Scatter(
        x=df["DateStr"], y=df["K"], mode="lines",
        name="K（快線）", line=dict(color="green"), xaxis="x", yaxis="y3"
    ))
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["D"], mode="lines",
        name="D（慢線）", line=dict(color="red"), xaxis="x", yaxis="y3"
    ))

    if df["buy_signal"].any():
        traces.append(go.Scatter(
            x=df.loc[df["buy_signal"], "DateStr"],
            y=df.loc[df["buy_signal"], "Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=14, color="black"),
            name="買點(K<20)", xaxis="x", yaxis="y"
        ))

    layout = go.Layout(
        title=dict(text=title_text),
        template="plotly_white",
        xaxis=dict(type="category", showspikes=True,
                   spikemode="across", spikesnap="cursor",
                   rangeslider=dict(visible=False), showticklabels=False),
        yaxis=dict(title="價格", domain=[0.3, 1.0]),
        yaxis2=dict(title="成交量", domain=[0.12, 0.3]),
        yaxis3=dict(title="KD", domain=[0.0, 0.1], showticklabels=True),
        hovermode="x",
        hoversubplots="axis",
        height=1250,
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()


def main():
    file_list = scan_csv_files()
    if not file_list:
        messagebox.showerror("找不到資料",
                             f"請先下載 CSV 至：\n{HIST_DIR}")
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

    for info in file_list:
        sid = info["stock_id"]
        period = info["period"]
        name = stock_map.get(sid) or stock_map.get(sid.lstrip("0")) or ""
        display = f"{sid:<6} {name:<12} ({period})" if period else f"{sid:<6} {name}"
        listbox.insert(tk.END, display)

    def on_plot():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("未選擇", "請選擇一檔股票")
            return
        idx = sel[0]
        info = file_list[idx]
        plot_stock(info, stock_map)

    tk.Button(root, text="繪製圖表", font=("Arial", 12), command=on_plot).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
