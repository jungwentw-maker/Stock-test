import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import graphic


# === 路徑設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(BASE_DIR, "historical data")
STOCK_MAP_PATH = os.path.join(BASE_DIR, "content", "stock_id.csv")

# === 回測設定（之後你可以改） ===
BUY_MODE = "amount"        # "shares"：固定股數；"amount"：固定金額
SHARES_PER_TRADE = 1       # 每次買幾股（BUY_MODE = "amount" 時有效）
AMOUNT_PER_TRADE = 1000   # 每次買入金額（BUY_MODE = "amount" 時有效）


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


def plot_stock(file_info, stock_name_map, years, buy_mode, shares_per_trade, amount_per_trade):

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

    # ======== ★ 僅保留最近 N 年資料（由 GUI 傳入） ========
    last_date = df["Date"].max()
    if years is not None and years > 0:
        days = int(years * 365.25)
        df = df[df["Date"] >= last_date - pd.Timedelta(days=days)].copy()
    else:
        df = df.copy()
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
    # 根據 GUI 傳入的參數決定實際買進設定
    mode = buy_mode if buy_mode else BUY_MODE
    shares_cfg = shares_per_trade if shares_per_trade not in (None, "") else SHARES_PER_TRADE
    amount_cfg = amount_per_trade if amount_per_trade not in (None, "") else AMOUNT_PER_TRADE

    # 超賣區回看天數 N：
    # N = 0：當天 K<20 且 KD cross
    # N = 1：前一天 K<20 且今天 KD cross（目前在用的邏輯）
    # N >= 2：交叉前 N 天內曾經有 K<20
    oversold_lookback = 1  # 這個 N 之後可以從 GUI 或設定帶進來

    # KD 黃金交叉：今天 K>D，昨天 K<D
    kd_cross = (df["K"] > df["D"]) & (df["K"].shift(1) < df["D"].shift(1))

    if oversold_lookback == 0:
        # 當天 K<20 且 KD cross
        oversold_cond = df["K"] < 20
    else:
        # 交叉前 N 天內曾經 K<20
        oversold_cond = df["K"].shift(1).rolling(window=oversold_lookback).min() < 20

    df["buy_signal"] = oversold_cond & kd_cross

    trades = []
    total_shares = 0
    final_close = df["Close"].iloc[-1]

    for _, row in df[df["buy_signal"]].iterrows():
        date = row["Date"]
        close = row["Close"]

        if mode == "shares":
            shares = int(shares_cfg)
        elif mode == "amount":
            shares = int(amount_cfg // close)
        else:
            raise ValueError(f"未知的 BUY_MODE: {mode}")

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

        name = stock_name_map.get(stock_id) or stock_name_map.get(stock_id.lstrip("0")) or ""
        if mode == "shares":
            buy_desc = f"每次買 {int(shares_cfg)} 股"
        elif mode == "amount":
            buy_desc = f"每次買 {amount_cfg:,.0f} 元"
        else:
            buy_desc = ""
        print(f"========== 回測{years}年 {stock_id} {name} ==========")
        print(f"{buy_desc}")
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


    # 使用 graphic 模組繪圖（K線 + 成交量 + KD + 買點）
    name = stock_name_map.get(stock_id) or stock_name_map.get(stock_id.lstrip("0")) or ""
    graphic.plot_backtest_figure(df, stock_id, name, period)
def main():
    file_list = scan_csv_files()
    if not file_list:
        messagebox.showerror("找不到資料",
                             f"請先下載 CSV 至：\n{HIST_DIR}")
        return

    stock_map = load_stock_map()

    root = tk.Tk()
    root.title("歷史股價繪圖工具")
    root.geometry("650x500")
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

    # === 回測期間單選（1 / 3 / 5 / 10 年） ===
    year_var = tk.IntVar(value=5)
    frame_year = tk.Frame(root)
    frame_year.pack(pady=5)
    tk.Label(frame_year, text="回測期間：").pack(side="left")
    for y in (1, 3, 5, 10):
        tk.Radiobutton(frame_year, text=f"{y}年", variable=year_var, value=y).pack(side="left")

    # === 買進模式單選（固定股數 / 固定金額） ===
    mode_var = tk.StringVar(value="amount")
    frame_mode = tk.Frame(root)
    frame_mode.pack(pady=5)
    tk.Label(frame_mode, text="買進模式：").pack(side="left")

    # === 數值輸入欄位（依模式切換） ===
    frame_input = tk.Frame(root)
    frame_input.pack(pady=5, fill="x")
    label_value = tk.Label(frame_input, text="每次買入金額：")
    label_value.pack(anchor="w")

    entry_shares = tk.Entry(frame_input)
    entry_amount = tk.Entry(frame_input)
    entry_amount.insert(0, str(AMOUNT_PER_TRADE))
    entry_amount.pack(fill="x")

    def update_input_fields():
        mode = mode_var.get()
        if mode == "shares":
            label_value.config(text="每次買入股數：")
            entry_amount.pack_forget()
            if not entry_shares.winfo_ismapped():
                entry_shares.pack(fill="x")
        else:
            label_value.config(text="每次買入金額：")
            entry_shares.pack_forget()
            if not entry_amount.winfo_ismapped():
                entry_amount.pack(fill="x")

    tk.Radiobutton(frame_mode, text="固定股數",
                   variable=mode_var, value="shares",
                   command=update_input_fields).pack(side="left")
    tk.Radiobutton(frame_mode, text="固定金額",
                   variable=mode_var, value="amount",
                   command=update_input_fields).pack(side="left")

    update_input_fields()

    def on_plot():
        sel = listbox.curselection()
        if not sel:
            messagebox.showwarning("未選擇", "請選擇一檔股票")
            return
        idx = sel[0]
        info = file_list[idx]

        years = year_var.get()
        buy_mode = mode_var.get()

        shares_per_trade = None
        amount_per_trade = None

        if buy_mode == "shares":
            val = entry_shares.get().strip()
            if not val:
                messagebox.showerror("輸入錯誤", "請輸入每次買入股數")
                return
            try:
                shares_per_trade = int(val)
            except ValueError:
                messagebox.showerror("輸入錯誤", "股數必須是正整數")
                return
            if shares_per_trade <= 0:
                messagebox.showerror("輸入錯誤", "股數必須大於 0")
                return
        else:
            val = entry_amount.get().strip()
            if not val:
                messagebox.showerror("輸入錯誤", "請輸入每次買入金額")
                return
            try:
                amount_per_trade = float(val)
            except ValueError:
                messagebox.showerror("輸入錯誤", "金額必須是數字")
                return
            if amount_per_trade <= 0:
                messagebox.showerror("輸入錯誤", "金額必須大於 0")
                return

        plot_stock(info, stock_map, years, buy_mode, shares_per_trade, amount_per_trade)

    tk.Button(root, text="繪圖&回測", font=("Arial", 12), command=on_plot).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()
