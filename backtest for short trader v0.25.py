import os
import traceback
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # 目前沒直接用到，先保留
import graphic


# === 路徑設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(BASE_DIR, "historical data")
STOCK_MAP_PATH = os.path.join(BASE_DIR, "content", "stock_id.csv")


def write_error_log(tag: str) -> str:
    error_log_path = os.path.join(BASE_DIR, "error.log")
    with open(error_log_path, "a", encoding="utf-8") as f:
        f.write(f"\n=== {tag} ===\n")
        f.write(traceback.format_exc())
    return error_log_path

# === 回測設定（預設值，可被 GUI 覆蓋） ===
BUY_MODE = "amount"        # "shares"：固定股數；"amount"：固定金額
SHARES_PER_TRADE = 1       # 每次買幾股（BUY_MODE = "shares" 時有效）
AMOUNT_PER_TRADE = 1000    # 每次買入金額（BUY_MODE = "amount" 時有效）

SELL_MODE = "shares"       # "shares"：固定股數；"amount"：固定金額
SELL_SHARES_PER_TRADE = 1  # 每次賣幾股（SELL_MODE = "shares" 時有效）
SELL_AMOUNT_PER_TRADE = 1000  # 每次賣出金額（SELL_MODE = "amount" 時有效）

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
        df.loc[i, "K"] = df.loc[i - 1, "K"] * 2 / 3 + df.loc[i, "RSV"] * 1 / 3
        df.loc[i, "D"] = df.loc[i - 1, "D"] * 2 / 3 + df.loc[i, "K"] * 1 / 3

    return df


def plot_stock(
    file_info,
    stock_name_map,
    years,
    buy_mode,
    shares_per_trade,
    amount_per_trade,
    trigger_periodic,
    sell_mode,
    sell_shares_per_trade,
    sell_amount_per_trade,
    weeks_interval,
    trigger_kd,
    oversold_lookback,
    start_mode,
    start_date,
    initial_capital,
    sell_conditions,
    trailing_stop_enabled,
    trailing_stop_pct,  
    trailing_profit_enabled,
    trailing_profit_pct,      
):
    """
    讀取單一 CSV、依參數執行回測與繪圖。
    """

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
        messagebox.showerror(
            "欄位錯誤",
            f"{csv_path}\n缺少必要欄位：{required_cols - set(df.columns)}"
        )
        return

    # 日期處理
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df.sort_values("Date", inplace=True)

    # ======== ★ 僅保留最近 N 年資料或自訂起始日（由 GUI 傳入） ========
    last_date = df["Date"].max()
    window_start = None
    window_end = None

    # 自訂日期：視為「區間終點」，再往回推 years 年得到起點
    if start_mode == "custom" and start_date is not None:
        anchor_date = start_date.normalize()
        end_date = min(anchor_date, last_date)
        if years is not None and years > 0:
            start_cut = anchor_date - pd.DateOffset(years=years)

        else:
            start_cut = anchor_date

        df = df[(df["Date"] >= start_cut) & (df["Date"] <= end_date)].copy()
        window_start, window_end = start_cut, end_date
    else:
        if years is not None and years > 0:
            days = int(years * 365.25)
            start_cut = last_date - pd.Timedelta(days=days)
            df = df[df["Date"] >= start_cut].copy()
            window_start, window_end = start_cut, last_date
        else:
            df = df.copy()
            window_start, window_end = df["Date"].min(), last_date


    df.reset_index(drop=True, inplace=True)
    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")

    # 移動平均線
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # KD
    df = compute_kd(df)

    # ======== ★ 回測觸發條件 ========
    mode = buy_mode if buy_mode else BUY_MODE
    shares_cfg = shares_per_trade if shares_per_trade not in (None, "") else SHARES_PER_TRADE
    amount_cfg = amount_per_trade if amount_per_trade not in (None, "") else AMOUNT_PER_TRADE

    sell_mode_cfg = sell_mode if sell_mode else SELL_MODE
    sell_shares_cfg = (
        sell_shares_per_trade
        if sell_shares_per_trade not in (None, "")
        else SELL_SHARES_PER_TRADE
    )
    sell_amount_cfg = (
        sell_amount_per_trade
        if sell_amount_per_trade not in (None, "")
        else SELL_AMOUNT_PER_TRADE
    )

    # 建立一個空的買點訊號（之後用 OR 疊加不同條件）
    buy_signal = pd.Series(False, index=df.index)

    # --- KD 觸發條件（K<20 且 KD 反轉，支援 N 天內曾低於 20） ---
    if trigger_kd:
        try:
            lookback = int(oversold_lookback) if oversold_lookback not in (None, "") else 1
        except ValueError:
            lookback = 1
        if lookback < 0:
            lookback = 0

        # KD 黃金交叉：今天 K > D，昨天 K < D
        kd_cross = (df["K"] > df["D"]) & (df["K"].shift(1) < df["D"].shift(1))

        if lookback == 0:
            # 當天 K<20 且 KD cross
            oversold_cond = df["K"] < 20
        else:
            # 交叉前 N 天內曾經 K<20
            oversold_cond = df["K"].shift(1).rolling(window=lookback).min() < 20

        buy_signal = buy_signal | (oversold_cond & kd_cross)

    # --- 定期買入條件（從篩選後的第一天起，每隔 weeks_interval*5 個交易日買入一次） ---
    if trigger_periodic and weeks_interval:
        step = max(int(weeks_interval) * 5, 1)
        periodic_signal = pd.Series(False, index=df.index)
        periodic_signal.iloc[::step] = True
        buy_signal = buy_signal | periodic_signal

    df["buy_signal"] = buy_signal

    # === 停利/停損賣出條件（多選且需同時滿足） ===
    cond_list = []
    if sell_conditions.get("ma5"):
        cond_list.append((df["Close"].shift(1) > df["MA5"].shift(1)) & (df["Close"] < df["MA5"]))
    if sell_conditions.get("ma20"):
        cond_list.append((df["Close"].shift(1) > df["MA20"].shift(1)) & (df["Close"] < df["MA20"]))
    if sell_conditions.get("vol_prev_day"):
        cond_list.append(df["Volume"] > df["Volume"].shift(1) * 1.5)
    if sell_conditions.get("vol_avg5"):
        cond_list.append(df["Volume"] > df["Volume"].shift(1).rolling(5).mean() * 1.5)
    if sell_conditions.get("drop_3pct"):
        cond_list.append(df["Close"] < df["Open"] * 0.97)
    if sell_conditions.get("kd_reversal"):
        cond_list.append((df["K"].shift(1) > df["D"].shift(1)) & (df["K"].shift(1) > 80) & (df["D"] > df["K"]))

    if cond_list:
        sell_signal = cond_list[0]
        for c in cond_list[1:]:
            sell_signal &= c
    else:
        sell_signal = pd.Series(False, index=df.index)

    df["sell_signal"] = sell_signal.fillna(False)

    trailing_stop_enabled = bool(trailing_stop_enabled)
    trailing_stop_pct = float(trailing_stop_pct) if trailing_stop_enabled else 0.0
    trailing_profit_enabled = bool(trailing_profit_enabled)
    trailing_profit_pct = float(trailing_profit_pct) if trailing_profit_enabled else 0.0

    # ======== ★ 依買 / 賣訊號逐日模擬交易 ========
    trades = []
    position_shares = 0
    avg_cost = 0.0  # 加權平均成本
    cost_basis = 0.0  # 目前持股的總成本（position_shares * avg_cost）
    capital = float(initial_capital)
    highest_since_entry = None  # 進場後的最高價（用於移動停損 / 移動停利）

    trailing_stop_marks = pd.Series(False, index=df.index)
    trailing_profit_marks = pd.Series(False, index=df.index)

    if len(df) == 0:
        messagebox.showwarning("資料不足", "篩選後沒有任何歷史資料可供回測")
        return

    # 逐日處理，預設「先賣後買」：先處理停利/停損，再判斷買進訊號
    for i, row in df.iterrows():
        date = row["Date"]
        close = row["Close"]
        intraday_high = row["High"]

        prev_highest = highest_since_entry
        if position_shares > 0:
            if prev_highest is None:
                prev_highest = intraday_high
            intraday_peak = max(prev_highest, intraday_high)
        else:
            intraday_peak = None

        trailing_stop_trigger = False
        trailing_profit_trigger = False
        if position_shares > 0 and intraday_peak is not None:
            if trailing_stop_enabled:
                stop_threshold = intraday_peak * (1 - trailing_stop_pct / 100)
                trailing_stop_trigger = close <= stop_threshold

            if trailing_profit_enabled:
                profit_threshold = intraday_peak * (1 + trailing_profit_pct / 100)
                trailing_profit_trigger = close >= profit_threshold

        # 1) 停利/停損賣出
        if (row["sell_signal"] or trailing_stop_trigger or trailing_profit_trigger) and position_shares > 0:
            if sell_mode_cfg == "shares":
                desired_sell = int(sell_shares_cfg)
            elif sell_mode_cfg == "amount":
                desired_sell = int(sell_amount_cfg // close)
            else:
                raise ValueError(f"未知的 SELL_MODE: {sell_mode_cfg}")

            sell_shares = min(desired_sell, position_shares)
            if sell_shares > 0:
                proceeds = close * sell_shares
                realized_pnl = (close - avg_cost) * sell_shares

                capital += proceeds

                position_shares -= sell_shares
                cost_basis -= avg_cost * sell_shares
                if position_shares == 0:
                    avg_cost = 0.0
                    cost_basis = 0.0
                    highest_since_entry = None

                if trailing_stop_trigger:
                    trailing_stop_marks.iat[i] = True
                if trailing_profit_trigger:
                    trailing_profit_marks.iat[i] = True

                trades.append({
                    "方向": "SELL",
                    "日期": date.strftime("%Y-%m-%d"),
                    "成交價": close,
                    "股數": sell_shares,
                    "金額": proceeds,
                    "剩餘資金": capital,
                    "持股數(收盤後)": position_shares,
                    "單筆實現損益": realized_pnl,
                })

        if position_shares == 0:
            highest_since_entry = None
        else:
            highest_since_entry = intraday_peak

        # 2) 買進訊號
        if row["buy_signal"]:
            if mode == "shares":
                buy_shares = int(shares_cfg)
                buy_shares = min(buy_shares, int(capital // close))
            elif mode == "amount":
                allowed_amount = min(float(amount_cfg), capital)
                buy_shares = int(allowed_amount // close)
            else:
                raise ValueError(f"未知的 BUY_MODE: {mode}")

            if buy_shares > 0:
                cost = close * buy_shares
                position_shares += buy_shares
                capital -= cost
                cost_basis += cost
                avg_cost = cost_basis / position_shares if position_shares > 0 else 0
                highest_since_entry = intraday_high if highest_since_entry is None else max(highest_since_entry, intraday_high)
                                
                trades.append({
                    "方向": "BUY",
                    "日期": date.strftime("%Y-%m-%d"),
                    "成交價": close,
                    "股數": buy_shares,
                    "金額": cost,
                    "剩餘資金": capital,                   
                    "持股數(收盤後)": position_shares,
                    "單筆實現損益": 0.0,
                })

    df["trailing_stop_sell"] = trailing_stop_marks
    df["trailing_profit_sell"] = trailing_profit_marks

    trade_df = pd.DataFrame(trades)

    if len(trade_df) > 0 or position_shares > 0:
        final_close = df["Close"].iloc[-1]
        final_value = position_shares * final_close
        final_capital = capital + final_value
        total_realized = trade_df["單筆實現損益"].sum() if len(trade_df) > 0 else 0
        unrealized = (final_close - avg_cost) * position_shares if position_shares > 0 else 0
        total_profit = final_capital - float(initial_capital)

        total_buy = trade_df.loc[trade_df["方向"] == "BUY", "金額"].sum() if len(trade_df) > 0 else 0
        total_sell = trade_df.loc[trade_df["方向"] == "SELL", "金額"].sum() if len(trade_df) > 0 else 0

        if len(trade_df) > 0:
            trade_df["期末價"] = final_close
            trade_df["期末持股"] = position_shares
            trade_df["最終市值"] = final_value
            trade_df["累計實現損益"] = total_realized
            trade_df["未實現損益"] = unrealized
            trade_df["最終資金池"] = final_capital
            trade_df["總報酬"] = total_profit

        # 輸出 Excel
        out_path = os.path.join(BASE_DIR, "result.xlsx")
        trade_df.to_excel(out_path, index=False)

        name = stock_name_map.get(stock_id) or stock_name_map.get(stock_id.lstrip("0")) or ""

        if mode == "shares":
            buy_desc = f"每次買 {int(shares_cfg)} 股"
        elif mode == "amount":
            buy_desc = f"每次買 {amount_cfg:,.0f} 元"
        else:
            buy_desc = ""

        if sell_mode_cfg == "shares":
            sell_desc = f"每次賣 {int(sell_shares_cfg)} 股"
        elif sell_mode_cfg == "amount":
            sell_desc = f"每次賣 {sell_amount_cfg:,.0f} 元"
        else:
            sell_desc = ""

        if window_start is not None and window_end is not None:
            header_range = f"{window_start.strftime('%Y-%m-%d')} ~ {window_end.strftime('%Y-%m-%d')}"
        elif start_mode == "custom" and start_date is not None:
            header_range = f"自 {start_date.strftime('%Y-%m-%d')} 起"
        else:
            header_range = f"{years}年"

        print(f"========== 回測{header_range} {stock_id} {name} ==========")
        if buy_desc:
            print(buy_desc)
        if sell_desc:
            print(sell_desc)
        print(f"總交易筆數：{len(trade_df)}（買：{(trade_df['方向']=='BUY').sum()}，賣：{(trade_df['方向']=='SELL').sum()}）")
        print(f"累積買進股數：{trade_df.loc[trade_df['方向']=='BUY','股數'].sum()}")
        print(f"累積賣出股數：{trade_df.loc[trade_df['方向']=='SELL','股數'].sum()}")
        print(f"總買入金額：{total_buy:,.2f}")
        print(f"總賣出金額：{total_sell:,.2f}")
        print(f"最後一天收盤：{final_close}")
        print(f"期末持股數：{position_shares}")
        print(f"期末市值：{final_value:,.2f}")
        print(f"已實現損益：{total_realized:,.2f}")
        print(f"未實現損益：{unrealized:,.2f}")
        print(f"期末資金池（含市值）：{final_capital:,.2f}")
        print(f"總報酬（期末資金池-初始資金池）：{total_profit:,.2f}")
        profit_pct = (total_profit / float(initial_capital)) * 100
        sign = "+" if profit_pct >= 0 else "-"
        print(f"總報酬百分比：{sign}{abs(profit_pct):.2f}%")
        print("=========================================")
    else:
        print("⚠ 篩選期間內沒有任何交易訊號（不會產生 result.xlsx）。")

    # 使用 graphic 模組繪圖（K線 + 成交量 + KD + 買點）
    name = stock_name_map.get(stock_id) or stock_name_map.get(stock_id.lstrip("0")) or ""
    graphic.plot_backtest_figure(df, stock_id, name, period)


def main():
    file_list = scan_csv_files()
    if not file_list:
        messagebox.showerror(
            "找不到資料",
            f"請先下載 CSV 至：\n{HIST_DIR}"
        )
        return

    stock_map = load_stock_map()

    root = tk.Tk()
    root.title("歷史股價繪圖工具")
    root.geometry("750x900")
    root.resizable(False, False)

    tk.Label(root, text="選擇要繪圖的股票：", font=("Arial", 11)).pack(pady=5)

    # 股票列表
    frame_list = tk.Frame(root)
    frame_list.pack(fill="both", expand=True, padx=10, pady=5)

    scrollbar = tk.Scrollbar(frame_list)
    scrollbar.pack(side="right", fill="y")

    listbox = tk.Listbox(
        frame_list,
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

    # 共用控制區，用 grid 對齊各標題
    controls = tk.Frame(root)
    controls.pack(pady=5, fill="x", padx=10)
    controls.columnconfigure(1, weight=1)

    # === 回測開始日（預設 / 自訂） ===
    start_mode_var = tk.StringVar(value="default")

    tk.Label(controls, text="回測開始日：")\
        .grid(row=0, column=0, sticky="w", padx=(0, 5))

    frame_start = tk.Frame(controls)
    frame_start.grid(row=0, column=1, sticky="w")

    rb_start_default = tk.Radiobutton(
        frame_start,
        text="預設（往前推回測期間）",
        variable=start_mode_var,
        value="default",
    )
    rb_start_default.pack(side="left", padx=5)

    rb_start_custom = tk.Radiobutton(
        frame_start,
        text="自訂：",
        variable=start_mode_var,
        value="custom",
    )
    rb_start_custom.pack(side="left")

    entry_start_date = tk.Entry(frame_start, width=12)
    entry_start_date.pack(side="left", padx=5)

    def update_start_mode(*args):
        if start_mode_var.get() == "custom":
            entry_start_date.config(state="normal")
        else:
            entry_start_date.config(state="disabled")

    start_mode_var.trace_add("write", update_start_mode)
    update_start_mode()

    # === 回測期間單選（1 / 3 / 5 / 10 年） ===
    year_var = tk.IntVar(value=5)

    tk.Label(controls, text="回測期間：")\
        .grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))

    frame_year = tk.Frame(controls)
    frame_year.grid(row=1, column=1, sticky="w", pady=(5, 0))

    for y in (1, 3, 5, 10):
        tk.Radiobutton(frame_year, text=f"{y}年",
                       variable=year_var, value=y).pack(side="left")

    # === 觸發條件 ===
    tk.Label(controls, text="觸發條件：")\
        .grid(row=2, column=0, sticky="nw", padx=(0, 5), pady=(5, 0))

    frame_trigger = tk.Frame(controls)
    frame_trigger.grid(row=2, column=1, sticky="w", pady=(5, 0))

    trigger_periodic_var = tk.BooleanVar(value=False)
    trigger_kd_var = tk.BooleanVar(value=True)

    chk_periodic = tk.Checkbutton(frame_trigger, text="定期每",
                                  variable=trigger_periodic_var)
    chk_periodic.grid(row=0, column=0, sticky="w")

    entry_weeks = tk.Entry(frame_trigger, width=4)
    entry_weeks.insert(0, "1")
    entry_weeks.grid(row=0, column=1, sticky="w")
    tk.Label(frame_trigger, text="周").grid(row=0, column=2, sticky="w")

    chk_kd = tk.Checkbutton(frame_trigger, text="K<20 且 KD 反轉，N=",
                             variable=trigger_kd_var)
    chk_kd.grid(row=1, column=0, sticky="w", pady=(2, 0))

    entry_N = tk.Entry(frame_trigger, width=4)
    entry_N.insert(0, "1")
    entry_N.grid(row=1, column=1, sticky="w", pady=(2, 0))

     # === 停利 / 停損條件（交集） ===
    tk.Label(controls, text="停利 / 停損條件：")\
        .grid(row=3, column=0, sticky="nw", padx=(0, 5), pady=(5, 0))

    frame_sell_cond = tk.Frame(controls)
    frame_sell_cond.grid(row=3, column=1, sticky="w", pady=(5, 0))

    sell_ma5_var = tk.BooleanVar(value=False)
    sell_ma20_var = tk.BooleanVar(value=False)
    sell_vol_prev_var = tk.BooleanVar(value=False)
    sell_vol_avg5_var = tk.BooleanVar(value=False)
    sell_drop_var = tk.BooleanVar(value=False)
    sell_kd_rev_var = tk.BooleanVar(value=False)

    tk.Checkbutton(frame_sell_cond, text="跌破5MA", variable=sell_ma5_var).grid(row=0, column=0, sticky="w")
    tk.Checkbutton(frame_sell_cond, text="跌破20MA", variable=sell_ma20_var).grid(row=0, column=1, sticky="w", padx=(10, 0))
    tk.Checkbutton(frame_sell_cond, text="大量（>前日1.5倍）", variable=sell_vol_prev_var).grid(row=1, column=0, sticky="w", pady=(2, 0))
    tk.Checkbutton(frame_sell_cond, text="大量（>前5日均量1.5倍）", variable=sell_vol_avg5_var).grid(row=1, column=1, sticky="w", padx=(10, 0), pady=(2, 0))
    tk.Checkbutton(frame_sell_cond, text="當日跌幅>3%", variable=sell_drop_var).grid(row=2, column=0, sticky="w", pady=(2, 0))
    tk.Checkbutton(frame_sell_cond, text="KD高檔反轉", variable=sell_kd_rev_var).grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(2, 0))

    # === 移動停損 ===
    tk.Label(controls, text="移動停損：")\
        .grid(row=4, column=0, sticky="w", padx=(0, 5), pady=(5, 0))

    frame_trailing = tk.Frame(controls)
    frame_trailing.grid(row=4, column=1, sticky="w", pady=(5, 0))

    trailing_stop_var = tk.BooleanVar(value=False)
    chk_trailing = tk.Checkbutton(frame_trailing, text="啟用，跌幅達", variable=trailing_stop_var)
    chk_trailing.grid(row=0, column=0, sticky="w")

    entry_trailing_pct = tk.Entry(frame_trailing, width=6, state="disabled")
    entry_trailing_pct.insert(0, "5")
    entry_trailing_pct.grid(row=0, column=1, sticky="w", padx=(5, 0))
    tk.Label(frame_trailing, text="% 即賣出").grid(row=0, column=2, sticky="w", padx=(5, 0))

    def update_trailing_state():
        if trailing_stop_var.get():
            entry_trailing_pct.config(state="normal")
        else:
            entry_trailing_pct.config(state="disabled")

    trailing_stop_var.trace_add("write", lambda *args: update_trailing_state())
    update_trailing_state()

    # === 移動停利 ===
    tk.Label(controls, text="移動停利：")\
        .grid(row=5, column=0, sticky="w", padx=(0, 5), pady=(5, 0))

    frame_trailing_profit = tk.Frame(controls)
    frame_trailing_profit.grid(row=5, column=1, sticky="w", pady=(5, 0))

    trailing_profit_var = tk.BooleanVar(value=False)
    chk_trailing_profit = tk.Checkbutton(frame_trailing_profit, text="啟用，漲幅達", variable=trailing_profit_var)
    chk_trailing_profit.grid(row=0, column=0, sticky="w")

    entry_trailing_profit_pct = tk.Entry(frame_trailing_profit, width=6, state="disabled")
    entry_trailing_profit_pct.insert(0, "5")
    entry_trailing_profit_pct.grid(row=0, column=1, sticky="w", padx=(5, 0))
    tk.Label(frame_trailing_profit, text="% 即賣出").grid(row=0, column=2, sticky="w", padx=(5, 0))

    def update_trailing_profit_state():
        if trailing_profit_var.get():
            entry_trailing_profit_pct.config(state="normal")
        else:
            entry_trailing_profit_pct.config(state="disabled")

    trailing_profit_var.trace_add("write", lambda *args: update_trailing_profit_state())
    update_trailing_profit_state()

    # === 買進模式 ===
    tk.Label(controls, text="買進模式：")\
        .grid(row=6, column=0, sticky="w", padx=(0, 5), pady=(5, 0))

    mode_var = tk.StringVar(value="amount")

    frame_mode = tk.Frame(controls)
    frame_mode.grid(row=6, column=1, sticky="w", pady=(5, 0))

    # === 買進數值輸入 ===
    frame_input = tk.Frame(controls)
    frame_input.grid(row=7, column=1, sticky="we", pady=(2, 0))

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

    tk.Radiobutton(
        frame_mode,
        text="固定股數",
        variable=mode_var,
        value="shares",
        command=update_input_fields,
    ).pack(side="left")
    tk.Radiobutton(
        frame_mode,
        text="固定金額",
        variable=mode_var,
        value="amount",
        command=update_input_fields,
    ).pack(side="left")

    update_input_fields()

    # === 賣出模式（停利 / 停損） ===
    tk.Label(controls, text="賣出模式：")\
        .grid(row=8, column=0, sticky="w", padx=(0, 5), pady=(5, 0))

    sell_mode_var = tk.StringVar(value="shares")

    frame_sell_mode = tk.Frame(controls)
    frame_sell_mode.grid(row=8, column=1, sticky="w", pady=(5, 0))

    frame_sell_input = tk.Frame(controls)
    frame_sell_input.grid(row=9, column=1, sticky="we", pady=(2, 0))

    label_sell_value = tk.Label(frame_sell_input, text="每次賣出股數：")
    label_sell_value.pack(anchor="w")

    entry_sell_shares = tk.Entry(frame_sell_input)
    entry_sell_amount = tk.Entry(frame_sell_input)
    entry_sell_amount.insert(0, str(SELL_AMOUNT_PER_TRADE))
    entry_sell_shares.insert(0, str(SELL_SHARES_PER_TRADE))
    entry_sell_shares.pack(fill="x")

    def update_sell_input_fields():
        sell_mode = sell_mode_var.get()
        if sell_mode == "shares":
            label_sell_value.config(text="每次賣出股數：")
            entry_sell_amount.pack_forget()
            if not entry_sell_shares.winfo_ismapped():
                entry_sell_shares.pack(fill="x")
        else:
            label_sell_value.config(text="每次賣出金額：")
            entry_sell_shares.pack_forget()
            if not entry_sell_amount.winfo_ismapped():
                entry_sell_amount.pack(fill="x")

    tk.Radiobutton(
        frame_sell_mode,
        text="固定股數",
        variable=sell_mode_var,
        value="shares",
        command=update_sell_input_fields,
    ).pack(side="left")
    tk.Radiobutton(
        frame_sell_mode,
        text="固定金額",
        variable=sell_mode_var,
        value="amount",
        command=update_sell_input_fields,
    ).pack(side="left")

    update_sell_input_fields()

    # === 初始資金池 ===
    tk.Label(controls, text="初始資金池：")\
        .grid(row=10, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
    entry_capital = tk.Entry(controls)
    entry_capital.insert(0, "100000")
    entry_capital.grid(row=10, column=1, sticky="we", pady=(5, 0))

    def on_plot():
        try:
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("未選擇", "請選擇一檔股票")
                return
            idx = sel[0]
            info = file_list[idx]

            years = year_var.get()
            buy_mode = mode_var.get()
            sell_mode = sell_mode_var.get()

            # 回測開始日設定
            start_mode = start_mode_var.get()
            start_date = None
            if start_mode == "custom":
                start_date_str = entry_start_date.get().strip()
                if not start_date_str:
                    messagebox.showerror("輸入錯誤", "請輸入自訂起始日（YYYY-MM-DD）")
                    return

                start_date = pd.to_datetime(start_date_str, format="%Y-%m-%d", errors="coerce")
                if pd.isna(start_date):
                    messagebox.showerror("輸入錯誤", "日期格式需為 YYYY-MM-DD，例如 2024-01-15")
                    return

            # 買進模式相關數字
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

            # 賣出模式相關數字
            sell_shares_per_trade = None
            sell_amount_per_trade = None

            if sell_mode == "shares":
                val = entry_sell_shares.get().strip()
                if not val:
                    messagebox.showerror("輸入錯誤", "請輸入每次賣出股數")
                    return
                try:
                    sell_shares_per_trade = int(val)
                except ValueError:
                    messagebox.showerror("輸入錯誤", "賣出股數必須是正整數")
                    return
                if sell_shares_per_trade <= 0:
                    messagebox.showerror("輸入錯誤", "賣出股數必須大於 0")
                    return
            else:
                val = entry_sell_amount.get().strip()
                if not val:
                    messagebox.showerror("輸入錯誤", "請輸入每次賣出金額")
                    return
                try:
                    sell_amount_per_trade = float(val)
                except ValueError:
                    messagebox.showerror("輸入錯誤", "賣出金額必須是數字")
                    return
                if sell_amount_per_trade <= 0:
                    messagebox.showerror("輸入錯誤", "賣出金額必須大於 0")
                    return

            # 觸發條件參數
            trigger_periodic = trigger_periodic_var.get()
            trigger_kd = trigger_kd_var.get()

            weeks_interval = None
            if trigger_periodic:
                w_val = entry_weeks.get().strip()
                if not w_val:
                    messagebox.showerror("輸入錯誤", "請輸入定期買入的週期（周）")
                    return
                try:
                    weeks_interval = int(w_val)
                except ValueError:
                    messagebox.showerror("輸入錯誤", "週期必須是整數")
                    return
                if weeks_interval <= 0:
                    messagebox.showerror("輸入錯誤", "週期必須大於 0")
                    return

            oversold_lookback = None
            if trigger_kd:
                n_val = entry_N.get().strip()
                if not n_val:
                    messagebox.showerror("輸入錯誤", "請輸入 N（超賣回看天數）")
                    return
                try:
                    oversold_lookback = int(n_val)
                except ValueError:
                    messagebox.showerror("輸入錯誤", "N 必須是整數（可為 0、1、2...）")
                    return
                if oversold_lookback < 0:
                    messagebox.showerror("輸入錯誤", "N 不可為負數")
                    return

            if (not trigger_periodic) and (not trigger_kd):
                messagebox.showerror("觸發條件錯誤", "請至少勾選一種觸發條件")
                return

            try:
                initial_capital = float(entry_capital.get().strip())
            except ValueError:
                messagebox.showerror("輸入錯誤", "初始資金池必須是數字")
                return
            if initial_capital <= 0:
                messagebox.showerror("輸入錯誤", "初始資金池必須大於 0")
                return

            trailing_stop_enabled = trailing_stop_var.get()
            trailing_stop_pct = None
            if trailing_stop_enabled:
                pct_val = entry_trailing_pct.get().strip()
                if not pct_val:
                    messagebox.showerror("輸入錯誤", "請輸入移動停損百分比")
                    return
                try:
                    trailing_stop_pct = float(pct_val)
                except ValueError:
                    messagebox.showerror("輸入錯誤", "移動停損百分比必須是數字")
                    return
                if trailing_stop_pct <= 0:
                    messagebox.showerror("輸入錯誤", "移動停損百分比必須大於 0")
                    return

            trailing_profit_enabled = trailing_profit_var.get()
            trailing_profit_pct = None
            if trailing_profit_enabled:
                profit_pct_val = entry_trailing_profit_pct.get().strip()
                if not profit_pct_val:
                    messagebox.showerror("輸入錯誤", "請輸入移動停利百分比")
                    return
                try:
                    trailing_profit_pct = float(profit_pct_val)
                except ValueError:
                    messagebox.showerror("輸入錯誤", "移動停利百分比必須是數字")
                    return
                if trailing_profit_pct <= 0:
                    messagebox.showerror("輸入錯誤", "移動停利百分比必須大於 0")
                    return

            sell_conditions = {
                "ma5": sell_ma5_var.get(),
                "ma20": sell_ma20_var.get(),
                "vol_prev_day": sell_vol_prev_var.get(),
                "vol_avg5": sell_vol_avg5_var.get(),
                "drop_3pct": sell_drop_var.get(),
                "kd_reversal": sell_kd_rev_var.get(),
            }

            plot_stock(
                info,
                stock_map,
                years,
                buy_mode,
                shares_per_trade,
                amount_per_trade,
                trigger_periodic,
                sell_mode,
                sell_shares_per_trade,
                sell_amount_per_trade,
                weeks_interval,
                trigger_kd,
                oversold_lookback,
                start_mode,
                start_date,
                initial_capital,
                sell_conditions,
                trailing_stop_enabled,
                trailing_stop_pct,
                trailing_profit_enabled,
                trailing_profit_pct,
            )
        except Exception:
            log_path = write_error_log("on_plot 未捕捉例外（backtest v0.24）")
            messagebox.showerror("程式錯誤", f"發生未預期錯誤，詳細資訊已寫入\n{log_path}")

    tk.Button(root, text="繪圖＆回測", font=("Arial", 12),
              command=on_plot).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error_log_path = write_error_log("未捕捉例外（backtest v0.24）")
        print(f"發生未預期錯誤，詳細訊息已寫入 {error_log_path}")