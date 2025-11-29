import os
import pandas as pd
import graphic

# === 路徑設定 ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(BASE_DIR, "historical data")
STOCK_MAP_PATH = os.path.join(BASE_DIR, "content", "stock_id.csv")

# === 固定回測設定 ===
TARGET_STOCK_ID = "2330"   # 只回測 2330
BACKTEST_DAYS = 365        # 回測期間：一年
BUY_MODE = "amount"        # 僅保留固定金額模式
AMOUNT_PER_TRADE = 1000    # 每次買入金額


def load_stock_map():
    """讀取股票代碼與名稱對照表。"""
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
    """尋找目標股票的 CSV 檔，優先 1y，其次 max。"""
    candidates = []
    if not os.path.isdir(HIST_DIR):
        return candidates

    for fname in os.listdir(HIST_DIR):
        if not fname.lower().endswith(".csv"):
            continue

        base = fname[:-4]
        parts = base.split()
        stock_id = parts[0]
        if stock_id != TARGET_STOCK_ID:
            continue

        period = " ".join(parts[1:]) if len(parts) >= 2 else ""
        candidates.append({
            "stock_id": stock_id,
            "period": period,
            "filename": fname,
            "path": os.path.join(HIST_DIR, fname),
        })

    # 先找標題含 "1y" 的，找不到就 fallback 第一個
    one_year = [c for c in candidates if "1y" in c.get("period", "").lower()]
    if one_year:
        return [one_year[0]]
    return candidates[:1]


def read_price_csv(csv_path: str) -> pd.DataFrame:
    """讀取股價 CSV 並處理欄位與日期。"""
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="utf-8")

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"缺少必要欄位：{missing}")

    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def filter_last_year(df: pd.DataFrame) -> pd.DataFrame:
    """保留最近一年資料。"""
    if df.empty:
        return df

    end_date = df["Date"].max()
    start_date = end_date - pd.Timedelta(days=BACKTEST_DAYS)
    return df[df["Date"] >= start_date].copy()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """增加繪圖用均線與字串日期欄位。"""
    df = df.copy()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df


def apply_fixed_conditions(df: pd.DataFrame) -> pd.Series:
    """計算固定買進訊號：
    1. 收盤價高於開盤價的 103%。
    2. 當日成交量大於前五日平均成交量的兩倍。
    """
    price_cond = df["Close"] > df["Open"] * 1.02
    volume_cond = df["Volume"] > 2 * df["Volume"].shift(1).rolling(5).mean()
    return price_cond & volume_cond


def simulate_trades(df: pd.DataFrame) -> pd.DataFrame:
    """根據固定買進訊號模擬交易並回傳交易明細。"""
    buy_signal = apply_fixed_conditions(df)
    trades = []
    total_shares = 0

    final_close = df["Close"].iloc[-1]

    for _, row in df[buy_signal].iterrows():
        close = row["Close"]
        shares = int(AMOUNT_PER_TRADE // close)
        if shares <= 0:
            continue

        total_shares += shares
        cost = close * shares

        trades.append({
            "日期": row["Date"].strftime("%Y-%m-%d"),
            "買入價": close,
            "股數": shares,
            "成本": cost,
        })

    trade_df = pd.DataFrame(trades)

    if len(trade_df) > 0:
        trade_df["期末價"] = final_close
        trade_df["最終市值"] = trade_df["股數"] * final_close
        trade_df["獲利"] = trade_df["最終市值"] - trade_df["成本"]
        trade_df["報酬率%"] = trade_df["獲利"] / trade_df["成本"] * 100

    return trade_df


def run_backtest():
    files = scan_csv_files()
    if not files:
        raise FileNotFoundError(f"請先下載 {TARGET_STOCK_ID} 的 CSV 至：{HIST_DIR}")

    stock_map = load_stock_map()
    info = files[0]

    try:
        df = read_price_csv(info["path"])
    except ValueError as exc:
        raise ValueError(f"⚠ {info['filename']} 無法處理：{exc}")

    df = filter_last_year(df)
    df = add_indicators(df)
    df["buy_signal"] = apply_fixed_conditions(df)

    trade_df = simulate_trades(df)
    stock_id = info["stock_id"]
    name = stock_map.get(stock_id) or stock_map.get(stock_id.lstrip("0")) or ""

    if len(trade_df) == 0:
        print(f"{stock_id} {name}：期間內無符合條件的訊號。")
    else:
        total_cost = trade_df["成本"].sum()
        final_value = trade_df["最終市值"].sum()
        total_profit = trade_df["獲利"].sum()
        profit_pct = total_profit / total_cost * 100 if total_cost else 0

        trade_df.insert(0, "股票代號", stock_id)
        trade_df.insert(1, "股票名稱", name)

        out_path = os.path.join(BASE_DIR, "result_fixed.xlsx")
        trade_df.to_excel(out_path, index=False)

        print(f"========== 回測 {BACKTEST_DAYS} 天 {stock_id} {name} ==========")
        print(f"每次買入金額：{AMOUNT_PER_TRADE:,.0f} 元")
        print(f"總買入次數：{len(trade_df)}")
        print(f"累積買進股數：{trade_df['股數'].sum()}")
        print(f"總成本：{total_cost:,.2f}")
        print(f"期末總市值：{final_value:,.2f}")
        sign = "+" if profit_pct >= 0 else "-"
        print(f"總報酬：{total_profit:,.2f}")
        print(f"總報酬百分比：{sign}{abs(profit_pct):.2f}%")
        print(f"已輸出交易明細至 {out_path}")
        print("===========================================\n")

    period_label = info.get("period") or "1y"
    fig = graphic.plot_backtest_figure(df, stock_id, name, period_label)
    chart_path = os.path.join(BASE_DIR, f"{stock_id}_{period_label}_fixed.html")
    fig.write_html(chart_path)
    print(f"已輸出 K 線圖至 {chart_path}")


if __name__ == "__main__":
    run_backtest()
