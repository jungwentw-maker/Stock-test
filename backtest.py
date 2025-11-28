import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from content.indicators import add_moving_averages, compute_kd


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

    # ======== ★ 僅保留最近五年資料 ★ ========
    last_date = df["Date"].max()
    five_years_ago = last_date - pd.Timedelta(days=1826)
    df = df[df["Date"] >= five_years_ago].copy()

    # 類別軸用的字串日期（只要有資料的日期才會出現一次）
    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")

    # 移動平均線（由 indicators.py 統一計算）
    df = add_moving_averages(df, windows=(5, 20, 60, 120))

    # KD（由 indicators.py 提供的 compute_kd）
    df = compute_kd(df)


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
        template="plotly_white",

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
        ),

        # 上：價格
        yaxis=dict(
            title="價格",
            domain=[0.3, 1.0],
        ),

        # 中：成交量
        yaxis2=dict(
            title="成交量",
            domain=[0.12, 0.3],
        ),

        # 下：KD
        yaxis3=dict(
            title="KD",
            domain=[0.0, 0.1],
            showticklabels=True, # ★ 顯示 X 軸 tick labels（日期）
        ),

        hovermode="x",
        hoversubplots="axis",  # 使用同一個 x 的所有 subplot 都吃 hover / spike
        height=1250,
    )
def plot_stock(file_info, stock_name_map):
    """
    根據選到的 CSV 畫圖：
    - 上：K 線 + MA (y)
    - 中：成交量 (y2)
    - 下：KD (y3)
    再加上：
    - ★ K < 20 的買點（黑色三角形）
    - ★ 回測 result.csv 輸出
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
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df.sort_values("Date", inplace=True)

    # 類別軸
    df["DateStr"] = df["Date"].dt.strftime("%Y-%m-%d")

    # 移動平均線
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA120"] = df["Close"].rolling(120).mean()

    # KD
    df = compute_kd(df)

    # ======== ★ 回測區：K < 20 每天買入 1 股 ========
    df["buy_signal"] = df["K"] < 20

    trades = []
    total_shares = 0
    last_close = df["Close"].iloc[-1]

    for idx, row in df[df["buy_signal"]].iterrows():
        date = row["Date"]
        close = row["Close"]
        shares = 1               # ★ 先固定買 1 股（你可換成固定金額模式）
        cost = close * shares

        total_shares += shares
        trades.append({
            "日期": date.strftime("%Y-%m-%d"),
            "買入價": close,
            "股數": shares,
            "成本": cost
        })

    # 轉成 DataFrame
    trade_df = pd.DataFrame(trades)

    if len(trade_df) > 0:
        trade_df["期末價"] = last_close
        trade_df["最終市值"] = trade_df["股數"] * last_close
        trade_df["獲利"] = trade_df["最終市值"] - trade_df["成本"]
        trade_df["報酬率%"] = trade_df["獲利"] / trade_df["成本"] * 100

        # ★ result.csv 寫在 graphic.py 同資料夾
        out_path = os.path.join(BASE_DIR, "result.csv")
        trade_df.to_csv(out_path, index=False, encoding="utf-8-sig")

        print("========== 回測摘要 ==========")
        print(f"總買入次數：{len(trade_df)}")
        print(f"累積買進股數：{total_shares}")
        print(f"總成本：{trade_df['成本'].sum():,.2f}")
        print(f"最後一天收盤：{last_close}")
        print(f"最終市值：{total_shares * last_close:,.2f}")
        print(f"總報酬：{total_shares * last_close - trade_df['成本'].sum():,.2f}")
        print("================================")

    else:
        print("⚠ 無任何買入訊號（K<20）。未產生 result.csv。")

    # colore for volume
    volume_colors = [
        "red" if c >= o else "green"
        for c, o in zip(df["Close"], df["Open"])
    ]

    name = stock_name_map.get(stock_id) or stock_name_map.get(stock_id.lstrip("0")) or ""
    title_text = f"{stock_id} {name}（{period}）" if period else f"{stock_id} {name}"

    # === 建立 traces（K 線 / MA / Volume / KD） ===
    traces = []

    # K 線
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

    # MA
    traces.append(go.Scatter(x=df["DateStr"], y=df["MA5"],  mode="lines", name="MA5",  line=dict(color="orange"), xaxis="x", yaxis="y"))
    traces.append(go.Scatter(x=df["DateStr"], y=df["MA20"], mode="lines", name="MA20", line=dict(color="blue"),   xaxis="x", yaxis="y"))
    traces.append(go.Scatter(x=df["DateStr"], y=df["MA60"], mode="lines", name="MA60", line=dict(color="purple"), xaxis="x", yaxis="y"))
    traces.append(go.Scatter(x=df["DateStr"], y=df["MA120"],mode="lines", name="MA120",line=dict(color="brown"),  xaxis="x", yaxis="y"))

    # 成交量
    traces.append(go.Bar(
        x=df["DateStr"], y=df["Volume"],
        name="成交量",
        marker=dict(color=volume_colors),
        xaxis="x", yaxis="y2",
    ))

    # KD
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["K"],
        mode="lines", name="K（快）",
        line=dict(color="green"),
        xaxis="x", yaxis="y3",
    ))
    traces.append(go.Scatter(
        x=df["DateStr"], y=df["D"],
        mode="lines", name="D（慢）",
        line=dict(color="red"),
        xaxis="x", yaxis="y3",
    ))

    # ======== ★ 加上買點記號：黑色三角形 ========
    if df["buy_signal"].any():
        traces.append(go.Scatter(
            x=df[df["buy_signal"]]["DateStr"],
            y=df[df["buy_signal"]]["Close"],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="black"
            ),
            name="買點(K<20)",
            xaxis="x",
            yaxis="y",
        ))

    # === Layout ===
    layout = go.Layout(
        title=dict(text=title_text),
        template="plotly_white",
        xaxis=dict(
            type="category",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1,
            spikedash="dot",
            rangeslider=dict(visible=False),
            showticklabels=False,
        ),
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


if __name__ == "__main__":
    main()
