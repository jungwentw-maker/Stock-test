import yfinance as yf
import pandas as pd
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


STOCK_NAME_MAP = load_stock_map()


def download_data():
    try:
        btn_download.config(state='disabled')
        lbl_status.config(text="下載中，請稍候...")

        stock_ids = []
        manual_input = entry_stock_ids.get().strip()
        if manual_input:
            stock_ids += [s.strip() for s in manual_input.split(',') if s.strip()]


        if not stock_ids:
            messagebox.showwarning("警告", "沒有可處理的股票代號")
            btn_download.config(state='normal')
            return

        # 自動取得程式所在目錄
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 自動建立 historical data 子資料夾
        save_dir = os.path.join(base_dir, "historical data")
        os.makedirs(save_dir, exist_ok=True)

        selected_period = period_var.get()
        selected_market = market_var.get()  # 'TW' 或 'US'

        for stock_id in stock_ids:
            sid = stock_id.strip()

            # ★ 依市場決定 yfinance 代號
            if selected_market == 'TW':
                # 台股：數字代號 + .TW
                primary_full_id = sid + '.TW'
                fallback_full_id = sid + '.TWO'
            else:
                # 美股：直接使用代號（轉大寫），不加後綴
                primary_full_id = sid.upper()
                fallback_full_id = None

            lbl_status.config(text=f"處理中：{stock_id} ({selected_market})")
            root.update()

            try:
                df = None
                primary_error = None

                try:
                    data = yf.Ticker(primary_full_id)
                    df = data.history(period=selected_period)
                except Exception as e:
                    primary_error = e

                # 當主要後綴取得資料失敗（例外或空資料），嘗試改用 TWO
                if (df is None or df.empty) and fallback_full_id:
                    print(f"下載 {stock_id} ({primary_full_id}) 失敗，嘗試 {fallback_full_id}")
                    try:
                        data = yf.Ticker(fallback_full_id)
                        df = data.history(period=selected_period)
                    except Exception as e:
                        print(f"下載 {stock_id} ({fallback_full_id}) 失敗：{e}")
                        df = None

                if df is None or df.empty:
                    if primary_error:
                        print(f"下載 {stock_id} ({primary_full_id}) 失敗：{primary_error}")
                    print(f"下載 {stock_id} 失敗：主要與備援代碼皆無資料")
                    continue

                df.index.name = 'Date'
                df = df.reset_index()
                df['stock_id'] = stock_id
                df['market'] = selected_market

                # 產生檔名，例如：2330 1y.csv 或 AAPL 1y.csv
                file_name = f"{stock_id} {selected_period}.csv"

                # 寫入到 historical data 子資料夾
                output_path = os.path.join(save_dir, file_name)
                df.to_csv(output_path, index=False, encoding='utf-8-sig')

            except Exception as e:
                print(f"下載 {stock_id} ({full_id}) 失敗：{e}")

            time.sleep(0.8)

        messagebox.showinfo("完成", f"資料已存入：\n{save_dir}")
        lbl_status.config(text="完成！")

    except Exception as e:
        messagebox.showerror("錯誤", str(e))
    finally:
        btn_download.config(state='normal')


def start_download():
    threading.Thread(target=download_data).start()


# GUI 主視窗
root = tk.Tk()
root.title("歷史股價下載器（台股 / 美股）")
root.geometry("500x380")
root.resizable(False, False)

# 手動輸入股票代號
tk.Label(root, text="手動輸入股票代號（用逗號分隔）：", font=('Arial', 10)).pack(pady=5)
stock_var = tk.StringVar()
entry_stock_ids = tk.Entry(root, font=('Arial', 11), width=50, textvariable=stock_var)
entry_stock_ids.pack(pady=5)

# 顯示對應的股票名稱（僅顯示第一個代號）
lbl_stock_name = tk.Label(root, text="", font=('Arial', 10), fg="green")
lbl_stock_name.pack(pady=2)


def update_stock_name(*args):
    text = stock_var.get().strip()
    if not text:
        lbl_stock_name.config(text="")
        return

    # 支援多個代號：用逗號分隔，逐一查詢名稱
    parts = [s.strip() for s in text.split(",") if s.strip()]
    if not parts:
        lbl_stock_name.config(text="")
        return

    display_list = []
    for sid in parts:
        name = STOCK_NAME_MAP.get(sid) or STOCK_NAME_MAP.get(sid.lstrip("0"))
        if name:
            display_list.append(f"{sid} {name}")
        else:
            display_list.append(f"{sid} (?)")

    lbl_stock_name.config(text="股票名稱：" + "、".join(display_list))


stock_var.trace_add("write", update_stock_name)

# ★ 市場選擇（台股 / 美股）
tk.Label(root, text="選擇市場：", font=('Arial', 11)).pack(pady=5)
market_var = tk.StringVar(value="TW")  # 預設台股
market_menu = tk.OptionMenu(root, market_var, "TW", "US")
market_menu.config(font=('Arial', 11), width=10)
market_menu.pack()

# 資料期間
tk.Label(root, text="選擇資料期間：", font=('Arial', 11)).pack(pady=5)
period_var = tk.StringVar(value="max")
period_options = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
period_menu = tk.OptionMenu(root, period_var, *period_options)
period_menu.config(font=('Arial', 11), width=10)
period_menu.pack()

# 按鈕 & 狀態
btn_download = tk.Button(root, text="下載歷史資料", command=start_download, font=('Arial', 12))
btn_download.pack(pady=20)

lbl_status = tk.Label(root, text="", fg="blue")
lbl_status.pack()

root.mainloop()
