import yfinance as yf
import pandas as pd
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os


def download_data():
    try:
        btn_download.config(state='disabled')
        lbl_status.config(text="下載中，請稍候...")

        stock_ids = []
        manual_input = entry_stock_ids.get().strip()
        if manual_input:
            stock_ids += [s.strip() for s in manual_input.split(',') if s.strip()]

        if use_csv.get():
            stock_file = filedialog.askopenfilename(title="選擇 stock_id.csv 檔案")
            if stock_file:
                stock_list = pd.read_csv(stock_file)
                stock_list.columns = ['stock_id', 'name']
                stock_ids += list(stock_list['stock_id'].astype(str))
            else:
                messagebox.showwarning("警告", "未選擇 CSV 檔，將只處理手動輸入股票代號")

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
                full_id = sid + '.TW'
            else:
                # 美股：直接使用代號（轉大寫），不加後綴
                full_id = sid.upper()

            lbl_status.config(text=f"處理中：{stock_id} ({selected_market})")
            root.update()

            try:
                data = yf.Ticker(full_id)
                df = data.history(period=selected_period)
                if df.empty:
                    print(f"下載 {stock_id} ({full_id}) 失敗：無資料")
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
entry_stock_ids = tk.Entry(root, font=('Arial', 11), width=50)
entry_stock_ids.pack(pady=5)

# 是否使用 CSV
use_csv = tk.BooleanVar()
chk_csv = tk.Checkbutton(root, text="另外載入 stock_id.csv 檔案", variable=use_csv, font=('Arial', 10))
chk_csv.pack()

# ★ 市場選擇（台股 / 美股）
tk.Label(root, text="選擇市場：", font=('Arial', 11)).pack(pady=5)
market_var = tk.StringVar(value="TW")  # 預設台股
market_menu = tk.OptionMenu(root, market_var, "TW", "US")
market_menu.config(font=('Arial', 11), width=10)
market_menu.pack()

# 資料期間
tk.Label(root, text="選擇資料期間：", font=('Arial', 11)).pack(pady=5)
period_var = tk.StringVar(value="1y")
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
