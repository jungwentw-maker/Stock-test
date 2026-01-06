import os
import threading
from datetime import datetime

import pandas as pd
import yfinance as yf
import tkinter as tk
from tkinter import messagebox

from graphic import plot_backtest_figure

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


def compute_kd(rsv_series: pd.Series):
    k_values = []
    d_values = []
    k_prev = 50.0
    d_prev = 50.0

    for rsv in rsv_series:
        if pd.isna(rsv):
            k_values.append(pd.NA)
            d_values.append(pd.NA)
            continue

        k_today = (2 / 3) * k_prev + (1 / 3) * rsv
        d_today = (2 / 3) * d_prev + (1 / 3) * k_today
        k_prev, d_prev = k_today, d_today

        k_values.append(k_today)
        d_values.append(d_today)

    return pd.Series(k_values), pd.Series(d_values)


def analyze_data():
    try:
        btn_analyze.config(state='disabled')
        lbl_status.config(text="分析中，請稍候...")

        stock_ids = []
        manual_input = entry_stock_ids.get().strip()
        if manual_input:
            stock_ids += [s.strip() for s in manual_input.split(',') if s.strip()]

        if not stock_ids:
            messagebox.showwarning("警告", "沒有可處理的股票代號")
            btn_analyze.config(state='normal')
            return

        save_root = os.path.join(BASE_DIR, "results")
        os.makedirs(save_root, exist_ok=True)

        selected_market = market_var.get()  # 'TW' 或 'US'
        success_list = []
        fail_list = []

        for stock_id in stock_ids:
            sid = stock_id.strip()

            # 準備不同市場需要嘗試的 ticker 後綴
            if selected_market == 'TW':
                candidate_ids = [sid + '.TW', sid + '.TWO']
            elif selected_market == 'US':
                sid_upper = sid.upper()
                candidate_ids = [sid_upper]
                if '.' not in sid_upper:
                    candidate_ids.append(sid_upper + '.US')
            else:
                candidate_ids = [sid]

            lbl_status.config(text=f"處理中：{stock_id} ({selected_market})")
            root.update()

            last_error = None
            df = None
            used_id = None

            for full_id in candidate_ids:
                try:
                    ticker = yf.Ticker(full_id)
                    df = ticker.history(period="6mo")
                    if df.empty:
                        last_error = "無資料"
                        continue

                    used_id = full_id
                    break
                except Exception as fetch_err:
                    last_error = str(fetch_err)

            if df is None or df.empty or used_id is None:
                msg = f"{stock_id} 下載失敗"
                if last_error:
                    msg += f"（{last_error}）"
                fail_list.append(msg)
                print(msg)
                continue

                df.index.name = 'Date'
                df = df.reset_index()
                date_series = pd.to_datetime(df['Date'])
                if date_series.dt.tz is not None:
                    date_series = date_series.dt.tz_convert(None)
                df['Date'] = date_series

                df['MA5'] = df['Close'].rolling(5).mean()
                df['MA20'] = df['Close'].rolling(20).mean()
                df['MA60'] = df['Close'].rolling(60).mean()

                high_9 = df['High'].rolling(9).max()
                low_9 = df['Low'].rolling(9).min()
                rsv = (df['Close'] - low_9) / (high_9 - low_9) * 100
                rsv = rsv.where((high_9 - low_9) != 0)

                k_series, d_series = compute_kd(rsv)
                df['K'] = k_series
                df['D'] = d_series

                three_months_ago = pd.Timestamp.today().normalize() - pd.DateOffset(months=3)
                df_recent = df[df['Date'] >= three_months_ago].copy()
                df_recent.sort_values('Date', inplace=True)

                stock_name = STOCK_NAME_MAP.get(sid) or STOCK_NAME_MAP.get(sid.lstrip('0')) or ''
                today_str = datetime.today().strftime('%Y%m%d')

                output_filename = f"{sid}{stock_name}{today_str}.csv"
                output_path = os.path.join(save_root, output_filename)
                df_recent.to_csv(output_path, index=False, encoding='utf-8-sig')

                success_list.append(f"{stock_id}（{used_id}）存入 {output_filename}")
                
                try:
                    fig = plot_backtest_figure(
                        df_recent,
                        stock_id=sid,
                        stock_name=stock_name,
                        period="近三個月",
                        show_raff_channels=False,
                        show=False,
                    )
                    fig.show()
                except Exception as plot_err:
                    print(f"繪圖 {stock_id} ({full_id}) 失敗：{plot_err}")

            except Exception as e:
                fail_list.append(f"分析 {stock_id} 失敗：{e}")
                print(f"分析 {stock_id} 失敗：{e}")

        if success_list and not fail_list:
            messagebox.showinfo("完成", "\n".join(success_list))
            lbl_status.config(text="完成！")
        elif success_list and fail_list:
            messagebox.showwarning(
                "部分失敗",
                "成功：\n" + "\n".join(success_list) + "\n\n失敗：\n" + "\n".join(fail_list),
            )
            lbl_status.config(text="部分完成")
        else:
            messagebox.showerror("全部失敗", "\n".join(fail_list) if fail_list else "全部失敗，無可用資料")
            lbl_status.config(text="失敗")

    except Exception as e:
        messagebox.showerror("錯誤", str(e))
    finally:
        btn_analyze.config(state='normal')


def start_analyze():
    threading.Thread(target=analyze_data).start()


# GUI 主視窗
root = tk.Tk()
root.title("股價技術指標分析（台股 / 美股）")
root.geometry("520x400")
root.resizable(False, False)

# 手動輸入股票代號
tk.Label(root, text="手動輸入股票代號（用逗號分隔）：", font=('Arial', 10)).pack(pady=5)
stock_var = tk.StringVar()
entry_stock_ids = tk.Entry(root, font=('Arial', 11), width=50, textvariable=stock_var)
entry_stock_ids.pack(pady=5)

# 顯示對應的股票名稱
lbl_stock_name = tk.Label(root, text="", font=('Arial', 10), fg="green")
lbl_stock_name.pack(pady=2)


def update_stock_name(*args):
    text = stock_var.get().strip()
    if not text:
        lbl_stock_name.config(text="")
        return

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

# 市場選擇
tk.Label(root, text="選擇市場：", font=('Arial', 11)).pack(pady=5)
market_var = tk.StringVar(value="TW")
market_menu = tk.OptionMenu(root, market_var, "TW", "US","手動輸入")
market_menu.config(font=('Arial', 11), width=10)
market_menu.pack()

# 按鈕 & 狀態
btn_analyze = tk.Button(root, text="開始分析", command=start_analyze, font=('Arial', 12))
btn_analyze.pack(pady=20)

lbl_status = tk.Label(root, text="", fg="blue")
lbl_status.pack()

root.mainloop()