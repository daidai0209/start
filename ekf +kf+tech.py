import asyncio
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import scrolledtext, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque, Counter
import math
import threading
import sys
import requests
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.font_manager as fm
import matplotlib.patches
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 系統與視覺參數
# ==========================================

# --- [新增] 介面縮放係數 (修改這個數字可以一次調整全部大小) ---
UI_SCALE = 1.3  

# 計算放大後的字體大小
FS_TITLE = int(16 * UI_SCALE)      # 標題
FS_HUGE = int(28 * UI_SCALE)       # 特大字 (目前位置)
FS_LARGE = int(16 * UI_SCALE)      # 大字 (狀態)
FS_MEDIUM = int(12 * UI_SCALE)     # 中字 (描述)
FS_SMALL = int(10 * UI_SCALE)      # 小字 (座標)
FS_LOG = int(10 * 1.3)             # Log 字體 (獨立微調)
FS_CHART_LABEL = 12                # 圖表內標籤
FS_CHART_AXIS = 11                 # 圖表座標軸

# TCP 連線設定
TCP_IP = '192.168.1.133'
TCP_PORT = 5002

# Telegram 設定
TELEGRAM_BOT_TOKEN = "8318610176:AAHw7PokTmFs0ZY6CirY2PNUP_9eAT9jCdo"
TELEGRAM_CHAT_ID = "8395718253"
NOTIFICATION_COOLDOWN = 15 

# 房間尺寸
ROOM_X_MIN, ROOM_X_MAX = 0.00, 3.50
ROOM_Y_MIN, ROOM_Y_MAX = 0.00, 3.10

# 警報邊界 (外推)
ALERT_MARGIN = 0.25
ALERT_X_MIN = ROOM_X_MIN - ALERT_MARGIN
ALERT_X_MAX = ROOM_X_MAX + ALERT_MARGIN
ALERT_Y_MIN = ROOM_Y_MIN - ALERT_MARGIN
ALERT_Y_MAX = ROOM_Y_MAX + ALERT_MARGIN

# 系統全域變數
SYSTEM_START_TIME = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
CURRENT_INTERVAL_START = datetime.datetime.now()
REPORT_INTERVAL_SECONDS = 600
last_notification_time = 0
ai_report_buffer = [] 
data_buffer = deque(maxlen=7)

is_first_point = True

# 資料儲存
history_data = {"raw_x": [], "raw_y": [], "ekf_x": [], "ekf_y": [], "time": []}

# 區域定義
ZONES = {
    "書桌區": {"x_min": 0, "x_max": 2.1, "y_min": 1.6, "y_max": 3.1, "desc": "專注工作/學習", "color": "#007bff"},
    "床":     {"x_min": 2.4, "x_max": 3.5, "y_min": 1.0, "y_max": 3.1, "desc": "休息/睡眠", "color": "#e83e8c"},
    "沙發區":   {"x_min": 0, "x_max": 2.1, "y_min": 0, "y_max": 0.65, "desc": "休閒娛樂", "color": "#28a745"},
    "門口":   {"x_min": 0.0, "x_max": 0.5, "y_min": 1.1, "y_max": 1.5, "desc": "進出/移動", "color": "#ffc107"},
}

# ==========================================
# 1. 核心演算法 (EKF)
# ==========================================

class ExtendedKalmanFilter:
    def __init__(self, dt=0.1):
        self.x = np.array([0.0, 0.0, 0.0, 0.0])
        self.P = np.eye(4) * 0.1
        self.dt = dt
        self.Q = np.diag([0.1, 0.1, 2.0, 1.0]) 
        self.R = np.eye(2) * 0.1  

    def update_dt(self, dt): 
        self.dt = dt

    def predict(self):
        x, y, theta, v = self.x
        dt = self.dt
        if abs(v) < 0.1:
            new_x, new_y, new_theta, new_v = x, y, theta, 0
        else:
            new_x = x + v * np.cos(theta) * dt
            new_y = y + v * np.sin(theta) * dt
            new_theta = theta
            new_v = v
        self.x = np.array([new_x, new_y, new_theta, new_v])
        F = np.eye(4)
        if abs(v) >= 0.1:
            F[0, 2] = -v * np.sin(theta) * dt
            F[0, 3] = np.cos(theta) * dt
            F[1, 2] = v * np.cos(theta) * dt
            F[1, 3] = np.sin(theta) * dt
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q

    def update(self, z):
        hx = np.array([self.x[0], self.x[1]])
        y = z - hx
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        try: K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        except: K = np.zeros((4, 2))
        self.x = self.x + np.dot(K, y)
        self.x[2] = (self.x[2] + np.pi) % (2 * np.pi) - np.pi
        self.P = np.dot((np.eye(4) - np.dot(K, H)), self.P)
        return self.x[:2]

ekf_filter = ExtendedKalmanFilter()

# ==========================================
# 2. 工具函式
# ==========================================

def send_telegram_message(message):
    global last_notification_time
    current_time = time.time()
    if "📊" in message or "🆘" in message or (current_time - last_notification_time >= NOTIFICATION_COOLDOWN):
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            threading.Thread(target=lambda: requests.post(url, data=data, timeout=5)).start()
            if "📊" not in message: last_notification_time = current_time
        except: pass

def get_current_zone(x, y):
    if x < ALERT_X_MIN or x > ALERT_X_MAX or y < ALERT_Y_MIN or y > ALERT_Y_MAX:
        return "超出範圍", "⚠️"
    for name, b in ZONES.items():
        if b["x_min"] <= x <= b["x_max"] and b["y_min"] <= y <= b["y_max"]:
            return name, b["desc"]
    return "走道區域", "移動中"

def save_report_to_txt(content):
    folder_name = "AI_Reports"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = f"{folder_name}/Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return filename
    except Exception as e:
        print(f"存檔失敗: {e}")
        return None

# ==========================================
# 3. 介面與繪圖邏輯
# ==========================================

app = ttk.Window(themename="cosmo") 
app.title(f"UWB 智慧守護系統 - 監測中心 (啟用: {SYSTEM_START_TIME})")
app.geometry("1400x900") # 稍微加大視窗以容納大字體

# 設定全域字體樣式 (讓按鈕等元件也變大)
style = ttk.Style()
style.configure('.', font=('Microsoft YaHei UI', FS_MEDIUM))
style.configure('TButton', font=('Microsoft YaHei UI', FS_MEDIUM, 'bold'))

try:
    font_prop = fm.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    font_name = font_prop.get_name()
except:
    font_prop = None
    font_name = "Arial"

left_panel = ttk.Frame(app, padding=10)
left_panel.pack(side=LEFT, fill=BOTH, expand=True)
right_panel = ttk.Frame(app, padding=15)
right_panel.pack(side=RIGHT, fill=Y)

ttk.Label(left_panel, text="📍 即時空間定位軌跡", font=(font_name, FS_TITLE, "bold"), bootstyle="primary").pack(anchor=W, pady=(0, 10))

plt.style.use('default') 
fig = Figure(figsize=(8, 6), dpi=100)
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111)
ax.set_facecolor('white')

canvas = FigureCanvasTkAgg(fig, master=left_panel)
canvas.get_tk_widget().pack(fill=BOTH, expand=True)

# 繪圖物件
line_traj, = ax.plot([], [], '-', color='#0056b3', linewidth=2, alpha=0.8, label='移動軌跡', zorder=3)
pt_current, = ax.plot([], [], '*', color='red', markersize=22, markeredgecolor='black', markeredgewidth=1, label='當前位置', zorder=5) # 星星也放大了

def init_map_visuals():
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(-0.5, 3.6)
    
    # 放大圖表座標軸標籤
    ax.set_xlabel("X (Meters)", color='black', fontsize=FS_CHART_AXIS)
    ax.set_ylabel("Y (Meters)", color='black', fontsize=FS_CHART_AXIS)
    ax.tick_params(axis='both', which='major', labelsize=FS_CHART_AXIS)
    
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black') 
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')

    try:
        img = mpimg.imread('floor_plan.png')
        ax.imshow(img, extent=[ROOM_X_MIN, ROOM_X_MAX, ROOM_Y_MIN, ROOM_Y_MAX], alpha=0.5, zorder=0)
        ax.grid(False)
    except:
        ax.grid(True, linestyle=':', color='gray', alpha=0.5)

    room_border = matplotlib.patches.Rectangle(
        (ROOM_X_MIN, ROOM_Y_MIN), 
        ROOM_X_MAX - ROOM_X_MIN, 
        ROOM_Y_MAX - ROOM_Y_MIN,
        linewidth=3,       
        edgecolor='black', 
        facecolor='none',  
        zorder=4           
    )
    ax.add_patch(room_border)
    
    for name, z in ZONES.items():
        rect = matplotlib.patches.Rectangle(
            (z["x_min"], z["y_min"]), z["x_max"]-z["x_min"], z["y_max"]-z["y_min"],
            linewidth=2, edgecolor=z["color"], facecolor='none', linestyle='--', zorder=1
        )
        ax.add_patch(rect)
        # 放大圖表內的區域文字
        ax.text(
            (z["x_min"]+z["x_max"])/2, (z["y_min"]+z["y_max"])/2, name,
            color=z["color"], fontsize=FS_CHART_LABEL, ha='center', va='center', fontproperties=font_prop, fontweight='bold', alpha=0.9
        )

    alert_rect = matplotlib.patches.Rectangle(
        (ALERT_X_MIN, ALERT_Y_MIN), ALERT_X_MAX-ALERT_X_MIN, ALERT_Y_MAX-ALERT_Y_MIN,
        linewidth=1, edgecolor='red', facecolor='none', linestyle=':', zorder=1
    )
    ax.add_patch(alert_rect)

init_map_visuals()

# 儀表板
frame_status = ttk.Labelframe(right_panel, text=" 🛡️ 系統狀態 ", bootstyle="primary", padding=15)
frame_status.pack(fill=X, pady=10)

# 狀態圖示放大
lbl_status_icon = ttk.Label(frame_status, text="🟢", font=("Segoe UI Emoji", int(32 * UI_SCALE)))
lbl_status_icon.pack(side=LEFT, padx=10)
# 狀態文字放大
lbl_status_main = ttk.Label(frame_status, text="監測中 - 正常", font=(font_name, FS_LARGE, "bold"), bootstyle="success")
lbl_status_main.pack(side=LEFT, padx=5)

frame_coords = ttk.Frame(frame_status)
frame_coords.pack(side=RIGHT, padx=10)
# 座標文字放大
lbl_coords = ttk.Label(frame_coords, text="X: 0.00\nY: 0.00", font=("Consolas", FS_SMALL), justify=RIGHT)
lbl_coords.pack()

frame_zone = ttk.Labelframe(right_panel, text=" 🏠 目前位置 ", bootstyle="info", padding=15)
frame_zone.pack(fill=X, pady=10)

# 區域大標題放大
lbl_zone_name = ttk.Label(frame_zone, text="偵測中...", font=(font_name, FS_HUGE, "bold"), bootstyle="info")
lbl_zone_name.pack(anchor=CENTER)
# 區域描述放大
lbl_zone_desc = ttk.Label(frame_zone, text="---", font=(font_name, FS_MEDIUM), bootstyle="secondary")
lbl_zone_desc.pack(anchor=CENTER, pady=(5, 0))

frame_log = ttk.Labelframe(right_panel, text=" 🤖 AI行為紀錄 ", bootstyle="secondary", padding=10)
frame_log.pack(fill=BOTH, expand=True, pady=10)

# Log 字體放大
txt_log = scrolledtext.ScrolledText(frame_log, height=10, font=("Consolas", FS_LOG), bg="white", fg="black", insertbackground="black")
txt_log.pack(fill=BOTH, expand=True)

def log_msg(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    txt_log.insert(tk.END, f"[{ts}] {msg}\n")
    txt_log.see(tk.END)

btn_frame = ttk.Frame(right_panel)
btn_frame.pack(side=BOTTOM, fill=X, pady=10)
# 按鈕已透過 style.configure 放大
ttk.Button(btn_frame, text="匯出報表 (.txt)", bootstyle="outline-primary", command=lambda: generate_ai_report(force=True)).pack(side=LEFT, expand=True, fill=X, padx=5)
ttk.Button(btn_frame, text="關閉系統", bootstyle="danger", command=app.destroy).pack(side=RIGHT, expand=True, fill=X, padx=5)

# ==========================================
# 4. 系統邏輯
# ==========================================

traj_x, traj_y = deque(), deque()
# ----------------------------------------

raw_x, raw_y = deque(maxlen=50), deque(maxlen=50) 
prev_time = None

def update_ui(ekf_x, ekf_y, raw_pt_x, raw_pt_y, current_time):
    global prev_time, is_first_point
    
    dt = current_time - prev_time if prev_time else 0.1
    dt = max(0.01, min(0.3, dt))
    prev_time = current_time
    ekf_filter.update_dt(dt)
    
    if is_first_point:
        ekf_filter.x = np.array([ekf_x, ekf_y, 0.0, 0.0])
        est_x, est_y = ekf_x, ekf_y
        is_first_point = False
    else:
        ekf_filter.predict()
        est_x, est_y = ekf_filter.update(np.array([ekf_x, ekf_y]))
        
    traj_x.append(est_x)
    traj_y.append(est_y)
    raw_x.append(raw_pt_x)
    raw_y.append(raw_pt_y)
    
    history_data["raw_x"].append(raw_pt_x)
    history_data["raw_y"].append(raw_pt_y)
    history_data["ekf_x"].append(est_x)
    history_data["ekf_y"].append(est_y)
    history_data["time"].append(current_time)
    
    zone_name, zone_desc = get_current_zone(est_x, est_y)
    ai_report_buffer.append((current_time, zone_name))
    
    line_traj.set_data(traj_x, traj_y)
    
    # 這裡只更新紅星位置，不繪製綠點
    pt_current.set_data([est_x], [est_y])
    canvas.draw_idle()
    
    lbl_coords.config(text=f"X: {est_x:.2f}\nY: {est_y:.2f}")
    lbl_zone_name.config(text=zone_name)
    lbl_zone_desc.config(text=zone_desc)
    
    is_out_of_bounds = (zone_name == "超出範圍")
    if is_out_of_bounds:
        lbl_status_icon.config(text="🆘", bootstyle="danger")
        lbl_status_main.config(text="異常 - 超出範圍", bootstyle="danger")
        lbl_zone_name.config(bootstyle="danger")
        if "🆘" not in lbl_status_main.cget("text"): 
            send_telegram_message(f"🆘緊急警報：目標超出安全範圍！位置({est_x:.1f}, {est_y:.1f})")
    else:
        lbl_status_icon.config(text="🟢", bootstyle="success")
        lbl_status_main.config(text="監測中 - 正常", bootstyle="success")
        lbl_zone_name.config(bootstyle="info")
 
async def tcp_reader():
    print("⏳System Starting...")
    log_msg("系統啟用，等待TCP連線...")
    
    while True:
        try:
            r, w = await asyncio.open_connection(TCP_IP, TCP_PORT)
            print("✅Connected!")
            log_msg("✅UWB基地台連線完成")
            lbl_status_main.config(text="連線完成 - 接收數據", bootstyle="success")
            
            while True:
                data = await r.readline()
                if not data: break
                try:
                    parts = data.decode().strip().split(',')
                    x = float(parts[0].split(':')[1])
                    y = float(parts[1].split(':')[1])
                    if not (math.isnan(x) or math.isnan(y)):
                        data_buffer.append((x, y))
                except: continue
                
                if len(data_buffer) == data_buffer.maxlen:
                    avg_x = sum(d[0] for d in data_buffer)/len(data_buffer)
                    avg_y = sum(d[1] for d in data_buffer)/len(data_buffer)
                    app.after(0, update_ui, avg_x, avg_y, avg_x, avg_y, time.time())
                    data_buffer.clear()
        except Exception as e:
            lbl_status_main.config(text="斷線 - 重連中...", bootstyle="warning")
            await asyncio.sleep(3)

def generate_ai_report(force=False):
    global ai_report_buffer, CURRENT_INTERVAL_START
    
    if not ai_report_buffer and not force:
        app.after(REPORT_INTERVAL_SECONDS * 1000, generate_ai_report)
        return

    interval_end = datetime.datetime.now()
    interval_start_str = CURRENT_INTERVAL_START.strftime("%H:%M")
    interval_end_str = interval_end.strftime("%H:%M")
    
    if not ai_report_buffer:
        msg = "📊 [AI 報表] 區間內無有效數據。"
    else:
        total = len(ai_report_buffer)
        zones = [z for _, z in ai_report_buffer]
        counts = Counter(zones)
        top_zone, cnt = counts.most_common(1)[0]
        pct = (cnt / total) * 100
        now_str = interval_end.strftime("%Y-%m-%d %H:%M")
        
        msg = f"📊 [AI 智慧報表] {now_str}\n"
        msg += f"🕒系統啟用: {SYSTEM_START_TIME}\n"
        msg += f"⏱️統計區間: {interval_start_str} ~ {interval_end_str}\n"
        msg += f"--------------------------------\n"
        msg += f"🏠主要活動: {top_zone} ({pct:.1f}%)\n"
        msg += f"📈行為分佈: " + ", ".join([f"{k}:{v/total*100:.0f}%" for k,v in counts.items()]) + "\n"
        
        if top_zone == "床": desc = "睡眠品質良好，長時間休息。"
        elif top_zone == "書桌區": desc = "專注度高，長時間處於工作區。"
        elif top_zone == "走道區域": desc = "活動力旺盛，頻繁移動中。"
        else: desc = "活動模式正常。"
        msg += f"💡AI判讀: {desc}"

    log_msg(msg)
    send_telegram_message(msg)
    
    saved_file = save_report_to_txt(msg)
    if saved_file:
        log_msg(f"💾報表已儲存至: {saved_file}")
    
    ai_report_buffer = []
    CURRENT_INTERVAL_START = datetime.datetime.now()
    if not force:
        app.after(REPORT_INTERVAL_SECONDS * 1000, generate_ai_report)

def on_closing():
    if history_data["raw_x"]:
        df = pd.DataFrame(history_data)
        fname = f"History_{time.strftime('%Y%m%d_%H%M%S')}.xlsx"
        try:
            df.to_excel(fname, index=False)
            print(f"Data saved to {fname}")
        except: pass
    app.destroy()

if __name__ == "__main__":
    app.protocol("WM_DELETE_WINDOW", on_closing)
    threading.Thread(target=lambda: asyncio.run(tcp_reader()), daemon=True).start()
    app.after(REPORT_INTERVAL_SECONDS * 1000, generate_ai_report)
    log_msg("監測系統初始化完成")
    app.mainloop()
