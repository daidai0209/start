# Clear Wave - UWB Indoor Positioning & Care System

## 專案簡介 (Project Overview)
本專案為一套「主動式、高精度、邊緣化」的智慧照護系統，專為獨居長者與失智症患者設計。透過結合超寬頻 (UWB) 技術與擴展卡爾曼濾波 (EKF) 演算法，系統能在本地端即時修正非線性軌跡，並自動識別「往返 (Pacing)」與「繞圈 (Lapping)」等異常行為，實現公分級的精準守護。

## 核心技術 (Core Technologies)
* **硬體架構 (Hardware):** Qorvo Decawave DWM1001 模組、Raspberry Pi 3
* **演算法 (Algorithm):** 擴展卡爾曼濾波 (Extended Kalman Filter, EKF)、航位推算 (Dead Reckoning)
* **軟體與介面 (Software):** Python, Tkinter (GUI)
* **警報系統 (Alert System):** Telegram Bot API

## 輕量化邊緣運算 (Lightweight Edge AI)
為克服穿戴裝置功耗與邊緣閘道器算力限制，本系統捨棄龐大的深度學習模型，改採精簡的數學模型 (EKF) 結合規則式特徵提取。這不僅大幅降低 CPU 負載，更達成低延遲、高隱私安全性的邊緣智慧。
