import cv2
import tkinter as tk
from tkinter import ttk, font
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
import time
import math
import random
from collections import defaultdict, deque
from ultralytics import YOLO
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH = "runs9c_30/detect/train/weights/best.pt"
CONF_THRESH = 0.40


CLASS_CONFIG = {
    "Aluminium foil":    {"color": "#C0C0C0", "bin": "Kim Loại", "emoji": "🫙"},
    "Bottle":            {"color": "#63B3ED", "bin": "Nhựa",     "emoji": "🍶"},
    "Bottle cap":        {"color": "#4299E1", "bin": "Nhựa",     "emoji": "🔘"},
    "Can":               {"color": "#A0AEC0", "bin": "Kim Loại", "emoji": "🥤"},
    "Carton":            {"color": "#F6AD55", "bin": "Giấy",     "emoji": "📦"},
    "Cup":               {"color": "#68D391", "bin": "Nhựa",     "emoji": "☕"},
    "Lid":               {"color": "#B794F4", "bin": "Nhựa",     "emoji": "🔵"},
    "Other plastic":     {"color": "#76E4F7", "bin": "Nhựa",     "emoji": "♻️"},
    "Plastic bag":       {"color": "#48BB78", "bin": "Nhựa",     "emoji": "🛍️"},
    "Plastic container": {"color": "#9F7AEA", "bin": "Nhựa",     "emoji": "🪴"},
    # –– Các tên biết dạng alternative từ các phiên bản dataset khác nhau ––
    "Aluminum foil":     {"color": "#C0C0C0", "bin": "Kim Loại", "emoji": "🫙"},
    "aluminium foil":    {"color": "#C0C0C0", "bin": "Kim Loại", "emoji": "🫙"},
    "aluminum foil":     {"color": "#C0C0C0", "bin": "Kim Loại", "emoji": "🫙"},
    "Metal can":         {"color": "#A0AEC0", "bin": "Kim Loại", "emoji": "🥤"},
    "Tin can":           {"color": "#A0AEC0", "bin": "Kim Loại", "emoji": "🥤"},
    "Cardboard":         {"color": "#F6AD55", "bin": "Giấy",     "emoji": "📦"},
    "cardboard":         {"color": "#F6AD55", "bin": "Giấy",     "emoji": "📦"},
    "Paper":             {"color": "#FBD38D", "bin": "Giấy",     "emoji": "📄"},
    "paper":             {"color": "#FBD38D", "bin": "Giấy",     "emoji": "📄"},
    "Plastic Film":      {"color": "#48BB78", "bin": "Nhựa",     "emoji": "🛍️"},
    "Styrofoam piece":   {"color": "#76E4F7", "bin": "Nhựa",     "emoji": "♻️"},
}

# Keyword → Bin fallback (nếu tên class không khớp bất kỳ cách nào)
KEYWORD_BIN = [
    (["alumin", "metal", "steel", "tin", "iron", "zinc", "can"],            "Kim Loại"),
    (["carton", "cardboard", "paper", "corrugated", "newspaper", "kraft"],  "Giấy"),
    (["plastic", "bottle", "bag", "cup", "lid", "container", "film",
      "polystyrene", "styrofoam", "foam", "nylon"],                         "Nhựa"),
]

# Màu và emoji mặc định cho từng bin
BIN_DEFAULT = {
    "Nhựa":     {"color": "#63B3ED", "emoji": "♻️"},
    "Kim Loại": {"color": "#A0AEC0", "emoji": "🥤"},
    "Giấy":     {"color": "#F6AD55", "emoji": "📦"},
    "Khác":     {"color": "#FC8181", "emoji": "❓"},
}

BIN_COLORS = {
    "Nhựa":     "#3182CE",
    "Kim Loại": "#718096",
    "Giấy":     "#DD6B20",
    "Khác":     "#E53E3E",
}

# ─── DARK THEME COLORS ───────────────────────────────────────────────────────
BG       = "#0D1117"
BG2      = "#161B22"
BG3      = "#21262D"
ACCENT   = "#58A6FF"
GREEN    = "#3FB950"
YELLOW   = "#D29922"
RED_C    = "#F85149"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
BORDER   = "#30363D"


# ─── ANIMATED CONVEYOR CANVAS ─────────────────────────────────────────────────
class ConveyorCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.belt_offset = 0
        self.running = True
        self.items_on_belt = deque()
        self._animate()

    def _animate(self):
        if not self.running:
            return
        self.delete("belt_stripe")
        w = int(self['width'])
        h = int(self['height'])
        belt_y1, belt_y2 = h // 2 - 18, h // 2 + 18

        self.create_rectangle(0, belt_y1, w, belt_y2, fill="#2D3748", outline=BORDER, tags="belt_stripe")
        stripe_w = 30
        for x in range(-stripe_w + (self.belt_offset % stripe_w), w + stripe_w, stripe_w):
            self.create_line(x, belt_y1, x - 20, belt_y2, fill="#4A5568", width=2, tags="belt_stripe")

        self.belt_offset += 2
        self.after(30, self._animate)

    def stop(self):
        self.running = False


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
class RobotWasteApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Robot Phân Loại Rác – AI Simulation")
        self.root.configure(bg=BG)
        self.root.geometry("1400x860")
        self.root.minsize(1200, 720)

        # State
        self.model = None
        self.cap = None
        self.running = False
        self.paused = False
        self.stats = defaultdict(int)
        self.bin_stats = defaultdict(int)
        self.fps_queue = deque(maxlen=30)
        self.last_detections = []
        self.total_processed = 0
        self.robot_angle = 0.0
        self.robot_target = 0.0
        self.arm_phase = "idle"       # idle | reach | grab | swing | drop | return
        self.arm_grab_item = None     # (color, emoji) being held
        self.arm_grab_timer = 0
        self.log_lines = deque(maxlen=50)
        self.confidence_threshold = tk.DoubleVar(value=CONF_THRESH)
        self.source_var = tk.StringVar(value="camera")
        self.camera_index = tk.IntVar(value=0)
        self._canvas_w = 900
        self.bin_positions = {}
        self._init_bin_positions()
        # class_map: model_name -> CLASS_CONFIG key
        self.class_map = {}
        # Tracking: set of track IDs already counted this session
        self.seen_track_ids = set()
        # Belt spawn control
        self._last_belt_time = 0.0
        self._belt_interval = 0.8    # min seconds between belt spawns
        self._pending_bin_angle = 0.0

        self._build_ui()
        self._load_model()

    def _init_bin_positions(self):
        """Pre-compute bin x centres using default canvas width."""
        w = self._canvas_w
        bin_list = list(BIN_COLORS.keys())
        spacing = w // (len(bin_list) + 1)
        for i, bname in enumerate(bin_list):
            self.bin_positions[bname] = spacing * (i + 1)

    # ── UI BUILD ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        # TOP BAR
        top = tk.Frame(self.root, bg=BG2, height=56)
        top.pack(fill=tk.X, side=tk.TOP)
        top.pack_propagate(False)

        tk.Label(top, text="🤖  ROBOT WASTE CLASSIFIER", fg=ACCENT, bg=BG2,
                 font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT, padx=20, pady=12)

        self.status_lbl = tk.Label(top, text="● OFFLINE", fg=RED_C, bg=BG2,
                                   font=("Segoe UI", 11, "bold"))
        self.status_lbl.pack(side=tk.LEFT, padx=10)

        self.fps_lbl = tk.Label(top, text="FPS: --", fg=SUBTEXT, bg=BG2,
                                font=("Segoe UI", 10))
        self.fps_lbl.pack(side=tk.LEFT, padx=16)

        self.total_lbl = tk.Label(top, text="Đã xử lý: 0", fg=TEXT, bg=BG2,
                                  font=("Segoe UI", 10))
        self.total_lbl.pack(side=tk.LEFT, padx=10)

        # RIGHT controls in top bar
        ctrl_frame = tk.Frame(top, bg=BG2)
        ctrl_frame.pack(side=tk.RIGHT, padx=16)

        self.btn_start = self._btn(ctrl_frame, "▶  Bắt Đầu", GREEN, self._start)
        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_pause = self._btn(ctrl_frame, "⏸  Tạm Dừng", YELLOW, self._pause, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, padx=4)
        self.btn_stop = self._btn(ctrl_frame, "⏹  Dừng", RED_C, self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self._btn(ctrl_frame, "🗑  Reset", SUBTEXT, self._reset_stats).pack(side=tk.LEFT, padx=4)

        # MAIN LAYOUT
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # LEFT: camera + robot visualizer
        left = tk.Frame(main, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_camera_panel(left)
        self._build_robot_panel(left)

        # RIGHT: controls + stats + log
        right = tk.Frame(main, bg=BG, width=360)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        right.pack_propagate(False)

        self._build_controls(right)
        self._build_stats_panel(right)
        self._build_log_panel(right)

    def _btn(self, parent, text, color, cmd, state=tk.NORMAL):
        return tk.Button(parent, text=text, bg=color, fg="white",
                         font=("Segoe UI", 9, "bold"), relief=tk.FLAT,
                         padx=10, pady=5, cursor="hand2",
                         command=cmd, state=state,
                         activebackground=color, activeforeground="white")

    def _card(self, parent, title, **kwargs):
        frame = tk.Frame(parent, bg=BG2, bd=0, relief=tk.FLAT,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill=tk.X, pady=4, **kwargs)
        if title:
            tk.Label(frame, text=title, fg=SUBTEXT, bg=BG2,
                     font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=10, pady=(8, 2))
        return frame

    # ── CAMERA PANEL ──────────────────────────────────────────────────────────
    def _build_camera_panel(self, parent):
        card = tk.Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        header = tk.Frame(card, bg=BG3)
        header.pack(fill=tk.X)
        tk.Label(header, text="📷  Camera Feed – YOLOv8 Detection", fg=TEXT, bg=BG3,
                 font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=12, pady=6)
        self.det_count_lbl = tk.Label(header, text="Phát hiện: 0 vật thể", fg=ACCENT, bg=BG3,
                                       font=("Segoe UI", 9))
        self.det_count_lbl.pack(side=tk.RIGHT, padx=12)

        # Fixed container prevents label from resizing to image size
        self.cam_container = tk.Frame(card, bg="#000000")
        self.cam_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self.cam_label = tk.Label(self.cam_container, bg="#000000",
                                  text="Camera chưa hoạt động\n\nNhấn ▶ Bắt Đầu",
                                  fg=SUBTEXT, font=("Segoe UI", 13))
        self.cam_label.place(relx=0, rely=0, relwidth=1, relheight=1)

    # ── ROBOT VISUALIZER ──────────────────────────────────────────────────────
    def _build_robot_panel(self, parent):
        card = tk.Frame(parent, bg=BG2, highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 0))

        tk.Label(card, text="🦾  Băng Chuyền & Cánh Tay Robot", fg=SUBTEXT, bg=BG2,
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=10, pady=(6, 2))

        self.robot_canvas = tk.Canvas(card, bg=BG, height=180, highlightthickness=0)
        self.robot_canvas.pack(fill=tk.X, padx=4, pady=4)
        self.robot_canvas.bind("<Configure>", self._on_canvas_resize)

        # belt_items: list of dicts {x, color, emoji, active}
        self.belt_items = []
        self.belt_offset = 0  # for stripe animation
        self._draw_robot_scene()

    def _on_canvas_resize(self, event):
        self._canvas_w = event.width
        self._init_bin_positions()

    def _draw_robot_scene(self):
        c = self.robot_canvas
        c.delete("all")
        w = max(c.winfo_width(), 400)
        h = 180
        belt_y = 110   # vertical centre of belt
        arm_base_x = w // 2
        arm_base_y = 20
        arm_len = 75

        # ── Bins ──────────────────────────────────────────────────────────────
        bin_list = list(BIN_COLORS.items())
        spacing = w // (len(bin_list) + 1)
        self.bin_positions = {}
        for i, (bname, bcol) in enumerate(bin_list):
            bx = spacing * (i + 1)
            # Bin body
            c.create_rectangle(bx - 36, belt_y + 22, bx + 36, belt_y + 72,
                                fill=bcol, outline="white", width=1)
            # Bin opening highlight
            c.create_rectangle(bx - 36, belt_y + 22, bx + 36, belt_y + 32,
                                fill="white", stipple="gray25", outline="")
            c.create_text(bx, belt_y + 47, text=bname, fill="white",
                          font=("Segoe UI", 7, "bold"))
            cnt = self.bin_stats.get(bname, 0)
            c.create_text(bx, belt_y + 63, text=str(cnt), fill="white",
                          font=("Segoe UI", 10, "bold"))
            self.bin_positions[bname] = bx

        # ── Belt ──────────────────────────────────────────────────────────────
        belt_top = belt_y - 14
        belt_bot = belt_y + 14
        c.create_rectangle(0, belt_top, w, belt_bot, fill="#2D3748", outline=BORDER)
        # Moving stripes
        stripe_gap = 28
        off = int(self.belt_offset) % stripe_gap
        for x in range(-stripe_gap + off, w + stripe_gap, stripe_gap):
            c.create_line(x, belt_top, x - 18, belt_bot, fill="#4A5568", width=2)
        # Belt rollers
        for rx in [12, w - 12]:
            c.create_oval(rx - 10, belt_top - 4, rx + 10, belt_bot + 4,
                          fill="#4A5568", outline=BORDER)

        # ── Belt items ────────────────────────────────────────────────────────
        alive = []
        for item in self.belt_items:
            if item.get("grabbed"):
                alive.append(item)
                continue
            # Move left
            if self.running and not self.paused:
                item["x"] -= 1.8
            bx = item["x"]
            # Remove if gone off-screen
            if bx < -30:
                continue
            alive.append(item)
            col = item["color"]
            emoji = item["emoji"]
            c.create_oval(bx - 14, belt_y - 28, bx + 14, belt_y - 2,
                          fill=col, outline="white", width=1)
            c.create_text(bx, belt_y - 15, text=emoji, font=("Segoe UI", 9))
        self.belt_items = alive

        # ── Robot arm ─────────────────────────────────────────────────────────
        angle_rad = math.radians(self.robot_angle)
        arm_tip_x = arm_base_x + arm_len * math.sin(angle_rad)
        arm_tip_y = arm_base_y + arm_len * math.cos(angle_rad)

        # Upper arm (thick)
        c.create_line(arm_base_x, arm_base_y, arm_tip_x, arm_tip_y,
                      fill=ACCENT, width=8, capstyle=tk.ROUND)
        # Forearm extension
        fore_x = arm_tip_x + 18 * math.sin(angle_rad)
        fore_y = arm_tip_y + 18 * math.cos(angle_rad)
        c.create_line(arm_tip_x, arm_tip_y, fore_x, fore_y,
                      fill="#90CAF9", width=5, capstyle=tk.ROUND)

        # Gripper claws
        if self.arm_phase == "grab":
            claw_open = 4
            grip_col = RED_C
        else:
            claw_open = 10
            grip_col = GREEN
        perp_rad = angle_rad + math.pi / 2
        for sign in (-1, 1):
            cx = fore_x + sign * claw_open * math.cos(perp_rad)
            cy = fore_y - sign * claw_open * math.sin(perp_rad)
            c.create_line(fore_x, fore_y, cx, cy,
                          fill=grip_col, width=4, capstyle=tk.ROUND)

        # Base joint
        c.create_oval(arm_base_x - 12, arm_base_y - 12,
                      arm_base_x + 12, arm_base_y + 12,
                      fill=ACCENT, outline="white", width=2)
        # Elbow joint
        c.create_oval(arm_tip_x - 7, arm_tip_y - 7,
                      arm_tip_x + 7, arm_tip_y + 7,
                      fill="#90CAF9", outline="white", width=1)

        # Item carried by gripper
        if self.arm_grab_item and self.arm_phase in ("swing", "drop"):
            col, emoji = self.arm_grab_item
            c.create_oval(fore_x - 12, fore_y - 12, fore_x + 12, fore_y + 12,
                          fill=col, outline="white")
            c.create_text(fore_x, fore_y, text=emoji, font=("Segoe UI", 8))

        # ── Animate arm state machine ─────────────────────────────────────────
        if self.running and not self.paused:
            self.belt_offset += 1.8

            if self.arm_phase == "idle":
                # Slowly drift back to center
                self.robot_angle += (0 - self.robot_angle) * 0.05

                # Auto-check belt for any item in range and trigger grab
                if self.belt_items:
                    cw = max(self.robot_canvas.winfo_width(), 400)
                    cx = cw // 2
                    for item in self.belt_items:
                        if item.get("grabbed") or item.get("targeted"):
                            continue
                        if abs(item["x"] - cx) < cw * 0.5:
                            self.arm_grab_item = (item["color"], item["emoji"])
                            item_angle = math.degrees(math.atan2(item["x"] - cx, 80))
                            self.robot_target = max(-65, min(65, item_angle))
                            bin_x = self.bin_positions.get(item.get("bin", "Nhựa"), cx)
                            bin_angle = math.degrees(math.atan2(bin_x - cx, 80))
                            self._pending_bin_angle = max(-65, min(65, bin_angle))
                            item["targeted"] = True
                            self.arm_phase = "reach"
                            break

            elif self.arm_phase == "reach":
                diff = self.robot_target - self.robot_angle
                self.robot_angle += diff * 0.14
                if abs(diff) < 1.5:
                    self.arm_phase = "grab"
                    self.arm_grab_timer = 10   # brief pause at item

            elif self.arm_phase == "grab":
                self.arm_grab_timer -= 1
                if self.arm_grab_timer <= 0:
                    # Grab the targeted item
                    for item in self.belt_items:
                        if item.get("targeted") and not item.get("grabbed"):
                            item["grabbed"] = True
                            break
                    self.arm_phase = "swing"
                    self.robot_target = self._pending_bin_angle

            elif self.arm_phase == "swing":
                diff = self.robot_target - self.robot_angle
                self.robot_angle += diff * 0.11
                if abs(diff) < 2.5:
                    self.arm_phase = "drop"
                    self.arm_grab_timer = 14

            elif self.arm_phase == "drop":
                self.arm_grab_timer -= 1
                if self.arm_grab_timer <= 0:
                    self.belt_items = [i for i in self.belt_items
                                       if not i.get("grabbed")]
                    self.arm_grab_item = None
                    self.arm_phase = "return"
                    self.robot_target = 0

            elif self.arm_phase == "return":
                self.robot_angle += (0 - self.robot_angle) * 0.12
                if abs(self.robot_angle) < 1.5:
                    self.robot_angle = 0.0
                    self.arm_phase = "idle"

        self.root.after(30, self._draw_robot_scene)

    # ── CONTROLS PANEL ────────────────────────────────────────────────────────
    def _build_controls(self, parent):
        card = self._card(parent, "⚙️  Cài Đặt")

        # Source
        src_frame = tk.Frame(card, bg=BG2)
        src_frame.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(src_frame, text="Nguồn:", fg=TEXT, bg=BG2, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(src_frame, text="Camera", variable=self.source_var, value="camera",
                       bg=BG2, fg=TEXT, selectcolor=BG3, font=("Segoe UI", 9),
                       activebackground=BG2).pack(side=tk.LEFT, padx=8)
        tk.Radiobutton(src_frame, text="Video", variable=self.source_var, value="video",
                       bg=BG2, fg=TEXT, selectcolor=BG3, font=("Segoe UI", 9),
                       activebackground=BG2).pack(side=tk.LEFT)

        # Camera index
        cam_frame = tk.Frame(card, bg=BG2)
        cam_frame.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(cam_frame, text="Camera ID:", fg=TEXT, bg=BG2, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Spinbox(cam_frame, from_=0, to=5, textvariable=self.camera_index, width=4,
                   bg=BG3, fg=TEXT, insertbackground=TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=8)

        # Confidence
        conf_frame = tk.Frame(card, bg=BG2)
        conf_frame.pack(fill=tk.X, padx=10, pady=4)
        tk.Label(conf_frame, text="Ngưỡng tin cậy:", fg=TEXT, bg=BG2, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        self.conf_val_lbl = tk.Label(conf_frame, text=f"{CONF_THRESH:.0%}", fg=ACCENT, bg=BG2,
                                     font=("Segoe UI", 9, "bold"))
        self.conf_val_lbl.pack(side=tk.RIGHT)

        slider = ttk.Scale(card, from_=0.1, to=0.95, variable=self.confidence_threshold,
                           orient=tk.HORIZONTAL, command=self._update_conf)
        slider.pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Frame(card, bg=BG2, height=4).pack()

    def _update_conf(self, val):
        self.conf_val_lbl.config(text=f"{float(val):.0%}")

    # ── STATS PANEL ───────────────────────────────────────────────────────────
    def _build_stats_panel(self, parent):
        card = self._card(parent, "📊  Thống Kê Phân Loại")
        inner = tk.Frame(card, bg=BG2)
        inner.pack(fill=tk.X, padx=10, pady=(0, 8))

        self.stat_rows = {}
        for cls_name, cfg in CLASS_CONFIG.items():
            row = tk.Frame(inner, bg=BG2)
            row.pack(fill=tk.X, pady=1)

            tk.Label(row, text=cfg["emoji"], bg=BG2, width=2,
                     font=("Segoe UI", 10)).pack(side=tk.LEFT)
            tk.Label(row, text=cls_name[:18], fg=TEXT, bg=BG2,
                     font=("Segoe UI", 8), width=18, anchor=tk.W).pack(side=tk.LEFT)

            bar_frame = tk.Frame(row, bg=BG3, height=8, width=80)
            bar_frame.pack(side=tk.LEFT, padx=4)
            bar_frame.pack_propagate(False)
            bar = tk.Frame(bar_frame, bg=cfg["color"], height=8, width=0)
            bar.place(x=0, y=0, relheight=1)

            cnt_lbl = tk.Label(row, text="0", fg=cfg["color"], bg=BG2,
                               font=("Segoe UI", 8, "bold"), width=4, anchor=tk.E)
            cnt_lbl.pack(side=tk.RIGHT)

            self.stat_rows[cls_name] = {"bar": bar, "count": cnt_lbl, "bar_frame": bar_frame}

        # Bin totals
        tk.Frame(card, bg=BORDER, height=1).pack(fill=tk.X, padx=10, pady=4)
        bin_row = tk.Frame(card, bg=BG2)
        bin_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.bin_total_lbls = {}
        for bname, bcol in BIN_COLORS.items():
            col = tk.Frame(bin_row, bg=BG2)
            col.pack(side=tk.LEFT, expand=True)
            tk.Label(col, text=bname, fg=SUBTEXT, bg=BG2,
                     font=("Segoe UI", 7)).pack()
            lbl = tk.Label(col, text="0", fg=bcol, bg=BG2,
                           font=("Segoe UI", 14, "bold"))
            lbl.pack()
            self.bin_total_lbls[bname] = lbl

    # ── LOG PANEL ─────────────────────────────────────────────────────────────
    def _build_log_panel(self, parent):
        card = self._card(parent, "📋  Nhật Ký Phân Loại")
        self.log_text = tk.Text(card, bg=BG3, fg=TEXT, font=("Consolas", 8),
                                height=10, wrap=tk.WORD, relief=tk.FLAT,
                                state=tk.DISABLED, insertbackground=TEXT)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.log_text.tag_config("green", foreground=GREEN)
        self.log_text.tag_config("yellow", foreground=YELLOW)
        self.log_text.tag_config("blue", foreground=ACCENT)
        self.log_text.tag_config("sub", foreground=SUBTEXT)

    # ── MODEL LOAD ────────────────────────────────────────────────────────────
    def _load_model(self):
        self._log(f"Đang tải mô hình: {MODEL_PATH}", "blue")
        def _load():
            try:
                self.model = YOLO(MODEL_PATH)
                # Warm up
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model(dummy, verbose=False)
                self.root.after(0, self._on_model_loaded)
            except Exception as e:
                self.root.after(0, lambda: self._log(f"❌ Lỗi tải model: {e}", "yellow"))
        threading.Thread(target=_load, daemon=True).start()

    def _on_model_loaded(self):
        names = self.model.names
        # Log tất cả class names để debug
        name_list = list(names.values())
        self._log(f"✅ Model loaded | {len(names)} lớp", "green")
        self._log(f"   Classes: {name_list}", "blue")

        # Build class_map: model_name → CLASS_CONFIG key
        cfg_lower = {k.lower(): k for k in CLASS_CONFIG}
        self.class_map = {}

        for cls_id, name in names.items():
            name_lo = name.lower()

            # 1) Exact match (case-insensitive)
            key = cfg_lower.get(name_lo)
            if key:
                self.class_map[name] = key
                continue

            # 2) Partial match: model name contained in config key or vice versa
            matched = False
            for ck_lo, ck in cfg_lower.items():
                if name_lo in ck_lo or ck_lo in name_lo:
                    self.class_map[name] = ck
                    matched = True
                    break
            if matched:
                continue

            # 3) Keyword → Bin fallback: not in CLASS_CONFIG but can infer bin
            bin_fallback = None
            for keywords, bin_name in KEYWORD_BIN:
                if any(kw in name_lo for kw in keywords):
                    bin_fallback = bin_name
                    break

            if bin_fallback:
                # Create a dynamic entry in CLASS_CONFIG on-the-fly
                bd = BIN_DEFAULT[bin_fallback]
                CLASS_CONFIG[name] = {
                    "color": bd["color"], "bin": bin_fallback, "emoji": bd["emoji"]
                }
                self.class_map[name] = name   # self-map
                self._log(f"   ⚠️ '{name}' không có trong config → tự động gán: {bin_fallback}", "yellow")
            else:
                # 4) Truly unknown → Khác
                CLASS_CONFIG[name] = {
                    "color": BIN_DEFAULT["Khác"]["color"],
                    "bin": "Khác",
                    "emoji": BIN_DEFAULT["Khác"]["emoji"]
                }
                self.class_map[name] = name
                self._log(f"   ❌ '{name}' không nhận ra → Khác", "yellow")

        matched_summary = [(n, CLASS_CONFIG[self.class_map[n]]["bin"])
                           for n in name_list if n in self.class_map]
        self._log("   Kết quả phân loại: " +
                  ", ".join(f"{n}→{b}" for n, b in matched_summary), "blue")

        self.status_lbl.config(text="● SẴN SÀNG", fg=GREEN)
        self.btn_start.config(state=tk.NORMAL)

    # ── CAMERA LOOP ───────────────────────────────────────────────────────────
    def _start(self):
        if not self.model:
            return
        src = 0 if self.source_var.get() == "camera" else self.source_var.get()
        if self.source_var.get() == "camera":
            src = self.camera_index.get()

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            self._log("❌ Không mở được nguồn video/camera", "yellow")
            return

        self.running = True
        self.paused = False
        self.status_lbl.config(text="● ĐANG CHẠY", fg=GREEN)
        self.btn_start.config(state=tk.DISABLED)
        self.btn_pause.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.NORMAL)
        self._log("▶ Bắt đầu phân loại rác...", "green")

        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _camera_loop(self):
        prev_time = time.time()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if self.source_var.get() == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            # Tracking inference (persist=True giữ ID qua các frame)
            conf = self.confidence_threshold.get()
            try:
                results = self.model.track(frame, conf=conf, persist=True,
                                           tracker="bytetrack.yaml", verbose=False)
            except Exception:
                results = self.model(frame, conf=conf, verbose=False)

            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                confidence = float(box.conf[0])
                # track_id = None nếu tracker chưa gán
                track_id = int(box.id[0]) if (box.id is not None) else None
                detections.append((cls_name, confidence, track_id))

            # Annotate
            annotated = results[0].plot()

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            self.fps_queue.append(fps)
            avg_fps = sum(self.fps_queue) / len(self.fps_queue)

            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(annotated, f"Conf: {conf:.0%}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

            # Update UI
            self.last_detections = detections
            self.root.after(0, lambda f=annotated, d=detections, fps=avg_fps:
                            self._update_ui(f, d, fps))

            # Process detections in UI thread
            now = time.time()
            self.root.after(0, lambda d=detections, t=now: self._process_detections(d, t))

            time.sleep(0.03)

        self.root.after(0, self._on_stop)

    def _update_ui(self, frame, detections, fps):
        # ── Letterbox camera image – never zoom, always fit with black bars ──
        h, w = frame.shape[:2]
        disp_w = self.cam_container.winfo_width()
        disp_h = self.cam_container.winfo_height()
        if disp_w > 20 and disp_h > 20:
            # Scale to fit inside container, keep aspect ratio
            scale = min(disp_w / w, disp_h / h)
            nw = int(w * scale)
            nh = int(h * scale)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_img = pil_img.resize((nw, nh), Image.LANCZOS)

            # Paste onto black canvas (letterbox)
            canvas_img = Image.new("RGB", (disp_w, disp_h), (0, 0, 0))
            x_off = (disp_w - nw) // 2
            y_off = (disp_h - nh) // 2
            canvas_img.paste(pil_img, (x_off, y_off))

            tk_img = ImageTk.PhotoImage(canvas_img)
            self.cam_label.config(image=tk_img, text="")
            self.cam_label._img = tk_img

        self.fps_lbl.config(text=f"FPS: {fps:.1f}")
        self.det_count_lbl.config(text=f"Phát hiện: {len(detections)} vật thể")

    def _process_detections(self, detections, frame_time):
        """Đếm mỗi track_id chỉ 1 lần. Gọi từ UI thread."""
        if not detections:
            return

        spawned_this_frame = False
        now = time.time()

        for cls_name, conf, track_id in detections:
            # ── Resolve class ──────────────────────────────────────────────
            cfg_key = self.class_map.get(cls_name)
            if cfg_key is None:
                continue   # model chưa load xong

            cfg = CLASS_CONFIG.get(cfg_key)
            if cfg is None:
                cfg = {"color": BIN_DEFAULT["Khác"]["color"],
                       "bin": "Khác", "emoji": BIN_DEFAULT["Khác"]["emoji"]}

            bin_name   = cfg["bin"]
            item_color = cfg["color"]
            item_emoji = cfg["emoji"]

            # ── Chỉ đếm khi track_id lần đầu xuất hiện ────────────────────
            if track_id is not None:
                if track_id in self.seen_track_ids:
                    continue   # vật này đã được đếm rồi, bỏ qua
                self.seen_track_ids.add(track_id)
            else:
                # Không có tracker (fallback) → dùng rate-limit 3s/class
                last = getattr(self, "_last_count_time", {})
                if now - last.get(cls_name, 0) < 3.0:
                    continue
                if not hasattr(self, "_last_count_time"):
                    self._last_count_time = {}
                self._last_count_time[cls_name] = now

            # ── Đếm thống kê ──────────────────────────────────────────────
            self.stats[cfg_key] += 1
            self.total_processed += 1
            self.bin_stats[bin_name] += 1

            # ── Spawn 1 item lên belt ──────────────────────────────────────
            if not spawned_this_frame and now - self._last_belt_time >= self._belt_interval:
                if len(self.belt_items) < 10:
                    cw = max(self.robot_canvas.winfo_width(), 400)
                    self.belt_items.append({
                        "x": float(cw + 40),
                        "color": item_color,
                        "emoji": item_emoji,
                        "bin":   bin_name,
                        "grabbed":  False,
                        "targeted": False,
                    })
                    self._last_belt_time = now
                    spawned_this_frame = True

            # ── Trigger robot nếu đang idle ────────────────────────────────
            if self.arm_phase == "idle":
                cw = max(self.robot_canvas.winfo_width(), 400)
                cx = cw // 2
                for item in self.belt_items:
                    if item.get("grabbed") or item.get("targeted"):
                        continue
                    if abs(item["x"] - cx) < cw * 0.55:
                        self.arm_grab_item = (item["color"], item["emoji"])
                        item_angle = math.degrees(math.atan2(item["x"] - cx, 80))
                        self.robot_target = max(-65, min(65, item_angle))
                        self.arm_phase = "reach"
                        bin_x = self.bin_positions.get(item["bin"], cx)
                        bin_angle = math.degrees(math.atan2(bin_x - cx, 80))
                        self._pending_bin_angle = max(-65, min(65, bin_angle))
                        item["targeted"] = True
                        break

            # ── Log ────────────────────────────────────────────────────────
            track_str = f"#{track_id}" if track_id is not None else ""
            ts = time.strftime("%H:%M:%S")
            self._log(f"[{ts}] {item_emoji} {cls_name}{track_str} ({conf:.1%}) → 🗑 {bin_name}",
                      "green")

        self._update_stats()
        self.total_lbl.config(text=f"Đã xử lý: {self.total_processed}")

    def _update_stats(self):
        max_val = max(self.stats.values(), default=1)
        for cls_name, row in self.stat_rows.items():
            cnt = self.stats.get(cls_name, 0)
            row["count"].config(text=str(cnt))
            bar_w = int((cnt / max_val) * 80) if max_val > 0 else 0
            row["bar"].place(x=0, y=0, relheight=1, width=bar_w)

        for bname, lbl in self.bin_total_lbls.items():
            lbl.config(text=str(self.bin_stats.get(bname, 0)))

    def _log(self, msg, tag=""):
        def _do():
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n", tag)
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
        self.root.after(0, _do)

    # ── CONTROLS ──────────────────────────────────────────────────────────────
    def _pause(self):
        self.paused = not self.paused
        if self.paused:
            self.btn_pause.config(text="▶  Tiếp Tục")
            self.status_lbl.config(text="● TẠM DỪNG", fg=YELLOW)
        else:
            self.btn_pause.config(text="⏸  Tạm Dừng")
            self.status_lbl.config(text="● ĐANG CHẠY", fg=GREEN)

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _on_stop(self):
        self.status_lbl.config(text="● ĐÃ DỪNG", fg=RED_C)
        self.btn_start.config(state=tk.NORMAL)
        self.btn_pause.config(state=tk.DISABLED, text="⏸  Tạm Dừng")
        self.btn_stop.config(state=tk.DISABLED)
        self.cam_label.config(image="", text="Camera đã dừng\n\nNhấn ▶ Bắt Đầu để tiếp tục", fg=SUBTEXT)
        self._log("⏹ Đã dừng hệ thống.", "yellow")

    def _reset_stats(self):
        self.stats.clear()
        self.bin_stats.clear()
        self.total_processed = 0
        self.belt_items.clear()
        self.seen_track_ids.clear()   # reset tracking IDs
        self.arm_phase = "idle"
        self.arm_grab_item = None
        self.robot_angle = 0.0
        self.robot_target = 0.0
        self._last_belt_time = 0.0
        self.total_lbl.config(text="Đã xử lý: 0")
        for row in self.stat_rows.values():
            row["count"].config(text="0")
            row["bar"].place(width=0)
        for lbl in self.bin_total_lbls.values():
            lbl.config(text="0")
        self._log("🗑 Đã reset thống kê.", "sub")

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Horizontal.TScale", background=BG2, troughcolor=BG3,
                    slidercolor=ACCENT, bordercolor=BORDER)

    app = RobotWasteApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
