import os
import sys
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "models")
sys.path.append(os.path.join(BASE, "src"))

CLASS_DISPLAY = {0: "NO PERSON", 1: "PERSON DETECTED"}
COLORS = {0: "#2ecc71", 1: "#e74c3c"}


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class PersonDetector:
    def __init__(self):
        d = np.load(os.path.join(MODEL_DIR, "preprocess.npz"))
        self.scaler_mean = d["scaler_mean"]
        self.scaler_std = d["scaler_std"]
        self.pca_mean = d["pca_mean"]
        self.pca_components = d["pca_components"]
        self.pca_scaler_mean = d["pca_scaler_mean"]
        self.pca_scaler_std = d["pca_scaler_std"]
        self.background = d["background"]
        self.img_size = int(d["img_size"])
        self.models = {}
        for name in ("linear_regression", "logistic_regression", "svm"):
            m = np.load(os.path.join(MODEL_DIR, f"{name}.npz"))
            self.models[name] = {"w": m["w"], "b": float(m["b"][0])}

    def _preprocess(self, gray):
        img = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        raw = img.reshape(-1)
        gx = np.abs(np.diff(img, axis=1, prepend=img[:, :1]))
        gy = np.abs(np.diff(img, axis=0, prepend=img[:1, :]))
        grad = (gx + gy).reshape(-1)
        diff = np.abs(raw - self.background)
        feat = np.concatenate([raw, grad, diff])
        feat = (feat - self.scaler_mean) / self.scaler_std
        z = (feat - self.pca_mean) @ self.pca_components.T
        z = (z - self.pca_scaler_mean) / self.pca_scaler_std
        return z

    def predict_all(self, gray):
        z = self._preprocess(gray)
        out = {}
        w = self.models["linear_regression"]
        lin = float(z @ w["w"] + w["b"])
        out["linear_regression"] = {"score": lin, "prob": float(np.clip(lin, 0, 1)), "pred": int(lin >= 0.5)}
        w = self.models["logistic_regression"]
        lp = float(sigmoid(z @ w["w"] + w["b"]))
        out["logistic_regression"] = {"score": lp, "prob": lp, "pred": int(lp >= 0.5)}
        w = self.models["svm"]
        sv = float(z @ w["w"] + w["b"])
        svp = float(sigmoid(sv * 2.0))
        out["svm"] = {"score": sv, "prob": svp, "pred": int(sv >= 0)}
        return out

class App:
    def __init__(self, root):
        self.root = root
        root.title("Person Detection - ML Midterm")
        root.configure(bg="#1e1e1e")
        self.detector = PersonDetector()
        self.cap = None
        self.running = False
        self.current_frame = None
        self.mode = "webcam"
        self._build_ui()

    def _build_ui(self):
        left = tk.Frame(self.root, bg="#1e1e1e")
        left.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(left, bg="#000")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        btns = tk.Frame(left, bg="#1e1e1e")
        btns.pack(pady=8)
        tk.Button(btns, text="Bat webcam", command=self.start_webcam, bg="#3498db", fg="white", width=14).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Dung", command=self.stop, bg="#95a5a6", fg="white", width=10).pack(side=tk.LEFT, padx=4)
        tk.Button(btns, text="Mo anh", command=self.open_image, bg="#9b59b6", fg="white", width=10).pack(side=tk.LEFT, padx=4)

        right = tk.Frame(self.root, bg="#1e1e1e", width=260)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        right.pack_propagate(False)
        self.overall = tk.Label(right, text="--", bg="#1e1e1e", fg="#f1c40f", font=("Segoe UI", 15, "bold"), wraplength=240, justify="left")
        self.overall.pack(anchor="w", pady=(0, 8))
        self.model_labels = {}
        names = {"linear_regression": "Linear Reg", "logistic_regression": "Logistic Reg", "svm": "Linear SVM"}
        for key, disp in names.items():
            box = tk.Frame(right, bg="#2b2b2b", bd=1, relief=tk.RIDGE)
            box.pack(fill=tk.X, pady=3)
            lbl = tk.Label(box, text=f"{disp}: --", bg="#2b2b2b", fg="white", font=("Segoe UI", 9), anchor="w")
            lbl.pack(anchor="w", padx=6, pady=(4, 0))
            bar = tk.Canvas(box, height=10, bg="#444", highlightthickness=0)
            bar.pack(fill=tk.X, padx=6, pady=(2, 6))
            self.model_labels[key] = (lbl, bar)

    def start_webcam(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Loi", "Khong mo duoc webcam (index 0).")
            self.cap = None
            return
        self.mode = "webcam"
        self.running = True
        self._loop()

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Anh", "*.jpg *.jpeg *.png *.bmp"), ("All", "*.*")])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Loi", "Khong doc duoc anh.")
            return
        self.stop()
        self.mode = "image"
        self.current_frame = img
        self._process_and_show(img)

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _loop(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self.current_frame = frame
            self._process_and_show(frame)
        self.root.after(15, self._loop)

    def _process_and_show(self, frame):
        frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = self.detector.predict_all(gray)
        pred = res["logistic_regression"]["pred"]
        color = "#e74c3c" if pred == 1 else "#2ecc71"
        label = CLASS_DISPLAY[pred]
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self._hex_bgr(color), 2)
        self._show_frame(frame)
        self.overall.config(text=label, fg=color)
        for key, (lbl, bar) in self.model_labels.items():
            r = res[key]
            c = "#e74c3c" if r["pred"] == 1 else "#2ecc71"
            name = {"linear_regression": "Linear Reg", "logistic_regression": "Logistic Reg", "svm": "Linear SVM"}[key]
            lbl.config(text=f"{name}: {r['prob'] * 100:.1f}%")
            self._draw_bar(bar, r["prob"], c)

    def _draw_bar(self, canvas, prob, color):
        canvas.delete("all")
        w = canvas.winfo_width()
        if w <= 1:
            w = 220
        canvas.create_rectangle(0, 0, w, 10, fill="#444", outline="")
        canvas.create_rectangle(0, 0, int(w * max(0.0, min(1.0, prob))), 10, fill=color, outline="")

    @staticmethod
    def _hex_bgr(h):
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b, g, r)

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.video_label.update_idletasks()
        tw = max(self.video_label.winfo_width(), 480)
        th = max(self.video_label.winfo_height(), 360)
        scale = min(tw / w, th / h)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1100x650")
    App(root)
    root.mainloop()
