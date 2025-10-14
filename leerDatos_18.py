import threading
import queue
import time
import os
import math
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ====== Opcional: Savitzky–Golay ======
HAS_SCIPY = True
try:
    from scipy.signal import savgol_filter
except Exception:
    HAS_SCIPY = False
# ======================================

# ==== CONFIG ====
BAUD = 115200
HEADER = "Time[s] CO2_smooth[ppm] dCO2_dt[ppm/s] d2CO2_dt2[ppm/s^2]"
PPM_MIN, PPM_MAX = 250, 200000   # permitir valores “ajustados” >100k
# =================


# ==================  Phantom (simulador)  ===================
class PhantomSensor:
    """
    Simula el flujo serie del ESP32:
    - Comandos: G, Z, S fi dv D, X, L fmin fmax
    - Emite 1 muestra/seg: 't ppm'
    - Incluye un “wrap” cerca de 65k para probar #FIX_APLICADO.
    """
    def __init__(self, on_line_cb):
        self.on_line_cb = on_line_cb
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        # Estado de estimulación
        self.stim_on = False
        self.stim_f = 0.0
        self.fmin = 0.5
        self.fmax = 5000.0
        # Estado adquisición simulada
        self.t0 = None
        self.t_s = 0

    def connect(self):
        if self.running:
            return True
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.on_line_cb(f"#STIM_LIMITS fmin={self.fmin:.6f} fmax={self.fmax:.6f}")
        return True

    def disconnect(self):
        self.running = False

    def sendln(self, s: str):
        cmd = s.strip()
        if not cmd:
            return True
        if cmd.upper() == "G":
            with self.lock:
                self.t0 = time.time()
                self.t_s = 0
            return True
        if cmd.upper() == "Z":
            self.on_line_cb("#ZERO_CAL_SENT")
            return True
        if cmd.upper().startswith("S"):
            try:
                _, a, b, c = cmd.split()
                fi = float(a); dv = float(b); D = float(c)
                with self.lock:
                    self.stim_on = True
                    self.stim_f = max(self.fmin, min(self.fmax, fi))
                self.on_line_cb(f"#STIM_START Fi={fi:.6f} Delta={dv:.6f} D={D:.6f} | LIMITS fmin={self.fmin:.6f} fmax={self.fmax:.6f}")
                self.on_line_cb(f"#STIM_F={self.stim_f:.6f}")
            except Exception:
                self.on_line_cb("#STIM_BAD_PARAMS")
            return True
        if cmd.upper() == "X":
            with self.lock:
                self.stim_on = False
                self.stim_f = 0.0
            self.on_line_cb("#STIM_STOP")
            self.on_line_cb("#STIM_F=0.000000")
            return True
        if cmd.upper().startswith("L"):
            try:
                _, a, b = cmd.split()
                fmin = float(a); fmax = float(b)
                if not (fmin > 0 and fmax > fmin):
                    raise ValueError
                with self.lock:
                    self.fmin = fmin; self.fmax = fmax
                    if self.stim_on:
                        self.stim_f = max(self.fmin, min(self.fmax, self.stim_f))
                self.on_line_cb(f"#STIM_LIMITS fmin={self.fmin:.6f} fmax={self.fmax:.6f}")
                if self.stim_on:
                    self.on_line_cb(f"#STIM_F={self.stim_f:.6f}")
            except Exception:
                self.on_line_cb("#STIM_LIMITS_BAD_PARAMS")
            return True
        return True

    def _worker(self):
        wrap_done = False
        while self.running:
            now = time.time()
            if self.t0 is None:
                time.sleep(0.05)
                continue
            t = int(now - self.t0)
            if t != self.t_s:
                self.t_s = t
                ppm = self._model_ppm(t, wrap_done)
                if not wrap_done and ppm >= 65000:
                    self.on_line_cb(f"{t:d} {int(12000)}")
                    wrap_done = True
                else:
                    self.on_line_cb(f"{t:d} {int(ppm)}")
            time.sleep(0.02)

    def _model_ppm(self, t, wrapped):
        if t < 10:
            return 400 + 30*math.sin(t/2.0)
        val = 400 + 1200*(t-10) + 8000*math.sin((t-10)/30.0)
        if not wrapped:
            return min(val, 66000)
        else:
            return min(12000 + 600*(t-10), 40000)
# ==================  /Phantom  ===================


class SerialManager:
    """Puerto serie persistente + hilo lector."""
    def __init__(self, on_line_cb):
        self.ser = None
        self.port = None
        self.running = False
        self.reader_thread = None
        self.on_line_cb = on_line_cb
        self.lock = threading.Lock()

    def connect(self, port):
        if self.ser and self.ser.is_open:
            return True
        try:
            self.ser = serial.Serial(
                port,
                BAUD,
                timeout=0.2,
                rtscts=False,
                dsrdtr=False,
            )
            try:
                self.ser.setDTR(False)
                self.ser.setRTS(False)
            except Exception:
                pass
            time.sleep(0.2)
            self.ser.reset_input_buffer()
            self.port = port
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader, daemon=True)
            self.reader_thread.start()
            return True
        except Exception as e:
            return f"Error al abrir {port}: {e}"

    def disconnect(self):
        self.running = False
        try:
            if self.ser:
                self.ser.close()
        except:
            pass
        self.ser = None
        self.port = None

    def sendln(self, s: str):
        with self.lock:
            if not self.ser or not self.ser.is_open:
                return False
            payload = (s.strip() + "\n").encode("utf-8")
            try:
                self.ser.write(payload)
                self.ser.flush()
                return True
            except Exception:
                return False

    def _reader(self):
        bad_streak = 0
        while self.running:
            try:
                line = self.ser.readline()
                if not line:
                    continue
                try:
                    text = line.decode("utf-8", errors="ignore").rstrip("\r\n")
                except:
                    continue
                if not text:
                    continue
                self.on_line_cb(text)
                bad_streak = 0
            except Exception as e:
                bad_streak += 1
                if bad_streak > 20:
                    self.on_line_cb(f"[SERIAL_ERROR] {e}")
                    time.sleep(0.3)
                    bad_streak = 0


class CO2GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CO₂ ESP32 - GUI + Simulador (suavizado + derivadas)")
        self.geometry("1450x960")

        # Estado
        self.connected = False
        self.use_phantom = tk.BooleanVar(value=False)
        self.autoscroll_var = tk.BooleanVar(value=True)

        self.serman = SerialManager(self._on_serial_line)
        self.phantom = PhantomSensor(self._on_serial_line)

        self.acquiring = False
        self.out_path = None
        self.t_vals, self.ppm_vals = [], []
        self.msg_queue = queue.Queue()
        self.data_queue = queue.Queue()

        # GUI vars
        self.port_var = tk.StringVar()
        self.folder_var = tk.StringVar(value=os.getcwd())
        self.filename_var = tk.StringVar()
        self.duration_var = tk.StringVar(value="900")
        self.fi_var = tk.StringVar(value="100.0")
        self.delta_var = tk.StringVar(value="0.5")
        self.d_var = tk.StringVar(value="10")
        self.fmin_var = tk.StringVar(value="0.5")
        self.fmax_var = tk.StringVar(value="5000")
        self.current_freq_var = tk.StringVar(value="—")

        # Suavizado
        self.smooth_method_var = tk.StringVar(value="Ninguno")
        self.smooth_win_var = tk.StringVar(value="0")
        self.sg_order_var = tk.StringVar(value="2")

        # Regla de “overflow simple”
        self.prev_ppm_for_rule = None

        self._build_ui()
        self._refresh_ports()
        if not HAS_SCIPY:
            # Avisa en log si Savitzky–Golay no disponible
            self._append_log("[INFO] scipy no está instalado: Savitzky–Golay deshabilitado.")
        self.after(100, self._process_queues)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- Helpers de suavizado ----------
    @staticmethod
    def _moving_average_centered(y, w):
        """Promedio móvil centrado (convolución same). w>=2."""
        if y.size == 0 or w <= 1:
            return y.copy()
        k = np.ones(int(w), dtype=float) / float(w)
        return np.convolve(y, k, mode="same")

    def _apply_smoothing(self, y_arr):
        """Aplica suavizado según selección de GUI y devuelve y_suavizada."""
        # Lee GUI
        method = self.smooth_method_var.get().strip()
        try:
            w = int(float(self.smooth_win_var.get().strip()))
        except:
            w = 0
        w = max(0, w)

        # Ninguno
        if method == "Ninguno" or w <= 1:
            return y_arr.copy()

        if method.startswith("Promedio"):
            return self._moving_average_centered(y_arr, w)

        if method.startswith("Savitzky"):
            if not HAS_SCIPY:
                # Fallback a promedio móvil si no hay scipy
                return self._moving_average_centered(y_arr, w)
            # Validaciones SG: ventana impar, > orden
            try:
                order = int(float(self.sg_order_var.get().strip()))
            except:
                order = 2
            order = max(0, order)
            # Forzar ventana impar
            if w % 2 == 0:
                w = w + 1
            if w <= order:
                w = order + 1 if (order + 1) % 2 == 1 else order + 2
            # Borde: si ventana > len(y), recorta a el impar máximo posible
            if w >= y_arr.size:
                w = max(3, (y_arr.size // 2) * 2 + 1)  # impar <= N
                if w <= order:
                    w = order + 2 + (order % 2)  # aseg. impar > order
            try:
                return savgol_filter(y_arr, window_length=w, polyorder=order, mode="interp")
            except Exception:
                # Fallback robusto
                return self._moving_average_centered(y_arr, max(3, w|1))

        # Fallback si algo raro
        return y_arr.copy()

    # ---------- UI ----------
    def _build_ui(self):
        # Layout principal 2 columnas: izquierda (plots) y derecha (controles+log)
        main = ttk.Frame(self, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)  # izquierda
        main.columnconfigure(1, weight=1)  # derecha
        main.rowconfigure(0, weight=1)

        # ===== Izquierda: TRES PANELES =====
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0,8))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(7.4,8.0), dpi=100)
        gs = self.fig.add_gridspec(3, 1, hspace=0.25)
        self.ax0 = self.fig.add_subplot(gs[0, 0])
        self.ax1 = self.fig.add_subplot(gs[1, 0], sharex=self.ax0)
        self.ax2 = self.fig.add_subplot(gs[2, 0], sharex=self.ax0)

        self.ax0.set_title("CO₂ (ppm, suavizado) vs Tiempo (s)")
        self.ax0.set_ylabel("CO₂ (ppm)")
        self.ax0.grid(True)
        self.line0, = self.ax0.plot([], [], marker='.', linestyle='-')

        self.ax1.set_title("d(CO₂)/dt (ppm/s) — serie suavizada")
        self.ax1.set_ylabel("ppm/s")
        self.ax1.grid(True)
        self.line1, = self.ax1.plot([], [], marker='.', linestyle='-')

        self.ax2.set_title("d²(CO₂)/dt² (ppm/s²) — serie suavizada")
        self.ax2.set_xlabel("Tiempo (s)")
        self.ax2.set_ylabel("ppm/s²")
        self.ax2.grid(True)
        self.line2, = self.ax2.plot([], [], marker='.', linestyle='-')

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # ===== Derecha: arriba controles, abajo log =====
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=0)  # controles
        right.rowconfigure(1, weight=1)  # log
        right.columnconfigure(0, weight=1)

        # ---- Controles
        controls = ttk.LabelFrame(right, text="Controles")
        controls.grid(row=0, column=0, sticky="new", padx=(8,0), pady=(0,8))
        for c in range(10):
            controls.columnconfigure(c, weight=1)

        # Fila 0: Puerto / Simulación
        ttk.Label(controls, text="COM:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.port_combo = ttk.Combobox(controls, textvariable=self.port_var, state="readonly", width=18)
        self.port_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Button(controls, text="Refrescar", command=self._refresh_ports).grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.use_phantom_chk = ttk.Checkbutton(controls, text="Simulación (sin HW)", variable=self.use_phantom)
        self.use_phantom_chk.grid(row=0, column=3, sticky="w", padx=8, pady=4)
        self.conn_btn = ttk.Button(controls, text="Conectar", command=self._toggle_connection)
        self.conn_btn.grid(row=0, column=4, sticky="w", padx=8, pady=4)

        # Fila 1-2: Archivo
        ttk.Label(controls, text="Archivo (sin ext.):").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        self.filename_entry = ttk.Entry(controls, textvariable=self.filename_var, width=28)
        self.filename_entry.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Button(controls, text="Carpeta…", command=self._choose_folder).grid(row=1, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(controls, text="Guardar en:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Label(controls, textvariable=self.folder_var).grid(row=2, column=1, columnspan=6, sticky="w", padx=4, pady=4)

        # Fila 3: Adquisición
        ttk.Label(controls, text="Duración (s):").grid(row=3, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(controls, textvariable=self.duration_var, width=10).grid(row=3, column=1, sticky="w", padx=4, pady=4)
        self.start_btn = ttk.Button(controls, text="Iniciar adquisición", command=self._start_acq, state="disabled")
        self.start_btn.grid(row=3, column=2, sticky="w", padx=4, pady=4)
        self.stop_btn = ttk.Button(controls, text="Detener", command=self._stop_acq, state="disabled")
        self.stop_btn.grid(row=3, column=3, sticky="w", padx=4, pady=4)
        self.zero_btn = ttk.Button(controls, text="Calibrar cero", command=self._cal_zero, state="disabled")
        self.zero_btn.grid(row=3, column=4, sticky="w", padx=4, pady=4)

        # Fila 4-5: Estimulación
        stim_lbl = ttk.LabelFrame(controls, text="Estimulación")
        stim_lbl.grid(row=4, column=0, columnspan=10, sticky="ew", padx=2, pady=(6,2))
        for c in range(12):
            stim_lbl.columnconfigure(c, weight=1)

        ttk.Label(stim_lbl, text="Fi (Hz):").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(stim_lbl, textvariable=self.fi_var, width=10).grid(row=0, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(stim_lbl, text="Delta (Hz):").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(stim_lbl, textvariable=self.delta_var, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(stim_lbl, text="D (s):").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        ttk.Entry(stim_lbl, textvariable=self.d_var, width=10).grid(row=0, column=5, sticky="w", padx=4, pady=4)
        self.stim_start_btn = ttk.Button(stim_lbl, text="Iniciar", command=self._stim_start, state="disabled")
        self.stim_start_btn.grid(row=0, column=6, sticky="w", padx=6, pady=4)
        self.stim_stop_btn = ttk.Button(stim_lbl, text="Detener", command=self._stim_stop, state="disabled")
        self.stim_stop_btn.grid(row=0, column=7, sticky="w", padx=6, pady=4)

        ttk.Label(stim_lbl, text="fmin (Hz):").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(stim_lbl, textvariable=self.fmin_var, width=10).grid(row=1, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(stim_lbl, text="fmax (Hz):").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(stim_lbl, textvariable=self.fmax_var, width=12).grid(row=1, column=3, sticky="w", padx=4, pady=4)
        self.limits_btn = ttk.Button(stim_lbl, text="Fijar límites", command=self._stim_set_limits, state="disabled")
        self.limits_btn.grid(row=1, column=4, sticky="w", padx=6, pady=4)

        ttk.Label(stim_lbl, text="Frecuencia actual:").grid(row=1, column=6, sticky="e", padx=4, pady=4)
        ttk.Label(stim_lbl, textvariable=self.current_freq_var).grid(row=1, column=7, sticky="w", padx=4, pady=4)

        # ===== Suavizado =====
        smooth_frame = ttk.LabelFrame(controls, text="Suavizado")
        smooth_frame.grid(row=6, column=0, columnspan=10, sticky="ew", padx=2, pady=(10,2))
        for c in range(10):
            smooth_frame.columnconfigure(c, weight=1)

        ttk.Label(smooth_frame, text="Método:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        methods = ["Ninguno", "Promedio móvil (centrado)"]
        if HAS_SCIPY:
            methods.append("Savitzky–Golay")
        self.smooth_method_combo = ttk.Combobox(smooth_frame, textvariable=self.smooth_method_var,
                                                values=methods, state="readonly", width=28)
        self.smooth_method_combo.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(smooth_frame, text="Ventana (puntos):").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        ttk.Entry(smooth_frame, textvariable=self.smooth_win_var, width=8).grid(row=0, column=3, sticky="w", padx=4, pady=4)
        ttk.Label(smooth_frame, text="Orden (SG):").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        ttk.Entry(smooth_frame, textvariable=self.sg_order_var, width=6).grid(row=0, column=5, sticky="w", padx=4, pady=4)
        ttk.Label(smooth_frame, text="(SG requiere ventana impar y > orden)").grid(row=0, column=6, columnspan=3, sticky="w", padx=4, pady=4)

        # ---- Log (abajo a la derecha)
        log_frame = ttk.LabelFrame(right, text="Log de valores y mensajes")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=(8,0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="none", height=14)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=yscroll.set)

        ttk.Checkbutton(right, text="Auto-scroll", variable=self.autoscroll_var).grid(
            row=2, column=0, sticky="e", padx=(8,0), pady=(6,0)
        )

    # ---------- UI helpers ----------
    def _append_log(self, s):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", s + "\n")
        if self.autoscroll_var.get():
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports and (not self.port_var.get() or self.port_var.get() not in ports):
            self.port_var.set(ports[0] if ports else "")
        elif not ports:
            self.port_var.set("")

    def _choose_folder(self):
        folder = filedialog.askdirectory(
            initialdir=self.folder_var.get() or os.getcwd(),
            title="Elegir carpeta de destino"
        )
        if folder:
            self.folder_var.set(folder)

    # ---------- Conectar ----------
    def _toggle_connection(self):
        if not self.connected:
            if self.use_phantom.get():
                res = self.phantom.connect()
            else:
                port = self.port_var.get().strip()
                if not port:
                    return messagebox.showerror("Error", "Seleccione un puerto COM.")
                res = self.serman.connect(port)
            if res is True:
                self.connected = True
                self.conn_btn.config(text="Desconectar")
                self.start_btn.config(state="normal")
                self.stop_btn.config(state="normal")
                self.zero_btn.config(state="normal")
                self.stim_start_btn.config(state="normal")
                self.stim_stop_btn.config(state="normal")
                self.limits_btn.config(state="normal")
                self._stim_set_limits()
                self._append_log("[OK] Conectado (simulador)" if self.use_phantom.get() else f"[OK] Conectado @ {BAUD} baud")
                if not HAS_SCIPY:
                    self._append_log("[INFO] Savitzky–Golay no disponible (instale scipy).")
            else:
                messagebox.showerror("Error", str(res))
        else:
            if self.use_phantom.get():
                self.phantom.disconnect()
            else:
                self.serman.disconnect()
            self.connected = False
            self.conn_btn.config(text="Conectar")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="disabled")
            self.zero_btn.config(state="disabled")
            self.stim_start_btn.config(state="disabled")
            self.stim_stop_btn.config(state="disabled")
            self.limits_btn.config(state="disabled")
            self._append_log("[OK] Desconectado")

    # ---------- Comandos ----------
    def _send_cmd(self, s, log_prefix=">>"):
        if not self.connected:
            return messagebox.showerror("Error", "No conectado.")
        ok = self.phantom.sendln(s) if self.use_phantom.get() else self.serman.sendln(s)
        if ok:
            self._append_log(f"{log_prefix} {s}")
        else:
            self._append_log(f"[ERROR] No se pudo enviar: {s}")

    def _cal_zero(self):
        self._send_cmd("Z", ">> Calibración cero")

    def _stim_set_limits(self):
        try:
            fmin = float(self.fmin_var.get().strip())
            fmax = float(self.fmax_var.get().strip())
            if not (fmin > 0 and fmax > fmin):
                raise ValueError
        except:
            return messagebox.showerror("Error", "Límites inválidos (fmin>0, fmax>fmin).")
        self._send_cmd(f"L {fmin} {fmax}", ">> Límites")

    def _stim_start(self):
        try:
            fi = float(self.fi_var.get().strip())
            dv = float(self.delta_var.get().strip())
            dsec = float(self.d_var.get().strip())
            if fi <= 0 or dsec <= 0:
                raise ValueError
        except:
            return messagebox.showerror("Error", "Parámetros inválidos (Fi>0, D>0).")
        self._send_cmd(f"S {fi} {dv} {dsec}", ">> Estimulación iniciar")

    def _stim_stop(self):
        self._send_cmd("X", ">> Estimulación detener")

    def _build_metadata_header(self):
        """Metadatos para registrar en el archivo de salida (líneas comentadas)."""
        from datetime import datetime
        method = self.smooth_method_var.get().strip()
        win_txt = self.smooth_win_var.get().strip()
        order_txt = self.sg_order_var.get().strip()
        lines = []
        lines.append(f"# Generated: {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"# Smoothing method: {method}")
        lines.append(f"# Smoothing window (points): {win_txt}")
        if method.startswith("Savitzky"):
            lines.append(f"# Savitzky–Golay order: {order_txt}")
        lines.append(f"# Columns: {HEADER}")
        lines.append(f"# Units: Time[s], CO2[ppm], dCO2/dt[ppm/s], d2CO2/dt2[ppm/s^2]")
        return "\n".join(lines) + "\n"

    # ---------- Adquisición ----------
    def _start_acq(self):
        if not self.connected:
            return messagebox.showerror("Error", "No conectado.")
        name = self.filename_var.get().strip()
        if not name:
            return messagebox.showerror("Error", "Ingrese nombre de archivo (sin extensión).")
        if not name.lower().endswith(".txt"):
            name += ".txt"
        folder = self.folder_var.get().strip() or os.getcwd()
        if not os.path.isdir(folder):
            return messagebox.showerror("Error", "La carpeta de destino no existe.")
        self.out_path = os.path.join(folder, name)
        if os.path.exists(self.out_path):
            if not messagebox.askyesno("Archivo existente", f"El archivo '{self.out_path}' ya existe.\n¿Sobrescribirlo?"):
                return
        try:
            self.target_end = float(self.duration_var.get().strip())
            if self.target_end <= 0: raise ValueError
        except:
            return messagebox.showerror("Error", "Duración inválida (>0).")

        # Reset buffers y gráficos
        self.t_vals.clear(); self.ppm_vals.clear()
        self.prev_ppm_for_rule = None
        self.line0.set_data([], [])
        self.line1.set_data([], [])
        self.line2.set_data([], [])
        for ax in (self.ax0, self.ax1, self.ax2):
            ax.relim(); ax.autoscale_view()
        self.canvas.draw()

        # Estados
        self.acquiring = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        # Archivo + encabezado (4 columnas)
        try:
            with open(self.out_path, "w", encoding="utf-8") as f:
                f.write(HEADER + "\n")
        except Exception as e:
            self.acquiring = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            return messagebox.showerror("Error", f"No se pudo abrir archivo: {e}")

        # Arranca el dispositivo/simulador
        self._send_cmd("G", ">> Adquisición")

    def _stop_acq(self):
        self.acquiring = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    # ---------- Regla solicitada ----------
    def _apply_simple_rule(self, ppm_raw: float) -> float:
        """
        Si la lectura anterior > 60000 y la actual < 35000,
        sumar 65532 a la lectura actual y loguear #FIX_APLICADO.
        """
        adj = ppm_raw
        try:
            if (self.prev_ppm_for_rule is not None
                and self.prev_ppm_for_rule > 60000
                and ppm_raw < 35000):
                adj = ppm_raw + 65532.0
                try:
                    self._append_log(f"#FIX_APLICADO prev={self.prev_ppm_for_rule:g} raw={ppm_raw:g} -> adj={adj:g}")
                except Exception:
                    pass
        except Exception:
            pass
        self.prev_ppm_for_rule = adj
        return adj

    # ---------- Callback serie (lector) ----------
    def _on_serial_line(self, line: str):
        if line.startswith("#STIM_F="):
            try:
                val = line.split("=", 1)[1].strip()
                self.current_freq_var.set(val + " Hz")
            except:
                pass
            self.msg_queue.put(line)
            return

        if line.startswith("#STIM_") or line.startswith("#ZERO_CAL_SENT") or line.startswith("#RESYNC_SENSOR") or line.startswith("#BAD_") or line.startswith("#OUT_OF_RANGE"):
            self.msg_queue.put(line)
            return

        if line.strip().startswith("Time["):
            return

        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                t = float(parts[0]); ppm_raw = float(parts[1])
                ppm = self._apply_simple_rule(ppm_raw)

                if self.acquiring and (PPM_MIN <= ppm <= PPM_MAX):
                    self.data_queue.put((t, ppm))
                    self.msg_queue.put(f"{t:g} {ppm:g}")
                    if t >= getattr(self, "target_end", 0.0) > 0:
                        self._stop_acq()
                else:
                    self.msg_queue.put(f"{t:g} {ppm:g}")
                return
            except:
                pass

        self.msg_queue.put(line)

    # ---------- Bucle UI para colas ----------
    def _process_queues(self):
        # Log
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                self._append_log(msg)
        except queue.Empty:
            pass

        # Datos
        need_draw = False
        to_write = []
        try:
            while True:
                t, ppm = self.data_queue.get_nowait()
                self.t_vals.append(t); self.ppm_vals.append(ppm)
                to_write.append((t, ppm))
                need_draw = True
        except queue.Empty:
            pass

        # Guardado con suavizado + derivadas
        if to_write and self.acquiring and self.out_path:
            try:
                t_arr = np.array(self.t_vals, dtype=float)
                y_arr = np.array(self.ppm_vals, dtype=float)

                y_s = self._apply_smoothing(y_arr)

                # Derivadas sobre serie suavizada
                if len(t_arr) >= 3:
                    d1 = np.gradient(y_s, t_arr, edge_order=2)
                    d2 = np.gradient(d1, t_arr, edge_order=2)
                else:
                    d1 = np.full_like(y_s, np.nan, dtype=float)
                    d2 = np.full_like(y_s, np.nan, dtype=float)

                # Índices de los nuevos puntos agregados
                start_idx = len(self.t_vals) - len(to_write)
                with open(self.out_path, "a", encoding="utf-8") as f:
                    for k in range(len(to_write)):
                        idx = start_idx + k
                        f.write(f"{t_arr[idx]:g} {y_s[idx]:g} {d1[idx]:g} {d2[idx]:g}\n")
            except Exception as e:
                self._append_log(f"[ERROR] guardando archivo: {e}")

        # Redibujar (3 paneles)
        if need_draw:
            t_arr = np.array(self.t_vals, dtype=float)
            y_arr = np.array(self.ppm_vals, dtype=float)

            y_s = self._apply_smoothing(y_arr)

            # Señal suavizada
            self.line0.set_data(t_arr, y_s)

            if len(t_arr) >= 3:
                dy_dt = np.gradient(y_s, t_arr, edge_order=2)
                d2y_dt2 = np.gradient(dy_dt, t_arr, edge_order=2)
                self.line1.set_data(t_arr, dy_dt)
                self.line2.set_data(t_arr, d2y_dt2)

                # Auto-escala amable para cada eje
                for ax, arr in ((self.ax0, y_s), (self.ax1, dy_dt), (self.ax2, d2y_dt2)):
                    if len(arr) >= 2:
                        vmin = float(np.nanmin(arr))
                        vmax = float(np.nanmax(arr))
                        if vmin == vmax:
                            vmin -= 1.0; vmax += 1.0
                        pad = 0.1 * (vmax - vmin) if vmax > vmin else 1.0
                        ax.set_ylim(vmin - pad, vmax + pad)
                if len(t_arr) >= 1:
                    self.ax0.set_xlim(t_arr[0], t_arr[-1])

            self.ax0.relim(); self.ax0.autoscale_view(scalex=True, scaley=True)
            self.ax1.relim(); self.ax1.autoscale_view(scalex=True, scaley=True)
            self.ax2.relim(); self.ax2.autoscale_view(scalex=True, scaley=True)
            self.canvas.draw_idle()

        # Si terminó la adquisición, guardar PNG
        if not self.acquiring and self.out_path:
            try:
                png_path = os.path.splitext(self.out_path)[0] + ".png"
                self.fig.savefig(png_path, dpi=300)
                self._append_log(f"Gráfico guardado en: {png_path}")
                self.out_path = None
            except Exception as e:
                self._append_log(f"[ERROR] al guardar PNG: {e}")

        self.after(100, self._process_queues)

    def _on_close(self):
        self.acquiring = False
        try:
            if self.use_phantom.get():
                self.phantom.disconnect()
            else:
                self.serman.disconnect()
        except:
            pass
        self.destroy()


if __name__ == "__main__":
    app = CO2GUI()
    app.mainloop()
