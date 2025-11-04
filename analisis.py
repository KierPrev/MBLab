#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visor interactivo de CO₂ (levaduras)
- Curvas Time vs CO2_smooth con bandas de error ±(50 + 5%) (mismo color)
- Check (abajo): Centrar en 0
- Botón (abajo): Agregar archivos

Escala de UI:
- Auto para DPI/2K; forzá con --ui-scale 0.60 o --compact para más chico
"""

from pathlib import Path
import argparse
import datetime as dt
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button

# -------------------- UI adaptativa (HiDPI / 2K) --------------------
def compute_ui_scale(user_scale: float | None, compact: bool) -> float:
    if user_scale is not None:
        s = max(0.4, min(2.0, float(user_scale)))
        return min(s, 0.65 if compact else s)
    w = h = dpi = None
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        dpi = float(root.winfo_fpixels('1i'))
        root.destroy()
    except Exception:
        pass
    f_dpi = (96.0 / dpi) if dpi else 1.0
    f_w   = (1920.0 / w)  if w else 1.0
    base = min(1.0, max(0.55, 0.9 * min(f_dpi**0.5, f_w)))
    if w and w >= 2300:
        base = min(base, 0.62)
    if compact:
        base = min(base, 0.58)
    return base

def apply_rcparams(scale: float, compact: bool):
    base = {"font":12,"title":15,"label":13,"legend":10.5,"tick":11,
            "lw":1.8,"ms":4.6,"dpi":140,"save_dpi":300,"fig_w":10.6,"fig_h":5.8}
    if compact:
        base.update({"font":11,"title":13,"label":12,"legend":10,"tick":10,
                     "lw":1.6,"ms":4.0,"fig_w":10.0,"fig_h":5.3})
    f = scale
    plt.rcParams.update({
        "figure.dpi":     int(base["dpi"] / max(0.75, f)),
        "savefig.dpi":    base["save_dpi"],
        "font.size":      max(8,  base["font"]  * f),
        "axes.titlesize": max(9,  base["title"] * f),
        "axes.labelsize": max(8,  base["label"] * f),
        "legend.fontsize":max(8,  base["legend"]* f),
        "xtick.labelsize":max(8,  base["tick"]  * f),
        "ytick.labelsize":max(8,  base["tick"]  * f),
        "lines.linewidth":max(1.2,base["lw"]   * f),
        "lines.markersize":max(3.0,base["ms"]  * f),
    })
    return (base["fig_w"] * f, base["fig_h"] * f)

# -------------------- IO --------------------
def leer_archivo(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    try:
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#",
                         usecols=[0,1], header=0, names=["Time_s","CO2_ppm"])
        if not np.issubdtype(df["Time_s"].dtype, np.number):
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python", comment="#",
                         usecols=[0,1], header=None, names=["Time_s","CO2_ppm"])
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["Time_s","CO2_ppm"])
    df = df.sort_values("Time_s").reset_index(drop=True)
    return df

def error_modelo(co2_ppm) -> np.ndarray:
    return 50.0 + 0.05 * np.asarray(co2_ppm, dtype=float)

def infer_group(label: str) -> str:
    m = re.split(r"[-_]", label)
    return (m[0] if m else label).lower()

# -------------------- Stats --------------------
def permutation_pvalue_two_groups(x, y, n_perm=2000, rng=None):
    rng = rng or np.random.default_rng()
    x = np.asarray(x, float); y = np.asarray(y, float)
    obs = abs(x.mean() - y.mean())
    pooled = np.concatenate([x, y]); nx = len(x); count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(pooled[:nx].mean() - pooled[nx:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1), obs

# -------------------- App --------------------
def plotear_archivos(rutas, salida=None, iniciar_centrado=False,
                      ui_scale=0.62, compact=False):
    figsize = apply_rcparams(ui_scale, compact)

    # Layout: solo curvas
    fig = plt.figure(figsize=figsize)
    ax_curv = fig.add_subplot(111)

    # Márgenes y espacio inferior para controles
    plt.subplots_adjust(bottom=0.23 if not compact else 0.20)

    ax_curv.set_xlabel("Tiempo [s]")
    ax_curv.grid(True, alpha=0.25)

    ax_box = None  # No hay boxplots

    datasets = []   # {"x","y","y0","label","line","err","fill","color"}
    rutas_agregadas = []
    rng = np.random.default_rng()

    state = {"centrar0": bool(iniciar_centrado), "violin_mode": False}

    # ---- helpers ----
    def first_finite(a, default=np.nan):
        for v in a:
            if np.isfinite(v): return float(v)
        return default

    def y_plot_from(d):
        y = d["y"]
        return (y - d["y0"] + 400) if (state["centrar0"] and np.isfinite(d["y0"])) else y

    def dibujar_fill(ax, d, yplot):
        if d["err"] is None:
            return None
        return ax.fill_between(d["x"], yplot - d["err"], yplot + d["err"],
                               alpha=0.18, linewidth=0, color=d["color"], label="_nolegend_")

    def actualizar_title():
        mode = "Violín" if state["violin_mode"] else "Curvas"
        ax_curv.set_title(f"CO₂ {mode} vs tiempo  •  A: agregar  |  S: guardar  |  Z: centrar 0  |  Q: salir")

    def actualizar_ylabel():
        if state["violin_mode"]:
            ax_curv.set_ylabel("Distribución de CO₂ [ppm]" + (" (ajustado)" if state["centrar0"] else ""))
        else:
            ax_curv.set_ylabel("CO₂ [ppm]" + (" (ajustado)" if state["centrar0"] else ""))

    # Función eliminada: extraer_valores_en_ventana

    # ---- agregar curvas ----
    def agregar_curvas(lista_paths):
        nonlocal rutas_agregadas
        nuevos_ok = 0
        for p in lista_paths:
            try:
                df = leer_archivo(p)
            except Exception as e:
                print(f"[!] No pude leer {p}: {e}")
                continue

            x = df["Time_s"].to_numpy(float)
            y = df["CO2_ppm"].to_numpy(float)
            y0 = first_finite(y)
            label = Path(p).stem

            yplot = (y - y0) if state["centrar0"] and np.isfinite(y0) else y
            (ln,) = ax_curv.plot(x, yplot, marker="o", label=label)
            color = ln.get_color()

            err = error_modelo(df["CO2_ppm"])
            fill = dibujar_fill(ax_curv, {"x":x,"err":err,"color":color,"y":y,"y0":y0}, yplot)

            datasets.append({"x":x,"y":y,"y0":y0,"label":label,"line":ln,"err":err,"fill":fill,"color":color})
            rutas_agregadas.append(str(p)); nuevos_ok += 1

        # Remove legend
        # if datasets and nuevos_ok:
        #     ax_curv.legend(loc="best", frameon=True, framealpha=0.95)
        actualizar_ylabel()
        actualizar_title()
        ax_curv.relim(); ax_curv.autoscale()
        fig.canvas.draw_idle()

    # Slider eliminado

    # Boxplots eliminados

    def replot_curvas():
        ax_curv.clear()
        ax_curv.set_xlabel("Tiempo [s]")
        ax_curv.grid(True, alpha=0.25)
        actualizar_title()
        actualizar_ylabel()

        if state["violin_mode"]:
            # Violin plot mode
            positions = []
            data = []
            labels = []
            colors = []
            for i, d in enumerate(datasets):
                yplot = y_plot_from(d)
                positions.append(i)
                data.append(yplot)
                labels.append(d["label"])
                colors.append(d["color"])
            ax_curv.violinplot(data, positions=positions, showmeans=True, showmedians=True)
            ax_curv.set_xticks(positions)
            ax_curv.set_xticklabels(labels, rotation=45, ha='right')
        else:
            # Curve mode
            for d in datasets:
                yplot = y_plot_from(d)
                d["line"] = ax_curv.plot(d["x"], yplot, marker="o", color=d["color"], label=d["label"])[0]
                if d["fill"] is not None:
                    try: d["fill"].remove()
                    except Exception: pass
                d["fill"] = dibujar_fill(ax_curv, d, yplot)

        ax_curv.relim(); ax_curv.autoscale()
        fig.canvas.draw_idle()

    # ----- Carga inicial -----
    if rutas:
        agregar_curvas(rutas)
    actualizar_title()

    # ----- Controles abajo -----
    ax_checks = fig.add_axes([0.10, 0.13, 0.80, 0.10 if not compact else 0.09])
    check_labels = ["Centrar en 0", "Modo violín"]
    check_states = [state["centrar0"], state["violin_mode"]]
    checks = CheckButtons(ax_checks, check_labels, check_states)
    for txt in checks.labels:
        txt.set_fontsize(max(8, 11*ui_scale))

    def on_check(label):
        status = checks.get_status()
        state["centrar0"] = bool(status[0])
        state["violin_mode"] = bool(status[1])
        replot_curvas()

    checks.on_clicked(on_check)

    # Botón Agregar (abajo)
    ax_btn_add = fig.add_axes([0.10, 0.05, 0.80, 0.07 if not compact else 0.065])
    btn_add = Button(ax_btn_add, "Agregar archivos")

    def seleccionar_archivos():
        nuevos = []
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw()
            seleccion = filedialog.askopenfilenames(
                title="Elegí archivos de CO₂",
                filetypes=[("Texto/CSV", "*.txt *.dat *.csv *.tsv"), ("Todos", "*.*")]
            )
            root.update(); root.destroy()
            nuevos = list(seleccion)
        except Exception:
            print("Pegá rutas separadas por coma y Enter:")
            entrada = input("> ").strip()
            if entrada:
                nuevos = [s.strip() for s in entrada.split(",") if s.strip()]
        return [n for n in nuevos if n not in rutas_agregadas]

    def on_btn_add(event):
        nuevos = seleccionar_archivos()
        if not nuevos:
            print("(No hay archivos nuevos para agregar.)"); return
        agregar_curvas(nuevos)

    btn_add.on_clicked(on_btn_add)

    # Teclas
    def on_key(event):
        key = (event.key or "").lower()
        if key == "a":
            nuevos = seleccionar_archivos()
            if nuevos: agregar_curvas(nuevos)
            else: print("(No hay archivos nuevos para agregar.)")
        elif key == "s":
            out = Path(salida) if salida else Path(f"co2_plot_{dt.datetime.now():%Y%m%d-%H%M%S}.png")
            fig.savefig(out, bbox_inches="tight"); print(f"[✓] Figura guardada en: {out.resolve()}")
        elif key == "z":
            checks.set_active(0)  # toggle centrado
        elif key == "q":
            plt.close(fig)

    # Hover functionality
    annot = ax_curv.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                             bbox=dict(boxstyle="round", fc="w"), fontsize=9)
    annot.set_visible(False)

    def update_annot(line, ind):
        x, y = line.get_xdata()[ind["ind"][0]], line.get_ydata()[ind["ind"][0]]
        annot.xy = (x, y)
        # Find the dataset label and color
        for d in datasets:
            if d["line"] is line:
                text = d["label"]
                color = d["color"]
                break
        else:
            text = "Unknown"
            color = "black"
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor(color)
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax_curv:
            for d in datasets:
                cont, ind = d["line"].contains(event)
                if cont:
                    update_annot(d["line"], ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Curvas para CO₂ de levaduras.")
    ap.add_argument("files", nargs="*", help="Archivos iniciales (opcional).")
    ap.add_argument("--folder", default="experimentos", help="Carpeta con archivos de datos (default: experimentos).")
    # Bandas de error siempre activadas
    ap.add_argument("--save", metavar="PNG", help="Guardar imagen con nombre específico.")
    ap.add_argument("--normalize0", action="store_true", help="Iniciar centrado en 0.")
    ap.add_argument("--ui-scale", type=float, default=None, help="Escala de UI (ej. 0.60).")
    ap.add_argument("--compact", action="store_true", help="Forzar modo compacto (más chico).")
    args = ap.parse_args()

    # Si no se pasan archivos, buscar en la carpeta especificada
    if not args.files:
        folder = Path(args.folder)
        if folder.exists() and folder.is_dir():
            args.files = [str(p) for p in folder.glob("*.txt") if p.is_file()]
        else:
            print(f"[!] Carpeta '{args.folder}' no existe o no es un directorio.")
            args.files = []

    scale = compute_ui_scale(args.ui_scale, args.compact)
    plotear_archivos(args.files, salida=args.save,
                     iniciar_centrado=args.normalize0, ui_scale=scale, compact=args.compact)

if __name__ == "__main__":
    main()

