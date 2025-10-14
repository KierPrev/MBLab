#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visor interactivo de CO₂ (levaduras)
- Izquierda: Curvas Time vs CO2_smooth, banda de error opcional ±(50 + 5%) (mismo color)
- Derecha:   Boxplots por curva o por grupo (prefijo antes de '-' o '_')
- Slider (abajo-izq) para elegir ventana temporal del boxplot
- Checks (abajo-der): Centrar en 0 / Mostrar error / Agrupar por prefijo
- Botón (abajo-der): Agregar archivos
- Perm-test entre grupos (p-valor) sobre medias por archivo (si hay ≥2 archivos/grupo)

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
from matplotlib.widgets import CheckButtons, Button, RangeSlider

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
def plotear_archivos(rutas, mostrar_error=False, salida=None, iniciar_centrado=False,
                     ui_scale=0.62, compact=False):
    figsize = apply_rcparams(ui_scale, compact)

    # Layout lado a lado: [curvas | boxplots]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.6, 1.0])
    ax_curv = fig.add_subplot(gs[0, 0])  # IZQ
    ax_box  = fig.add_subplot(gs[0, 1])  # DER

    # Márgenes y espacio inferior para controles
    plt.subplots_adjust(bottom=0.23 if not compact else 0.20, wspace=0.28 if not compact else 0.24)

    ax_curv.set_title("CO₂ vs tiempo  •  A: agregar  |  S: guardar  |  Z: centrar 0  |  Q: salir")
    ax_curv.set_xlabel("Tiempo [s]"); ax_curv.set_ylabel("CO₂ [ppm]")
    ax_curv.grid(True, alpha=0.25)

    datasets = []   # {"x","y","y0","label","line","err","fill","color"}
    rutas_agregadas = []
    rng = np.random.default_rng()

    state = {"centrar0": bool(iniciar_centrado),
             "mostrar_error": bool(mostrar_error),
             "group_by_prefix": False,
             "tmin": None, "tmax": None}

    # ---- helpers ----
    def first_finite(a, default=np.nan):
        for v in a:
            if np.isfinite(v): return float(v)
        return default

    def y_plot_from(d):
        y = d["y"]
        return (y - d["y0"]) if (state["centrar0"] and np.isfinite(d["y0"])) else y

    def dibujar_fill(ax, d, yplot):
        if not state["mostrar_error"] or d["err"] is None:
            return None
        return ax.fill_between(d["x"], yplot - d["err"], yplot + d["err"],
                               alpha=0.18, linewidth=0, color=d["color"], label="_nolegend_")

    def actualizar_ylabel():
        ax_curv.set_ylabel("ΔCO₂ [ppm]" if state["centrar0"] else "CO₂ [ppm]")
        ax_box.set_ylabel("ΔCO₂ [ppm]" if state["centrar0"] else "CO₂ [ppm]")

    def extraer_valores_en_ventana(d):
        x = d["x"]; y = y_plot_from(d)
        if state["tmin"] is None or state["tmax"] is None:
            return y
        mask = (x >= state["tmin"]) & (x <= state["tmax"])
        return y[mask]

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

            err = error_modelo(df["CO2_ppm"]) if state["mostrar_error"] else None
            fill = dibujar_fill(ax_curv, {"x":x,"err":err,"color":color,"y":y,"y0":y0}, yplot)

            datasets.append({"x":x,"y":y,"y0":y0,"label":label,"line":ln,"err":err,"fill":fill,"color":color})
            rutas_agregadas.append(str(p)); nuevos_ok += 1

        if datasets and nuevos_ok:
            ax_curv.legend(loc="best", frameon=True, framealpha=0.95)
        actualizar_ylabel()
        ax_curv.relim(); ax_curv.autoscale()
        fig.canvas.draw_idle()
        actualizar_slider_limits()
        actualizar_boxplots()

    # ---- slider tiempo (abajo-izquierda) ----
    ax_slider = fig.add_axes([0.10, 0.13, 0.55, 0.028 if compact else 0.032])
    slider = None
    def actualizar_slider_limits():
        nonlocal slider
        if not datasets: return
        tmin = min(float(d["x"][0]) for d in datasets if d["x"].size)
        tmax = max(float(d["x"][-1]) for d in datasets if d["x"].size)
        if tmin == tmax: tmax = tmin + 1.0
        if slider is None:
            init = (tmin, tmax)
            slider = RangeSlider(ax_slider, "Ventana [s]", tmin, tmax, valinit=init)
            state["tmin"], state["tmax"] = init
            slider.on_changed(on_slider_change)
        else:
            old = slider.val
            slider.valmin = tmin; slider.valmax = tmax
            slider.ax.set_xlim(tmin, tmax)
            v0 = max(tmin, min(old[0], tmax-1e-9))
            v1 = max(tmin+1e-9, min(old[1], tmax))
            slider.set_val((v0, v1))
        fig.canvas.draw_idle()

    def on_slider_change(val):
        state["tmin"], state["tmax"] = float(val[0]), float(val[1])
        actualizar_boxplots()

    # ---- boxplots (derecha) ----
    stats_annot = None
    warn_annot  = None

    def actualizar_boxplots():
        nonlocal stats_annot, warn_annot
        ax_box.clear()
        ax_box.set_xlabel("Curvas / Grupos")
        ax_box.grid(True, axis="y", alpha=0.23)

        if not datasets:
            ax_box.set_title("Boxplots • (sin datos)")
            fig.canvas.draw_idle(); return

        if state["group_by_prefix"]:
            groups = {}; group_colors = {}
            for d in datasets:
                g = infer_group(d["label"])
                groups.setdefault(g, []).append(d)
                group_colors.setdefault(g, d["color"])
            labels, data, colors = [], [], []
            for g, lst in groups.items():
                vals = np.concatenate([extraer_valores_en_ventana(dd)
                                       for dd in lst if extraer_valores_en_ventana(dd).size])
                if vals.size == 0: continue
                labels.append(g); data.append(vals); colors.append(group_colors[g])
            ax_box.set_title("Distribución en ventana • Agrupado por prefijo")
        else:
            labels = [d["label"] for d in datasets]
            data   = [extraer_valores_en_ventana(d) for d in datasets]
            colors = [d["color"] for d in datasets]
            ax_box.set_title("Distribución en ventana • Por curva")

        kept = [(lab, vals, col) for lab, vals, col in zip(labels, data, colors) if np.asarray(vals).size > 0]
        if not kept:
            ax_box.text(0.5, 0.5, "Sin datos en la ventana", ha="center", va="center", transform=ax_box.transAxes)
            fig.canvas.draw_idle(); return

        labels, data, colors = zip(*kept)
        bp = ax_box.boxplot(data, vert=True, patch_artist=True, labels=labels, showfliers=False)
        for patch, col in zip(bp['boxes'], colors):
            patch.set_facecolor(col); patch.set_alpha(0.25); patch.set_edgecolor(col)
        for med in bp['medians']:
            med.set_color("black"); med.set_linewidth(1.3)

        ax_box.set_ylabel("ΔCO₂ [ppm]" if state["centrar0"] else "CO₂ [ppm]")

        # ---- anotaciones dentro del panel derecho ----
        if stats_annot: stats_annot.remove(); stats_annot = None
        if warn_annot:  warn_annot.remove();  warn_annot  = None

        if state["group_by_prefix"]:
            per_group_means = {}
            for g in set(labels): per_group_means[g] = []
            for d in datasets:
                g = infer_group(d["label"])
                if g not in per_group_means: continue
                vals = extraer_valores_en_ventana(d)
                if vals.size: per_group_means[g].append(np.mean(vals))
            groups = [g for g in per_group_means if len(per_group_means[g])>0]
            lines = []
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    gi, gj = groups[i], groups[j]
                    xi, xj = per_group_means[gi], per_group_means[gj]
                    if len(xi)>=2 and len(xj)>=2:
                        p, diff = permutation_pvalue_two_groups(xi, xj, n_perm=2000, rng=rng)
                        lines.append(f"{gi} vs {gj}: p={p:.3g}  (Δμ={diff:.1f} ppm)")
                    else:
                        lines.append(f"{gi} vs {gj}: insuf. réplicas")

            if lines:
                txt = "Perm-test (medias por archivo):\n" + "\n".join(lines)
                stats_annot = ax_box.text(0.02, 0.98, txt, transform=ax_box.transAxes,
                                          va="top", ha="left", fontsize=max(8, 8.5*ui_scale))
            warn_annot = ax_box.text(0.02, 0.02, "Nota: requiere ≥2 archivos por grupo.",
                                     transform=ax_box.transAxes, va="bottom", ha="left",
                                     fontsize=max(8, 8.5*ui_scale))
        else:
            warn_annot = ax_box.text(0.02, 0.02,
                                     "Nota: cajas con puntos en el tiempo como 'réplicas' (visual, no inferencia).",
                                     transform=ax_box.transAxes, va="bottom", ha="left",
                                     fontsize=max(8, 8.5*ui_scale))

        fig.canvas.draw_idle()

    def replot_curvas():
        for d in datasets:
            yplot = y_plot_from(d)
            d["line"].set_ydata(yplot)
            if state["mostrar_error"]:
                if d["err"] is None: d["err"] = error_modelo(d["y"])
                if d["fill"] is not None:
                    try: d["fill"].remove()
                    except Exception: pass
                d["fill"] = dibujar_fill(ax_curv, d, yplot)
            else:
                if d["fill"] is not None:
                    try: d["fill"].remove()
                    except Exception: pass
                    d["fill"] = None
        actualizar_ylabel()
        ax_curv.relim(); ax_curv.autoscale()
        fig.canvas.draw_idle()
        actualizar_boxplots()

    # ----- Carga inicial -----
    if rutas:
        agregar_curvas(rutas)

    # ----- Controles abajo (derecha) -----
    ax_checks = fig.add_axes([0.70, 0.13, 0.25, 0.10 if not compact else 0.09])
    check_labels = ["Centrar en 0", "Mostrar error", "Agrupar por prefijo"]
    check_states = [state["centrar0"], state["mostrar_error"], state["group_by_prefix"]]
    checks = CheckButtons(ax_checks, check_labels, check_states)
    for txt in checks.labels:
        txt.set_fontsize(max(8, 11*ui_scale))

    def on_check(label):
        status = checks.get_status()
        state["centrar0"] = bool(status[0])
        state["mostrar_error"] = bool(status[1])
        state["group_by_prefix"] = bool(status[2])
        replot_curvas()

    checks.on_clicked(on_check)

    # Botón Agregar (abajo-der)
    ax_btn_add = fig.add_axes([0.70, 0.05, 0.25, 0.07 if not compact else 0.065])
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

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser(description="Curvas + boxplots lado a lado para CO₂ de levaduras.")
    ap.add_argument("files", nargs="*", help="Archivos iniciales (opcional).")
    ap.add_argument("--error", action="store_true", help="Iniciar con banda de error activada.")
    ap.add_argument("--save", metavar="PNG", help="Guardar imagen con nombre específico.")
    ap.add_argument("--normalize0", action="store_true", help="Iniciar centrado en 0.")
    ap.add_argument("--ui-scale", type=float, default=None, help="Escala de UI (ej. 0.60).")
    ap.add_argument("--compact", action="store_true", help="Forzar modo compacto (más chico).")
    args = ap.parse_args()

    scale = compute_ui_scale(args.ui_scale, args.compact)
    plotear_archivos(args.files, mostrar_error=args.error, salida=args.save,
                     iniciar_centrado=args.normalize0, ui_scale=scale, compact=args.compact)

if __name__ == "__main__":
    main()

