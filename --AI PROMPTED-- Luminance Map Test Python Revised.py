# elqd_sim_no_overlap.py
# ELQD foveated simulator — final non-overlapping labels version
# Controls:
# 1 Uniform, 2 Conservative, 3 Mild, 4 Strong
# ]/[ change PPD (window = HFOV*PPD, VFOV*PPD)
# J decrease TILE_SIZE (more tiles), K increase TILE_SIZE (fewer tiles)
# V regenerate overlays, G toggle savings-only overlay
# D debug print raw & scaled table, P save overlays, E save frame, L log, m toggle mapping, s toggle gamma signal
# ESC quit

import os
import sys
import time
import csv
import math
import numpy as np
import pygame
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Human FOV and sim params ----------
HFOV = 220.0
VFOV = 150.0

PPD = 8
PPD_MAX = 128
LMAX = 1000.0
MIN_L = 10.0
P_OVERHEAD = 0.20
GAMMA = 2.2

render_scale = 0.35
TILE_SIZE = 2
TARGET_FPS = 60

FOVEA_DEG = 10.0
UNIFORM_REF_W = 100.0  # forced baseline

# ---------- IVL proxy (synthetic) ----------
iv_l_vals = np.array([1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0])
iv_i_vals_mA = np.array([0.005, 0.02, 0.05, 0.25, 0.85, 2.2, 6.0, 18.0, 40.0])
iv_v_vals = np.array([2.2, 2.25, 2.30, 2.45, 2.55, 2.70, 2.95, 3.10, 3.25])

I_from_L = interp1d(iv_l_vals, iv_i_vals_mA / 1000.0, kind="linear",
                    bounds_error=False, fill_value=(iv_i_vals_mA[0] / 1000.0, iv_i_vals_mA[-1] / 1000.0))
V_from_I = interp1d(iv_i_vals_mA / 1000.0, iv_v_vals, kind="linear",
                    bounds_error=False, fill_value=(iv_v_vals[0], iv_v_vals[-1]))

# ---------- Presets (Uniform first) ----------
PRESETS = {
    1: dict(name="Uniform",      PER_DEG_REDUCTION=0.0),
    2: dict(name="Conservative", PER_DEG_REDUCTION=0.0040),
    3: dict(name="Mild",         PER_DEG_REDUCTION=0.0063),
    4: dict(name="Strong",       PER_DEG_REDUCTION=0.0075),
}
current_preset = 3  # default to Mild

# ---------- Sweep settings ----------
SWEEP_MIN = 0.0
SWEEP_MAX = 0.030
SWEEP_STEP = 0.0005

# ---------- Overlay filenames ----------
IVL_PNG = "overlay_ivl_vs_ecc.png"
BAR_PNG = "overlay_watt_bars.png"
SAVINGS_PNG = "overlay_savings_scatter.png"

# ---------- Window helpers ----------
def window_size_exact(ppd):
    return int(round(HFOV * ppd)), int(round(VFOV * ppd))

WINDOW_W, WINDOW_H = window_size_exact(PPD)

# ---------- Pygame init ----------
pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
pygame.display.set_caption("ELQD Simulator , final (no overlap)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

# ---------- Internal grid helpers ----------
def compute_internal_sizes(scale, tile_size, w, h):
    rw = max(4, int(round(w * scale)))
    rh = max(4, int(round(h * scale)))
    cols = max(2, rw // tile_size)
    rows = max(2, rh // tile_size)
    return rows, cols, rw, rh

def make_centers(rows, cols, rw, rh):
    px = (np.arange(cols) + 0.5) * (rw / cols)
    py = (np.arange(rows) + 0.5) * (rh / rows)
    return np.meshgrid(px, py)

rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
px_centers, py_centers = make_centers(rows, cols, rw, rh)

# ---------- Geometry and eccentricity ----------
def pixel_coords_to_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd):
    x_win = px_centers * (win_w / px_centers.max())
    y_win = py_centers * (win_h / py_centers.max())
    x_deg = (x_win - gaze_x) * (1.0 / ppd)
    y_deg = (y_win - gaze_y) * (1.0 / ppd)
    return x_deg, y_deg

def elliptical_ecc_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd):
    x_deg, y_deg = pixel_coords_to_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd)
    a = HFOV / 2.0
    b = VFOV / 2.0
    d_norm = np.sqrt((x_deg / a) ** 2 + (y_deg / b) ** 2)
    eff_deg = d_norm * max(a, b)
    return eff_deg

def elliptical_fovea_mask(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd):
    x_deg, y_deg = pixel_coords_to_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd)
    scale_x = HFOV / max(HFOV, VFOV)
    scale_y = VFOV / max(HFOV, VFOV)
    a_fovea = FOVEA_DEG * scale_x
    b_fovea = FOVEA_DEG * scale_y
    mask = ((x_deg / a_fovea) ** 2 + (y_deg / b_fovea) ** 2) <= 1.0
    return mask

# ---------- Luminance mapping ----------
def luminance_from_edge_offset(edge_offset_deg, per_deg_reduction, min_l):
    L = LMAX * np.maximum(0.0, 1.0 - per_deg_reduction * edge_offset_deg)
    return np.clip(L, min_l, LMAX)

def luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, win_w, win_h, ppd, per_deg):
    if per_deg == 0.0:
        L = np.full(px_centers.shape, LMAX, dtype=float)
    else:
        if mode == "elliptical":
            ecc = elliptical_ecc_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd)
        else:
            x_deg, y_deg = pixel_coords_to_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd)
            ecc = np.hypot(x_deg, y_deg)
        edge_offset = np.maximum(0.0, ecc - FOVEA_DEG)
        L = luminance_from_edge_offset(edge_offset, per_deg, MIN_L)
    mask = elliptical_fovea_mask(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd)
    L[mask] = LMAX
    return L

# ---------- Rendering ----------
def luminance_to_rgb_tiles(L, show_signal=False, min_l=MIN_L, uniform=False):
    if uniform:
        t = np.ones_like(L, dtype=float)
    else:
        denom = (LMAX - min_l) if (LMAX - min_l) > 0 else 1.0
        t = np.clip((L - min_l) / denom, 0.0, 1.0)
        if show_signal:
            t = np.power(t, 1.0 / GAMMA)
    gray = (t * 255).astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=2)
    return rgb

# ---------- Power model ----------
def power_from_luminance_map(Lmap):
    L_clipped = np.clip(Lmap, iv_l_vals[0], LMAX)
    I_map = I_from_L(L_clipped)
    V_map = V_from_I(I_map)
    P_map = I_map * V_map
    P_dyn = float(np.mean(P_map))
    P_frame = P_dyn + float(P_OVERHEAD)
    return dict(P_map=P_map, I_map=I_map, V_map=V_map, P_dyn=P_dyn, P_frame=P_frame)

# ---------- Fast ecc-axis proxy helper ----------
def frame_power_for_coeff_array(coeffs):
    ecc_axis = np.linspace(0.0, max(HFOV, VFOV), 400)
    def frame_power_for_coeff(per_deg):
        edge_offset = np.maximum(0.0, ecc_axis - FOVEA_DEG)
        Lvals = LMAX * np.maximum(0.0, 1.0 - per_deg * edge_offset)
        Lvals = np.clip(Lvals, MIN_L, LMAX)
        Ivals = I_from_L(Lvals)
        Vvals = V_from_I(Ivals)
        Pvals = Ivals * Vvals
        return float(np.mean(Pvals) + P_OVERHEAD)
    return np.array([frame_power_for_coeff(c) for c in coeffs])

# ---------- IVL / Tile power profile (legend lowered to avoid clipping) ----------
def save_ivl_plot(out_path):
    ecc_axis = np.linspace(0.0, max(HFOV, VFOV), 400)
    plt.figure(figsize=(7.5, 4.0))
    ax = plt.gca()
    colors = {1: "black", 2: "tab:green", 3: "tab:gray", 4: "tab:orange"}
    for idx in sorted(PRESETS.keys()):
        per = PRESETS[idx]["PER_DEG_REDUCTION"]
        edge_offset = np.maximum(0.0, ecc_axis - FOVEA_DEG)
        Lvals = np.clip(LMAX * np.maximum(0.0, 1.0 - per * edge_offset), MIN_L, LMAX)
        Ivals = I_from_L(Lvals)
        Vvals = V_from_I(Ivals)
        Pvals = Ivals * Vvals
        ax.plot(ecc_axis, Pvals, color=colors[idx], linewidth=2, label=f"{PRESETS[idx]['name']} ({per:.4f})")
    ax.axvline(FOVEA_DEG, color="cyan", linestyle="--", linewidth=1, label=f"Fovea edge {int(FOVEA_DEG)}")
    ax.set_xlabel("Elliptical eccentricity (deg)")
    ax.set_ylabel("Tile power (W per tile)")
    ax.set_title("Tile Power Profile")
    ax.grid(True, linestyle=":", alpha=0.5)
    # place legend below plot to avoid clipping
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize="small")
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        print("Failed to save IVL plot:", e)
    plt.close()

# ---------- Watt bar (full 2D sim per preset; scaled so Uniform -> 100.00 W; boxed labels, no overlap) ----------
def save_watt_bar(out_path, gaze_x, gaze_y, mode, ppd):
    ordered = sorted(PRESETS.keys())  # ensures Uniform first
    raw_powers = []
    for idx in ordered:
        per = PRESETS[idx]['PER_DEG_REDUCTION']
        Lmap_tmp = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, ppd, per)
        metrics_tmp = power_from_luminance_map(Lmap_tmp)
        raw_powers.append(float(metrics_tmp['P_frame']))
    raw_powers = np.array(raw_powers)
    raw_uniform = raw_powers[0]
    scale = (UNIFORM_REF_W / raw_uniform) if raw_uniform > 0 else 1.0
    scaled_powers = raw_powers * scale

    names = [PRESETS[i]['name'] for i in ordered]
    colors = ["black", "tab:green", "silver", "tab:orange"][:len(names)]

    plt.figure(figsize=(6.0, 3.2))
    bars = plt.bar(names, scaled_powers, color=colors)
    plt.ylabel("Frame power (W)", fontsize=9)
    plt.title("Frame Power , Preset Comparison (Uniform = 100.00 W)", fontsize=10)
    ymax = max(scaled_powers) * 1.25 if max(scaled_powers) > 0 else 1.0
    plt.ylim(0.0, ymax)

    # Build target annotation positions above bars avoiding collisions:
    # compute desired y positions and then resolve collisions by pushing up.
    desired_ys = []
    for v in scaled_powers:
        desired_ys.append(v + ymax * 0.04)  # base offset
    # resolve collisions by sorting by y and ensuring minimum spacing
    min_spacing = ymax * 0.04
    # pair indices with desired ys, sort by desired_y
    items = sorted(list(enumerate(desired_ys)), key=lambda x: x[1])
    placed = [None] * len(items)
    for i, (idx, y) in enumerate(items):
        # ensure new y is at least min_spacing above previous placed that collides
        new_y = y
        for j in range(i):
            prev_idx, prev_y = items[j]
            if placed[prev_idx] is not None and abs(new_y - placed[prev_idx]) < min_spacing:
                new_y = placed[prev_idx] + min_spacing
        placed[idx] = new_y
    # now annotate each bar at placed[idx]
    text_fs = 9
    for b, scaled_v, raw_v, ann_y in zip(bars, scaled_powers, raw_powers, placed):
        cx = b.get_x() + b.get_width() / 2
        label_main = f"{scaled_v:.2f} W"       # two decimal formatted scaled
        label_sub = f"(raw {raw_v:.3f} W)"     # more precise raw
        label = f"{label_main}\n{label_sub}"
        # offset in points: (ann_y - scaled_v) * 72 approx (1 unit -> 72 points)
        offset_points = max(6, int((ann_y - scaled_v) * 72))
        plt.annotate(label,
                     xy=(cx, scaled_v),
                     xytext=(0, offset_points),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=text_fs,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.95),
                     arrowprops=dict(arrowstyle="-", alpha=0.0))

    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        print("Failed to save Watt bar:", e)
    plt.close()

# ---------- Savings scatter (red polyline through points, black preset markers, boxed labels, no overlap) ----------
def save_savings_scatter(out_path):
    coeffs = np.arange(SWEEP_MIN, SWEEP_MAX + 1e-12, SWEEP_STEP)
    sim_powers_raw = frame_power_for_coeff_array(coeffs)
    raw_uniform = float(sim_powers_raw[0])
    scale = (UNIFORM_REF_W / raw_uniform) if raw_uniform > 0 else 1.0
    sim_powers_scaled = sim_powers_raw * scale
    savings_pct = 100.0 * (1.0 - sim_powers_scaled / UNIFORM_REF_W)

    coeffs_percent = coeffs * 100.0
    plt.figure(figsize=(8.0, 4.5))

    # red polyline through all simulated points
    plt.plot(coeffs_percent, savings_pct, color="red", linewidth=1.5, alpha=0.9)
    plt.scatter(coeffs_percent, savings_pct, color="tab:blue", s=14, alpha=0.9, label="Simulated points")

    # key presets (excluding Uniform) — black markers
    key_items = [(idx, PRESETS[idx]["PER_DEG_REDUCTION"]) for idx in sorted(PRESETS.keys()) if PRESETS[idx]["PER_DEG_REDUCTION"] > 0.0]
    annotations = []
    if key_items:
        key_coeffs = np.array([p for (_, p) in key_items])
        key_raw = frame_power_for_coeff_array(key_coeffs)
        key_scaled = key_raw * scale
        key_savings = 100.0 * (1.0 - key_scaled / UNIFORM_REF_W)
        plt.scatter(key_coeffs * 100.0, key_savings, color="black", s=60, zorder=6, label="Presets")
        # prepare annotation candidates
        for (idx, per), ks, kW in zip(key_items, key_savings, key_scaled):
            annotations.append({
                "idx": idx,
                "x": per * 100.0,
                "y": ks,
                "text": f"{ks:.2f}%\n({kW:.2f} W)"
            })

    # place annotations without overlap: greedy vertical placement
    if annotations:
        # target baseline y for each annotation (slightly above point)
        baseline_ys = [a["y"] + 0.35 for a in annotations]
        # create list of (i, baseline) sorted by baseline
        order = sorted(range(len(baseline_ys)), key=lambda i: baseline_ys[i])
        placed_ys = [None] * len(baseline_ys)
        min_v_spacing = (max(savings_pct) - min(savings_pct) + 1e-9) * 0.04  # fraction of range
        if min_v_spacing == 0:
            min_v_spacing = 0.5
        for i in order:
            desired = baseline_ys[i]
            # push up if collides with any previously placed
            while any(abs(desired - py) < min_v_spacing for py in placed_ys if py is not None):
                desired += min_v_spacing
            placed_ys[i] = desired
        # apply annotations
        text_fs = 9
        for a, py in zip(annotations, placed_ys):
            x_pos = a["x"]
            y_pos = a["y"]
            # offset in points (ann_y - base_y) * 72
            offset_points = int(max(8, (py - y_pos) * 72))
            plt.annotate(a["text"],
                         xy=(x_pos, y_pos),
                         xytext=(0, offset_points),
                         textcoords='offset points',
                         ha='center', va='bottom',
                         fontsize=text_fs,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", alpha=0.95),
                         arrowprops=dict(arrowstyle="-", alpha=0.0))

    # perceptibility vertical line with boxed label
    percept_x_percent = 0.8
    plt.axvline(percept_x_percent, color="red", linestyle="--", linewidth=1.2)
    if len(savings_pct) > 0:
        y_label_pos = np.max(savings_pct) * 0.40
    else:
        y_label_pos = 1.0
    plt.text(percept_x_percent + 0.12, y_label_pos, "Perceptibility threshold (0.8%)", color="red", fontsize=9,
             bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1))

    plt.xlabel("PER_DEG_REDUCTION coefficient (%)", fontsize=9)
    plt.ylabel("Power savings vs Uniform (%)", fontsize=9)
    plt.title("Percent Power Saved vs Falloff Coefficient (Uniform = 100.00 W)", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    try:
        plt.savefig(out_path, dpi=150)
    except Exception as e:
        print("Failed to save savings scatter:", e)
    plt.close()

# ---------- Debug print function (D key) ----------
def debug_print_preset_powers(gaze_x, gaze_y, mode, ppd):
    ordered = sorted(PRESETS.keys())
    raw_powers = []
    for idx in ordered:
        per = PRESETS[idx]['PER_DEG_REDUCTION']
        Lmap_tmp = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, ppd, per)
        metrics_tmp = power_from_luminance_map(Lmap_tmp)
        raw_powers.append(float(metrics_tmp['P_frame']))
    raw_powers = np.array(raw_powers)
    raw_uniform = raw_powers[0]
    scale = (UNIFORM_REF_W / raw_uniform) if raw_uniform > 0 else 1.0
    scaled = raw_powers * scale

    print("\nPreset | coeff     | raw P_frame W | scaled P_frame W | saved %")
    print("---------------------------------------------------------------")
    for idx, raw_v, scaled_v in zip(ordered, raw_powers, scaled):
        coeff = PRESETS[idx]['PER_DEG_REDUCTION']
        saved_pct = 100.0 * (1.0 - scaled_v / UNIFORM_REF_W)
        print(f"{PRESETS[idx]['name']:<12s} | {coeff:6.4f} | {raw_v:12.6f} | {scaled_v:14.6f} | {saved_pct:7.3f}%")
    print("Scale factor used:", scale, "\n")

# ---------- Logging ----------
LOG_FILE = "foveation_power_log_final_no_overlap.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","preset","mode","Lavg_nits","P_model_scaled_W","P_uniform_ref_W","power_saved_pct","window_w","window_h","ppd"])

# ---------- State ----------
gaze_x, gaze_y = WINDOW_W / 2, WINDOW_H / 2
show_signal = False
mode = "elliptical"
last_compute = 0.0
compute_interval = 0.02

ivl_surface = None
bar_surface = None
savings_surface = None

overlay_enabled = False
savings_enabled = False

# initial compute
Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD, PRESETS[current_preset]['PER_DEG_REDUCTION'])
RGB_tiles = luminance_to_rgb_tiles(Lmap, show_signal=show_signal, uniform=(PRESETS[current_preset]['PER_DEG_REDUCTION'] == 0.0))

# ---------- Main loop ----------
running = True
while running:
    dt = clock.tick(TARGET_FPS) / 1000.0
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.VIDEORESIZE:
            WINDOW_W, WINDOW_H = ev.w, ev.h
            rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
            px_centers, py_centers = make_centers(rows, cols, rw, rh)
        elif ev.type == pygame.MOUSEMOTION:
            gaze_x, gaze_y = ev.pos
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_s:
                show_signal = not show_signal
            elif ev.key == pygame.K_m:
                mode = "circular" if mode == "elliptical" else "elliptical"
            elif ev.key == pygame.K_1:
                current_preset = 1
            elif ev.key == pygame.K_2:
                current_preset = 2
            elif ev.key == pygame.K_3:
                current_preset = 3
            elif ev.key == pygame.K_4:
                current_preset = 4

            # PPD change: update window size EXACTLY HFOV*PPD and VFOV*PPD and internal grid
            elif ev.key == pygame.K_RIGHTBRACKET or (hasattr(ev, "unicode") and ev.unicode == "]"):
                PPD = min(PPD_MAX, PPD + 1)
                WINDOW_W, WINDOW_H = window_size_exact(PPD)
                screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)

            elif ev.key == pygame.K_LEFTBRACKET or (hasattr(ev, "unicode") and ev.unicode == "["):
                PPD = max(1, PPD - 1)
                WINDOW_W, WINDOW_H = window_size_exact(PPD)
                screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)

            # Pixel controls J/K (single-step)
            elif ev.key == pygame.K_j:
                TILE_SIZE = max(1, TILE_SIZE - 1)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
            elif ev.key == pygame.K_k:
                TILE_SIZE = min(512, TILE_SIZE + 1)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)

            elif ev.key == pygame.K_COMMA:
                render_scale = max(0.05, render_scale * 0.8)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
            elif ev.key == pygame.K_PERIOD:
                render_scale = min(1.0, render_scale * 1.25)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)

            elif ev.key == pygame.K_e:
                rgb_tiles = luminance_to_rgb_tiles(Lmap, show_signal=show_signal,
                                                   uniform=(PRESETS[current_preset]['PER_DEG_REDUCTION'] == 0.0))
                surf = pygame.surfarray.make_surface(RGB_tiles.swapaxes(0, 1))
                fname = f"fovea_frame_{PRESETS[current_preset]['name']}_{int(time.time())}.png"
                try:
                    pygame.image.save(pygame.transform.scale(surf, (WINDOW_W, WINDOW_H)), fname)
                    print("Saved frame:", fname)
                except Exception as e:
                    print("Failed to save frame:", e)

            elif ev.key == pygame.K_l:
                metrics = power_from_luminance_map(Lmap)
                raw_model = metrics["P_frame"]
                raw_uniform = frame_power_for_coeff_array(np.array([0.0]))[0]
                scale = (UNIFORM_REF_W / raw_uniform) if raw_uniform > 0 else 1.0
                P_model_scaled = raw_model * scale
                power_saved_pct = 100.0 * (1.0 - (P_model_scaled / UNIFORM_REF_W)) if UNIFORM_REF_W > 0 else 0.0
                try:
                    with open(LOG_FILE, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([time.time(), PRESETS[current_preset]['name'], mode, float(np.mean(Lmap)),
                                    P_model_scaled, UNIFORM_REF_W, power_saved_pct, WINDOW_W, WINDOW_H, PPD])
                    print("Logged metrics")
                except Exception as e:
                    print("Failed to log metrics:", e)

            elif ev.key == pygame.K_v:
                try:
                    save_ivl_plot(IVL_PNG)
                    save_watt_bar(BAR_PNG, gaze_x, gaze_y, mode, PPD)
                    save_savings_scatter(SAVINGS_PNG)
                    ivl_surface = pygame.image.load(IVL_PNG).convert_alpha() if os.path.exists(IVL_PNG) else None
                    bar_surface = pygame.image.load(BAR_PNG).convert_alpha() if os.path.exists(BAR_PNG) else None
                    savings_surface = pygame.image.load(SAVINGS_PNG).convert_alpha() if os.path.exists(SAVINGS_PNG) else None
                except Exception as e:
                    print("Failed to regenerate/load overlays:", e)
                    ivl_surface = bar_surface = savings_surface = None
                overlay_enabled = not overlay_enabled
                print("Overlay toggled:", overlay_enabled)

            elif ev.key == pygame.K_g:
                try:
                    save_savings_scatter(SAVINGS_PNG)
                    savings_surface = pygame.image.load(SAVINGS_PNG).convert_alpha()
                except Exception as e:
                    print("Failed to generate/load savings scatter:", e)
                    savings_surface = None
                savings_enabled = not savings_enabled
                print("Savings scatter toggled:", savings_enabled)

            elif ev.key == pygame.K_p:
                try:
                    save_ivl_plot(IVL_PNG)
                    save_watt_bar(BAR_PNG, gaze_x, gaze_y, mode, PPD)
                    save_savings_scatter(SAVINGS_PNG)
                    print("Saved overlays")
                except Exception as e:
                    print("Failed to save overlays:", e)

            elif ev.key == pygame.K_d:
                debug_print_preset_powers(gaze_x, gaze_y, mode, PPD)

    # periodic recompute
    now = time.time()
    if (now - last_compute) > compute_interval:
        Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD, PRESETS[current_preset]['PER_DEG_REDUCTION'])
        RGB_tiles = luminance_to_rgb_tiles(Lmap, show_signal=show_signal, uniform=(PRESETS[current_preset]['PER_DEG_REDUCTION'] == 0.0))
        last_compute = now

    # render tiles
    surf = pygame.surfarray.make_surface(RGB_tiles.swapaxes(0, 1))
    upscaled = pygame.transform.scale(surf, (WINDOW_W, WINDOW_H))
    screen.blit(upscaled, (0, 0))

    # crosshair and elliptical fovea outline
    cx, cy = int(WINDOW_W / 2), int(WINDOW_H / 2)
    pygame.draw.rect(screen, (255, 255, 255), (cx - 1, cy - 1, 2, 2))
    pygame.draw.line(screen, (255, 255, 255), (gaze_x - 12, gaze_y), (gaze_x + 12, gaze_y), 1)
    pygame.draw.line(screen, (255, 255, 255), (gaze_x, gaze_y - 12), (gaze_x, gaze_y + 12), 1)
    scale_x = HFOV / max(HFOV, VFOV)
    scale_y = VFOV / max(HFOV, VFOV)
    fovea_px_rx = int(FOVEA_DEG * scale_x * PPD)
    fovea_px_ry = int(FOVEA_DEG * scale_y * PPD)
    ellipse_rect = pygame.Rect(int(gaze_x - fovea_px_rx), int(gaze_y - fovea_px_ry), int(2 * fovea_px_rx), int(2 * fovea_px_ry))
    pygame.draw.ellipse(screen, (255, 255, 255), ellipse_rect, 1)

    # metrics and HUD: show scaled P_frame
    metrics = power_from_luminance_map(Lmap)
    P_model_raw = metrics["P_frame"]
    raw_uniform = frame_power_for_coeff_array(np.array([0.0]))[0]
    scale = (UNIFORM_REF_W / raw_uniform) if raw_uniform > 0 else 1.0
    P_model_scaled = P_model_raw * scale
    power_saved_pct = 100.0 * (1.0 - (P_model_scaled / UNIFORM_REF_W)) if UNIFORM_REF_W > 0 else 0.0
    Lavg = float(np.mean(Lmap))

    hud_lines = [
        f"Preset: {PRESETS[current_preset]['name']}  per-deg: {PRESETS[current_preset]['PER_DEG_REDUCTION']:.4f}  Min {MIN_L} nits  Overhead {P_OVERHEAD:.2f} W",
        f"Falloff mapping: {mode}  FOV: {int(HFOV)} x {int(VFOV)} deg  Window: {WINDOW_W} x {WINDOW_H} px",
        f"Mean L: {Lavg:.1f} nits  Scaled P_frame: {P_model_scaled:.3f} W  Uniform ref: {UNIFORM_REF_W:.2f} W  Saved: {power_saved_pct:.2f}%",
        f"PPD: {PPD} (max {PPD_MAX})  Tiles: {cols}x{rows}"
    ]
    hud_lines.append("Keys: 1-4 presets  ]/[ change PPD  J/K pixel +/-  D debug  V overlays  G savings  P/E/L/ESC")

    hud = pygame.Surface((1000, 140), pygame.SRCALPHA)
    hud.fill((0, 0, 0, 180))
    for i, ln in enumerate(hud_lines):
        hud.blit(font.render(ln, True, (255, 255, 255)), (8, 6 + i * 20))
    screen.blit(hud, (8, 8))

    # overlay stack (top-right)
    y_off = 12
    if overlay_enabled:
        if ivl_surface is None:
            try:
                save_ivl_plot(IVL_PNG)
                ivl_surface = pygame.image.load(IVL_PNG).convert_alpha()
            except Exception:
                ivl_surface = None
        if bar_surface is None:
            try:
                bar_surface = pygame.image.load(BAR_PNG).convert_alpha()
            except Exception:
                bar_surface = None
        if savings_surface is None:
            try:
                savings_surface = pygame.image.load(SAVINGS_PNG).convert_alpha()
            except Exception:
                savings_surface = None

        if ivl_surface is not None:
            ow, oh = ivl_surface.get_size()
            target_w = min(560, ow)
            scale_img = target_w / ow
            target_h = int(oh * scale_img)
            screen.blit(pygame.transform.smoothscale(ivl_surface, (target_w, target_h)), (WINDOW_W - target_w - 12, y_off))
            y_off += target_h + 8
        if bar_surface is not None:
            bw, bh = bar_surface.get_size()
            target_bw = min(360, bw)
            bscale = target_bw / bw
            target_bh = int(bh * bscale)
            screen.blit(pygame.transform.smoothscale(bar_surface, (target_bw, target_bh)), (WINDOW_W - target_bw - 12, y_off))
            y_off += target_bh + 8
        if savings_surface is not None:
            sw, sh = savings_surface.get_size()
            target_sw = min(700, sw)
            sscale = target_sw / sw
            target_sh = int(sh * sscale)
            screen.blit(pygame.transform.smoothscale(savings_surface, (target_sw, target_sh)), (WINDOW_W - target_sw - 12, y_off))
            y_off += target_sh + 8

    # savings-only display when G toggled
    if savings_enabled and not overlay_enabled:
        if savings_surface is None:
            try:
                savings_surface = pygame.image.load(SAVINGS_PNG).convert_alpha()
            except Exception:
                savings_surface = None
        if savings_surface is not None:
            sw, sh = savings_surface.get_size()
            target_sw = min(900, sw)
            sscale = target_sw / sw
            target_sh = int(sh * sscale)
            screen.blit(pygame.transform.smoothscale(savings_surface, (target_sw, target_sh)), (WINDOW_W - target_sw - 12, 12))

    pygame.display.flip()

pygame.quit()
sys.exit()
