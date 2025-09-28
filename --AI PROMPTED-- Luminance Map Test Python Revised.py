# fovea_fov_elliptical_power_realistic.py
# Human-FOV sized foveated brightness demo with circular vs elliptical sensitivity modes
# More realistic power model using IVL lookup/interpolation (literature-like proxy)
# Requirements: numpy, pygame, matplotlib (matplotlib optional), scipy
# Install: pip install numpy pygame matplotlib scipy

"""
Notes on realism:
- This script uses a literature-style IVL proxy (lookup tables) for an InP-based QD LED.
  Replace the arrays iv_l_vals / iv_i_vals / iv_v_vals with measured IVL data when available.
- The power model computes per-tile current by interpolating I(L), then voltage by interpolating V(I),
  finally P = V * I. A fixed P_OVERHEAD models driver/controller losses.
- Default parameters and IVL proxy chosen to produce realistic mid-range device behavior.
- Toggle presets with keys 1 (conservative), 2 (realistic), 3 (optimistic).
- Press 'l' to append a CSV row with current numeric metrics.
"""

import sys, time, math, csv, os
import numpy as np
import pygame
from scipy.interpolate import interp1d

# Optional colormap
USE_MATPLOTLIB = True
try:
    import matplotlib.cm as cm
except Exception:
    USE_MATPLOTLIB = False

# ---------- Human FOV and mapping parameters ----------
HFOV_DEG = 180.0       # horizontal field of view in degrees
VFOV_DEG = 130.0       # vertical field of view in degrees
PPD = 8                # pixels per degree; window size = HFOV*PPD x VFOV*PPD
LMAX = 1000.0          # peak luminance (nits) reference for uniform frame
MIN_L = 10.0           # allow deep peripheral dim (nits)
PER_DEG_REDUCTION = 0.0063  # 0.63% per degree (base falloff)
GAMMA = 2.2

# ---------- Performance and visual parameters ----------
render_scale = 0.35    # compute at fraction of window resolution then upscale
TILE_SIZE = 2          # visible block size after upscale
TARGET_FPS = 60

# ---------- IVL proxy (replace with measured IVL when available) ----------
# These arrays represent a plausible InP QD LED behavior: rows map luminance (cd/m2 ~ nits)
# to drive current density or current (A) and voltage (V). Values are illustrative and should
# be swapped for your measured IVL as soon as possible.
#
# iv_l_vals: luminance points (nits)
# iv_i_vals: measured current per tile (mA) required to produce that luminance (per unit tile area)
# iv_v_vals: measured voltage across device at that drive current (V)
#
# The numbers below are synthetic but shaped like real IVL curves:
iv_l_vals = np.array([1.0, 10.0, 50.0, 100.0, 200.0, 400.0, 600.0, 800.0, 1000.0])  # nits
# Current per tile (mA) at each luminance point for a representative tile area.
# Values chosen to give realistic nonlinearity: low L requires tiny current; high L requires much higher current.
iv_i_vals_mA = np.array([0.01, 0.05, 0.3, 0.9, 2.5, 6.5, 13.0, 22.0, 35.0])  # mA per tile
# Voltage at those current points (V). Slightly rising with current due to diode characteristic + series loss.
iv_v_vals = np.array([2.4, 2.45, 2.5, 2.55, 2.65, 2.8, 3.0, 3.15, 3.3])  # V

# Build interpolators (L -> I (A) and I -> V (V))
I_from_L_interp = interp1d(iv_l_vals, iv_i_vals_mA / 1000.0, kind='linear', bounds_error=False,
                          fill_value=(iv_i_vals_mA[0] / 1000.0, iv_i_vals_mA[-1] / 1000.0))
V_from_I_interp = interp1d(iv_i_vals_mA / 1000.0, iv_v_vals, kind='linear', bounds_error=False,
                           fill_value=(iv_v_vals[0], iv_v_vals[-1]))

# ---------- System overhead and presets ----------
P_OVERHEAD_DEFAULT = 0.20  # W, controller + converters baseline (realistic mid)
P_OVERHEAD = P_OVERHEAD_DEFAULT

# Preset parameter sets (conservative / realistic / optimistic) - adjust IVL or overhead subtly
PRESETS = {
    1: dict(name="Conservative", P_OVERHEAD=0.5, MIN_L=200.0),   # favors low reported savings
    2: dict(name="Realistic", P_OVERHEAD=0.20, MIN_L=10.0),      # balanced
    3: dict(name="Optimistic", P_OVERHEAD=0.05, MIN_L=1.0)       # shows best-case potential
}
current_preset = 2
MIN_L = PRESETS[current_preset]['MIN_L']
P_OVERHEAD = PRESETS[current_preset]['P_OVERHEAD']

# ---------- Derived and init ----------
def window_size(ppd):
    return max(200, int(HFOV_DEG * ppd)), max(200, int(VFOV_DEG * ppd))

WINDOW_W, WINDOW_H = window_size(PPD)
pygame.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
pygame.display.set_caption("Foveated Brightness — Elliptical (Realistic Power)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 16)

def compute_internal_sizes(scale, tile_size, w, h):
    rw = max(4, int(w * scale))
    rh = max(4, int(h * scale))
    cols = max(2, rw // tile_size)
    rows = max(2, rh // tile_size)
    return rows, cols, rw, rh

def make_centers(rows, cols, rw, rh):
    px = (np.arange(cols) + 0.5) * (rw / cols)
    py = (np.arange(rows) + 0.5) * (rh / rows)
    px_centers, py_centers = np.meshgrid(px, py)
    return px_centers, py_centers

rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
px_centers, py_centers = make_centers(rows, cols, rw, rh)
deg_per_pixel = 1.0 / PPD  # degrees per window pixel

# ---------- Luminance math ----------
def circular_ecc_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd):
    x_win = px_centers * (win_w / px_centers.max())
    y_win = py_centers * (win_h / py_centers.max())
    x_deg = (x_win - gaze_x) * (1.0 / ppd)
    y_deg = (y_win - gaze_y) * (1.0 / ppd)
    ecc = np.hypot(x_deg, y_deg)
    return ecc

def elliptical_effective_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd):
    x_win = px_centers * (win_w / px_centers.max())
    y_win = py_centers * (win_h / py_centers.max())
    x_deg = (x_win - gaze_x) * (1.0 / ppd)
    y_deg = (y_win - gaze_y) * (1.0 / ppd)
    a = HFOV_DEG / 2.0
    b = VFOV_DEG / 2.0
    d_norm = np.sqrt((x_deg / a) ** 2 + (y_deg / b) ** 2)
    eff_deg = d_norm * max(a, b)
    return eff_deg

def luminance_from_effective_deg(eff_deg):
    L = LMAX * np.maximum(0.0, 1.0 - PER_DEG_REDUCTION * eff_deg)
    L = np.clip(L, MIN_L, LMAX)
    return L

def luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, win_w, win_h, ppd):
    if mode == "circular":
        ecc = circular_ecc_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd)
        L = luminance_from_effective_deg(ecc)
    else:
        eff = elliptical_effective_deg(px_centers, py_centers, gaze_x, gaze_y, win_w, win_h, ppd)
        L = luminance_from_effective_deg(eff)
    return L

def luminance_to_rgb(L, show_signal=False):
    if show_signal:
        V = np.power(L / LMAX, 1.0 / GAMMA)
        t = np.clip(V, 0.0, 1.0)
    else:
        t = (L - MIN_L) / (LMAX - MIN_L)
        t = np.clip(t, 0.0, 1.0)
    if USE_MATPLOTLIB:
        cmap = cm.get_cmap('inferno')
        rgba = cmap(t)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    else:
        gray = (t * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=2)
    return rgb

# ---------- State ----------
gaze_x, gaze_y = WINDOW_W / 2, WINDOW_H / 2
show_signal = False
mode = "elliptical"    # default 'elliptical'
last_compute = 0.0
compute_interval = 0.02

# initial compute
Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
RGB = luminance_to_rgb(Lmap, show_signal=show_signal)

# CSV logging setup
LOG_FILE = "foveation_power_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "preset", "mode", "Lavg_nits", "P_foveated_W", "P_uniform_W", "power_saved_pct"])

# ---------- Controls ----------
# s = toggle signal view
# m = toggle mode circular<->elliptical
# ] / [ = increase / decrease PPD (resizes window)
# + / - = change TILE_SIZE
# , / . = change render_scale down / up
# e = export PNG
# l = log current numeric metrics to CSV
# 1/2/3 = preset parameter sets
# ESC = quit

# ---------- Main loop ----------
running = True
while running:
    dt = clock.tick(TARGET_FPS) / 1000.0
    fps = clock.get_fps()
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.VIDEORESIZE:
            WINDOW_W, WINDOW_H = ev.w, ev.h
            PPD = WINDOW_W / HFOV_DEG
            rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
            px_centers, py_centers = make_centers(rows, cols, rw, rh)
            Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
            RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
        elif ev.type == pygame.MOUSEMOTION:
            gaze_x, gaze_y = ev.pos
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_s:
                show_signal = not show_signal
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_m:
                mode = "circular" if mode == "elliptical" else "elliptical"
                Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_RIGHTBRACKET:
                PPD = min(48, PPD + 1)
                WINDOW_W, WINDOW_H = window_size(PPD)
                pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
                Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_LEFTBRACKET:
                PPD = max(1, PPD - 1)
                WINDOW_W, WINDOW_H = window_size(PPD)
                pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.RESIZABLE)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
                Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_PLUS or ev.key == pygame.K_EQUALS:
                TILE_SIZE = max(1, TILE_SIZE - 1)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
                Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_MINUS or ev.key == pygame.K_UNDERSCORE:
                TILE_SIZE = min(128, TILE_SIZE + 1)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
                Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_COMMA:
                render_scale = max(0.05, render_scale * 0.8)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
                Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_PERIOD:
                render_scale = min(1.0, render_scale * 1.25)
                rows, cols, rw, rh = compute_internal_sizes(render_scale, TILE_SIZE, WINDOW_W, WINDOW_H)
                px_centers, py_centers = make_centers(rows, cols, rw, rh)
                Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
                RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
            elif ev.key == pygame.K_e:
                rgb_up = pygame.surfarray.make_surface(RGB.swapaxes(0,1))
                fname = f"fovea_fov_{mode}_{int(time.time())}.png"
                pygame.image.save(pygame.transform.scale(rgb_up, (WINDOW_W, WINDOW_H)), fname)
                print("Saved", fname)
            elif ev.key == pygame.K_l:
                # log current metrics to CSV
                with open(LOG_FILE, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([time.time(), PRESETS[current_preset]['name'], mode, float(np.mean(Lmap)),
                                P_foveated, P_uniform, power_saved_pct])
                print("Logged metrics to", LOG_FILE)
            elif ev.key in (pygame.K_1, pygame.K_KP1):
                current_preset = 1
                MIN_L = PRESETS[current_preset]['MIN_L']
                P_OVERHEAD = PRESETS[current_preset]['P_OVERHEAD']
                print("Preset ->", PRESETS[current_preset]['name'])
            elif ev.key in (pygame.K_2, pygame.K_KP2):
                current_preset = 2
                MIN_L = PRESETS[current_preset]['MIN_L']
                P_OVERHEAD = PRESETS[current_preset]['P_OVERHEAD']
                print("Preset ->", PRESETS[current_preset]['name'])
            elif ev.key in (pygame.K_3, pygame.K_KP3):
                current_preset = 3
                MIN_L = PRESETS[current_preset]['MIN_L']
                P_OVERHEAD = PRESETS[current_preset]['P_OVERHEAD']
                print("Preset ->", PRESETS[current_preset]['name'])

    now = time.time()
    if (now - last_compute) > compute_interval:
        Lmap = luminance_map(px_centers, py_centers, gaze_x, gaze_y, mode, WINDOW_W, WINDOW_H, PPD)
        RGB = luminance_to_rgb(Lmap, show_signal=show_signal)
        last_compute = now

    surf = pygame.surfarray.make_surface(RGB.swapaxes(0,1))
    upscaled = pygame.transform.scale(surf, (WINDOW_W, WINDOW_H))
    screen.blit(upscaled, (0,0))

    # --- realistic power estimate using IVL interpolation ---
    # Lmap in nits (rows x cols). Clip again to current MIN_L enforced by preset.
    Lmap = np.clip(Lmap, MIN_L, LMAX)
    Lavg = float(np.mean(Lmap))       # mean luminance in nits (keep this displayed)

    # Interpolate to per-tile current (A) from L
    I_map = I_from_L_interp(Lmap)     # A per tile
    # Interpolate voltage from per-tile current
    V_map = V_from_I_interp(I_map)    # V per tile
    # Per-tile power (W)
    P_map = I_map * V_map
    # Mean electrical power for the current (foveated) frame plus overhead
    P_foveated = float(np.mean(P_map)) + float(P_OVERHEAD)

    # Reference uniform 1000-nit frame
    L_uniform = np.full_like(Lmap, LMAX)
    I_uniform_map = I_from_L_interp(L_uniform)
    V_uniform_map = V_from_I_interp(I_uniform_map)
    P_uniform_map = I_uniform_map * V_uniform_map
    P_uniform = float(np.mean(P_uniform_map)) + float(P_OVERHEAD)

    # Percent saved
    power_saved_pct = 100.0 * (1.0 - (P_foveated / P_uniform))
    power_saved_pct = max(0.0, min(100.0, power_saved_pct))

    # center-of-screen luminance (nits) display (compute used L at center)
    center_px = WINDOW_W / 2
    center_py = WINDOW_H / 2
    if mode == "circular":
        center_ecc = math.hypot((center_px - gaze_x) / PPD, (center_py - gaze_y) / PPD)
        center_L = LMAX * max(0.0, 1.0 - PER_DEG_REDUCTION * center_ecc)
    else:
        a = HFOV_DEG / 2.0
        b = VFOV_DEG / 2.0
        x_deg = (center_px - gaze_x) / PPD
        y_deg = (center_py - gaze_y) / PPD
        d_norm = math.sqrt((x_deg / a) ** 2 + (y_deg / b) ** 2)
        eff_deg = d_norm * max(a, b)
        center_L = LMAX * max(0.0, 1.0 - PER_DEG_REDUCTION * eff_deg)
    if center_L < MIN_L:
        center_L = MIN_L

    # HUD
    hud_lines = [
        f"Preset: {PRESETS[current_preset]['name']}   Mode: {mode}   PPD {PPD}   Window {WINDOW_W}x{WINDOW_H}   Tiles {cols}x{rows}",
        f"Center luminance: {center_L:.1f} nits   Lmax {LMAX}   Min {MIN_L}",
        f"Mean frame luminance: {Lavg:.1f} nits   Power saved (est): {power_saved_pct:.1f}%",
        f"Estimated P_frame: {P_foveated:.3f} W   Uniform P_ref: {P_uniform:.3f} W",
        f"IVL proxy: L_range {iv_l_vals[0]}..{iv_l_vals[-1]} nits  |  Overhead: {P_OVERHEAD:.2f} W",
        "Keys: 1/2/3 presets  m toggle mode  s toggle signal  [ ] change PPD  + - tile  , . scale  e export  l log  ESC quit"
    ]
    hud_w = 980
    hud = pygame.Surface((hud_w, 140), pygame.SRCALPHA)
    hud.fill((0,0,0,180))
    for i, ln in enumerate(hud_lines):
        hud.blit(font.render(ln, True, (255,255,255)), (8, 6 + i*20))
    screen.blit(hud, (8, 8))

    # crosshair
    pygame.draw.line(screen, (255,255,255), (gaze_x-12, gaze_y), (gaze_x+12, gaze_y), 1)
    pygame.draw.line(screen, (255,255,255), (gaze_x, gaze_y-12), (gaze_x, gaze_y+12), 1)

    pygame.display.flip()

pygame.quit()
sys.exit()
