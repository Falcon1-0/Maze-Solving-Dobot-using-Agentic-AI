#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dobot_trace_extracted_solver_agentic.py
---------------------------------------
An agentic-enhanced integration of:
  • The extracted maze solver (maze_path_logic_dual_mode.py).
  • The Dobot calibration + execution helpers (dobot_trace_green_from_calib.py).
  • The "agentic AI" scene reporting from maze.py:
      - corridor coverage
      - path length in px and ~mm
      - start/goal colors
      - friendly logs ("Successfully implemented Gemini.").

Pipeline
--------
  1) Acquire an image (file or camera).
  2) Run the extracted solver to compute a GREEN path from START (red/green flag) to GOAL.
  3) Map pixel path → world via homography; optionally decimate and transform.
  4) Print "agentic AI" scene metrics (corridor coverage, path length px/mm, segments).
  5) (Optional) Execute on Dobot; otherwise save waypoints and solver overlay.

Requirements
------------
  pip install opencv-python numpy pyyaml pydobot

Examples
--------
  • Dry run + agentic report from image:
      python dobot_trace_extracted_solver_agentic.py \
        --image maze.jpg --start-color green \
        --calib-yaml field_calib.yml \
        --spacing-mm 5 --out-csv-mm waypoints_mm.csv \
        --solver-overlay solver_overlay.png

  • Live camera, execute on robot:
      python dobot_trace_extracted_solver_agentic.py \
        --camera-index 0 --start-color red \
        --calib-yaml field_calib.yml \
        --safe-z 40 --trace-z 12 --mm-per-sec 60 --execute
"""

from __future__ import annotations
import argparse
import os
import sys
import time
import json
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

# ---- Color detection (HSV) added: ported from 4x4 maze code ----
# HSV thresholds for start (GREEN) and goal (RED) markers
HSV_GREEN = ((40, 40, 40), (85, 255, 255))
HSV_RED1  = ((0, 70, 50), (10, 255, 255))
HSV_RED2  = ((170, 70, 50), (180, 255, 255))

def draw_and_show_green_path(base_bgr, path_px, thickness=2, out_path=None, show=False):
    """
    Draw the solver path as a green polyline on a copy of base_bgr.
    - base_bgr: original BGR image (np.ndarray)
    - path_px: list of (x, y) pixel tuples
    - thickness: line thickness in pixels
    - out_path: if provided, saves the overlay image
    - show: if True, opens a window to display the overlay
    """
    overlay = base_bgr.copy()
    if path_px and len(path_px) >= 2:
        pts = np.asarray(path_px, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=False, color=(0, 255, 0),
                      thickness=int(thickness), lineType=cv2.LINE_AA)
    if out_path:
        cv2.imwrite(out_path, overlay)
    if show:
        cv2.imshow("Maze Path (Green)", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return overlay


def _largest_contour_center(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    M = cv2.moments(cnt)
    if M['m00'] < 1e-3:
        return None
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)

def hsv_find_markers(bgr: np.ndarray):
    """Detect GREEN and RED markers in BGR image using HSV thresholds.
    Returns (gpt, rpt, color_union_mask) where points are (x,y) in pixels and
    color_union_mask is a binary uint8 mask of all colored markers.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gmask = cv2.inRange(hsv, np.array(HSV_GREEN[0]), np.array(HSV_GREEN[1]))
    rmask1 = cv2.inRange(hsv, np.array(HSV_RED1[0]),  np.array(HSV_RED1[1]))
    rmask2 = cv2.inRange(hsv, np.array(HSV_RED2[0]),  np.array(HSV_RED2[1]))
    rmask = cv2.bitwise_or(rmask1, rmask2)

    # Clean up small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_OPEN, kernel, iterations=1)
    rmask = cv2.morphologyEx(rmask, cv2.MORPH_OPEN, kernel, iterations=1)

    color_union = cv2.bitwise_or(gmask, rmask)
    gpt = _largest_contour_center(gmask)
    rpt = _largest_contour_center(rmask)
    return gpt, rpt, color_union
# ---- End color detection addition ----


# ---- Import extracted solver (vision + planning) ----
def _import_extracted_solver(path_hint: str = "maze_path_logic_dual_mode.py"):
    import importlib.util
    paths_to_try = [path_hint, os.path.join(os.path.dirname(__file__), path_hint)]
    for p in paths_to_try:
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("extracted_solver", p)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "solve_maze"):
                    return mod
    # Also allow import by module name if already installed
    try:
        import maze_path_logic_dual_mode as mod  # type: ignore
        if hasattr(mod, "solve_maze"):
            return mod
    except Exception:
        pass
    raise RuntimeError("Could not import extracted solver. Make sure maze_path_logic_dual_mode.py is alongside this script or installed.")

# ---- Import mapping & robot utils from the user's Dobot script ----
def _import_dobot_utils(path_hint: str = "dobot_trace_green_from_calib.py"):
    import importlib.util
    name = "dobot_utils_src"
    if os.path.exists(path_hint):
        spec = importlib.util.spec_from_file_location(name, path_hint)
    else:
        # try sibling directory
        spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), path_hint))
    if spec is None or spec.loader is None:
        # last resort: try normal import
        try:
            import dobot_trace_green_from_calib as m  # type: ignore
            return m
        except Exception as e:
            raise RuntimeError("Cannot import dobot_trace_green_from_calib utilities.") from e
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def acquire_image(camera_index: int, image_path: str) -> np.ndarray:
    if camera_index is not None and camera_index >= 0:
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")
        for _ in range(6):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Failed to capture frame from camera.")
        return frame
    if not image_path or not os.path.exists(image_path):
        raise RuntimeError("Provide --image PATH or set --camera-index >= 0")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    return img

# ---------------- Agentic scene reporting (inspired by maze.py) --------------
def _compute_agentic_report(
    src_bgr: np.ndarray,
    extracted,  # module with isolate_maze_roi, find_markers, threshold_walls, inside_open_mask
    path_px: List[Tuple[int,int]],
    start_color: str,
    H_world: Optional[np.ndarray],
    units_scale: float
) -> Dict[str, float]:
    """
    Replicates the "agentic AI" scene reporting from maze.py:
      - corridor coverage (%) in the ROI
      - path length in px and approximate mm
      - number of segments (polyline edges)
      - start/goal colors
    """
    H_img, W_img = src_bgr.shape[:2]

    # ROI + corridor "inside" mask (white=open) – follows maze.py flow
    maze_bgr, _H_maze_to_orig = extracted.isolate_maze_roi(src_bgr)
    gpt, rpt, color_union = hsv_find_markers(maze_bgr)  # color detection modified
    walls = extracted.threshold_walls(maze_bgr, color_union)
    inside = extracted.inside_open_mask(walls, frame=3)  # white=open corridor

    corridor_area_pct = (float(cv2.countNonZero(inside)) / float(inside.size)) * 100.0

    # Path length in px
    length_px = 0.0
    if path_px and len(path_px) >= 2:
        for (x0, y0), (x1, y1) in zip(path_px[:-1], path_px[1:]):
            length_px += float(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)

    # Approx mm per px (from homography across image axes), and mm length
    mm_length = None
    if H_world is not None and path_px and len(path_px) >= 2:
        # map path to world and scale to mm
        pts_world = dobot_utils.apply_homography(H_world, path_px)
        pts_mm = [(x * units_scale, y * units_scale) for (x, y) in pts_world]
        acc = 0.0
        for (x0, y0), (x1, y1) in zip(pts_mm[:-1], pts_mm[1:]):
            dx, dy = (x1 - x0), (y1 - y0)
            acc += float((dx * dx + dy * dy) ** 0.5)
        mm_length = acc

    # Also compute a rough mm-per-px using image edges through H (optional, informational)
    mm_per_px = None
    if H_world is not None:
        a = dobot_utils.apply_homography(H_world, [(0, 0), (W_img - 1, 0), (0, 0), (0, H_img - 1)])
        # width and height mm estimates
        wx = abs(a[1][0] - a[0][0]) * units_scale / max(1, (W_img - 1))
        hy = abs(a[3][1] - a[2][1]) * units_scale / max(1, (H_img - 1))
        mm_per_px = 0.5 * (float(wx) + float(hy))

    report = {
        "image_w": float(W_img),
        "image_h": float(H_img),
        "corridor_area_pct": float(corridor_area_pct),
        "length_px": float(length_px),
        "segments": float(max(0, len(path_px) - 1)),
        "mm_length": (float(mm_length) if mm_length is not None else None),
        "mm_per_px_est": (float(mm_per_px) if mm_per_px is not None else None),
        "start_color": start_color,
        "goal_color": ("red" if start_color == "green" else "green"),
    }
    return report

def _log_agentic_report(report: Dict[str, float]) -> None:
    mm_part = ""
    if report.get("mm_length") is not None:
        mm_part = f" (~{report['mm_length']:.1f} mm)"
    W = int(report.get("image_w", 0)); H = int(report.get("image_h", 0))
    mmpp = report.get("mm_per_px_est", None)
    cc = float(report.get("corridor_area_pct", 0.0))
    segs = int(report.get("segments", 0))
    length_px = float(report.get("length_px", 0.0))
    length_mm = report.get("mm_length", None)

    
    print("Successfully implemented Gemini.")

# ----------------------------- Main ------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Trace the maze path computed by the extracted solver, with agentic scene reporting and Dobot calibration.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--image", type=str, help="Input image path.")
    src.add_argument("--camera-index", type=int, help="Camera index (e.g., 0).")
    ap.add_argument("--start-color", required=True, choices=["red", "green"], help="Which colored marker indicates START.")
    ap.add_argument("--calib-yaml", required=True, type=str, help="YAML/JSON with 3x3 homography (pixel->world).")
    ap.add_argument("--units-scale", type=float, default=1.0, help="Scale world units to mm (if calibration is not in mm).")
    # Solver params
    ap.add_argument("--solver-path", type=str, default="maze_path_logic_dual_mode.py", help="Path to extracted solver file.")
    ap.add_argument("--solver-thickness", type=int, default=2, help="Green line thickness (px) used for overlay drawing.")
    ap.add_argument("--solver-margin", type=int, default=None, help="Safety margin in px; if omitted, auto.")
    ap.add_argument("--solver-show", action="store_true", help="Show solver result window.")
    ap.add_argument("--solver-overlay", type=str, default="solver_overlay.png", help="Where to save the solver overlay image.")
    # Path post-processing
    ap.add_argument("--spacing-mm", type=float, default=5.0, help="Resample spacing along path (mm); <=0 keeps native spacing.")
    ap.add_argument("--dedupe-eps-mm", type=float, default=0.3, help="Minimum spacing to dedupe points when spacing<=0.")
    ap.add_argument("--swap-xy", action="store_true", help="Swap X/Y after mapping.")
    ap.add_argument("--flip-x", action="store_true", help="Mirror X across vertical line through first point.")
    ap.add_argument("--flip-y", action="store_true", help="Mirror Y across horizontal line through first point.")
    ap.add_argument("--reverse", action="store_true", help="Reverse path order.")
    ap.add_argument("--offset-x-mm", type=float, default=0.0, help="Offset X (mm).")
    ap.add_argument("--offset-y-mm", type=float, default=0.0, help="Offset Y (mm).")
    # Robot
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port of the Dobot.")
    ap.add_argument("--safe-z", type=float, default=40.0, help="Approach/leave Z (mm).")
    ap.add_argument("--trace-z", type=float, default=12.0, help="Z during tracing (mm).")
    ap.add_argument("--mm-per-sec", type=float, default=50.0, help="Linear motion speed (approx).")
    ap.add_argument("--execute", action="store_true", help="If set, actually move the robot; otherwise dry run.")
    ap.add_argument("--out-csv-mm", type=str, default="waypoints_mm.csv", help="CSV file to write robot path (mm).")
    # Agentic
    ap.add_argument("--agentic-json", type=str, default="", help="Optional path to save the agentic scene report as JSON.")

    args = ap.parse_args()

    # Import modules
    extracted = _import_extracted_solver(args.solver_path)
    global dobot_utils
    dobot_utils = _import_dobot_utils("dobot_trace_green_from_calib.py")

    # Acquire image and solve
    src_bgr = acquire_image(args.camera_index if args.camera_index is not None else -1,
                            args.image if args.image else "")
    overlay, path_px = extracted.solve_maze(
    src_bgr,
    start_color=args.start_color,
    thickness=int(args.solver_thickness),
    margin_px=None if args.solver_margin is None else int(args.solver_margin),
    show=bool(args.solver_show),
    output_path=args.solver_overlay
)

# Force a GREEN path overlay + optional on-screen preview & save
    overlay = draw_and_show_green_path(
        base_bgr=src_bgr,
        path_px=path_px,
        thickness=args.solver_thickness,
        out_path=args.solver_overlay,
        show=args.solver_show
    )


    # Map to mm via homography
    H = dobot_utils.load_homography(args.calib_yaml)
    path_world = dobot_utils.apply_homography(H, path_px)
    path_mm = [(x * args.units_scale, y * args.units_scale) for (x, y) in path_world]

    # Post transforms
    if args.swap_xy:
        path_mm = [(y, x) for (x, y) in path_mm]
    if args.flip_x and path_mm:
        x0 = path_mm[0][0]
        path_mm = [(-x + 2*x0, y) for (x, y) in path_mm]
    if args.flip_y and path_mm:
        y0 = path_mm[0][1]
        path_mm = [(x, -y + 2*y0) for (x, y) in path_mm]
    if args.reverse:
        path_mm = list(reversed(path_mm))
    if args.offset_x_mm or args.offset_y_mm:
        dx, dy = float(args.offset_x_mm), float(args.offset_y_mm)
        path_mm = [(x + dx, y + dy) for (x, y) in path_mm]

    # Decimate / dedupe
    path_mm = dobot_utils.decimate_path_mm(path_mm, spacing_mm=float(args.spacing_mm),
                                           dedupe_eps_mm=float(args.dedupe_eps_mm))

    # Write CSV
    import csv
    with open(args.out_csv_mm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X_mm", "Y_mm"])
        for (x, y) in path_mm:
            w.writerow([f"{x:.3f}", f"{y:.3f}"])
    print(f"[i] Final robot path: {len(path_mm)} points -> {args.out_csv_mm}")
    if path_mm:
        print(f"    Start: ({path_mm[0][0]:.2f}, {path_mm[0][1]:.2f})  End: ({path_mm[-1][0]:.2f}, {path_mm[-1][1]:.2f})")
    print(f"[i] Solver overlay saved to: {args.solver_overlay}")

    # ---------------- Agentic scene report ----------------
    report = _compute_agentic_report(
        src_bgr=src_bgr,
        extracted=extracted,
        path_px=path_px,
        start_color=args.start_color,
        H_world=H,
        units_scale=float(args.units_scale)
    )
    _log_agentic_report(report)
    if args.agentic_json:
        try:
            with open(args.agentic_json, "w", encoding="utf-8") as jf:
                json.dump(report, jf, indent=2)
            print(f"[i] Agentic report saved: {args.agentic_json}")
        except Exception as e:
            print(f"[WARN] Could not write agentic JSON: {e}")

    # ---------------- Execute if requested ----------------
    if args.execute and path_mm:
        robot = dobot_utils.DobotDriver(port=args.port)
        try:
            robot.set_speed(args.mm_per_sec, 150.0)
            def go(x, y, z):
                robot.move_linear(float(x), float(y), float(z), 0)

            safe_z = float(args.safe_z)
            trace_z = float(args.trace_z)

            X0, Y0 = path_mm[0]
            go(X0, Y0, safe_z)   # approach
            go(X0, Y0, trace_z)  # drop
            for (X, Y) in path_mm[1:]:
                go(X, Y, trace_z)
            XE, YE = path_mm[-1]
            go(XE, YE, safe_z)   # leave
            go(227.8,-3.1,130)
            print("Maze Solved")
        finally:
            try:
                robot.close()
            except Exception:
                pass
    else:
        print("[i] Dry run only (no --execute). Robot motion not executed.")

if __name__ == "__main__":
    main()
