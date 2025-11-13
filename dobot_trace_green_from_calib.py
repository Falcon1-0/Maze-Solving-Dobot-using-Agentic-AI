#!/usr/bin/env python3
"""
dobot_trace_green_from_calib.py (UPDATED: --solution-image mode)
----------------------------------------------------------------

NEW:
  • --solution-image IMAGE.png
    Treat IMAGE.png as an overlay with the GREEN solution already drawn.
    Extract its green polyline (pixel coords) → map to mm via homography → trace.

EXISTING:
  • --path-csv (mm or pixel) → mm → trace
  • --solver-file (import + run) → solver CSV (pixels) → mm → trace
  • legacy --image/--camera green extraction → mm → trace

USAGE (solution image):
-----------------------
python dobot_trace_green_from_calib.py \
  --solution-image solver_overlay.png \
  --calib-yaml field_calib.yml \
  --spacing-mm 5 \
  --safe-z 40 --trace-z 12 \
  --mm-per-sec 60 \
  --execute

TIPS:
  • Dry-run first (omit --execute) and inspect waypoints_mm.csv
  • If the overlay green is a different hue, tweak --hsv-lower/--hsv-upper

DEPENDENCIES
------------
pip install opencv-contrib-python numpy pyyaml pydobot
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import importlib.util
from typing import List, Tuple, Optional

import numpy as np
import cv2

try:
    import yaml
except Exception:
    yaml = None  # Allow JSON-only environment


# ---------------------- Calibration / Homography -----------------------------

def load_homography(calib_path: str) -> np.ndarray:
    """
    Load a 3x3 homography matrix H (pixel -> world) from a YAML or JSON file.
    Accepted keys: "homography" or "H".
    """
    if not calib_path:
        raise ValueError("--calib-yaml is required for pixel→mm mapping.")
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    with open(calib_path, "r") as f:
        if calib_path.lower().endswith(".json"):
            data = json.load(f)
        else:
            if yaml is None:
                # attempt JSON parse anyway
                try:
                    data = json.load(f)
                except Exception:
                    raise RuntimeError("PyYAML not installed and file is not valid JSON. Install with: pip install pyyaml")
            else:
                data = yaml.safe_load(f)
    H = None
    if isinstance(data, dict):
        if "homography" in data:
            H = np.array(data["homography"], dtype=float)
        elif "H" in data:
            H = np.array(data["H"], dtype=float)
    if H is None or H.shape != (3, 3):
        raise ValueError("Could not find a 3x3 homography matrix under keys 'homography' or 'H'.")
    return H


def apply_homography(H: np.ndarray, pts_xy: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return [(float(x), float(y)) for (x, y) in out]


def decimate_path_mm(xy_mm: List[Tuple[float, float]], spacing_mm: float, dedupe_eps_mm: float = 0.3) -> List[Tuple[float, float]]:
    """Uniformly sample the path at ~spacing_mm along arclength (keep endpoints)."""
    if spacing_mm is None or spacing_mm <= 0:
        # Only deduplicate
        if not xy_mm:
            return xy_mm
        cleaned = [tuple(xy_mm[0])]
        for p in xy_mm[1:]:
            if np.linalg.norm(np.array(cleaned[-1]) - np.array(p)) > dedupe_eps_mm:
                cleaned.append(tuple(p))
        return cleaned

    if len(xy_mm) < 2:
        return xy_mm
    out = [np.array(xy_mm[0], float)]
    acc = 0.0
    for i in range(1, len(xy_mm)):
        v = np.array(xy_mm[i], float) - np.array(xy_mm[i-1], float)
        L = float(np.linalg.norm(v))
        if L < 1e-9:
            continue
        d = v / L
        s = acc
        while s + spacing_mm <= L:
            out.append(out[-1] + d * spacing_mm)
            s += spacing_mm
        acc = (s + L) % spacing_mm
        out.append(np.array(xy_mm[i], float))
    # Deduplicate very-near points
    cleaned = [tuple(out[0])]
    for p in out[1:]:
        if np.linalg.norm(np.array(cleaned[-1]) - p) > dedupe_eps_mm:
            cleaned.append(tuple(p))
    return cleaned


# ---------------------- GREEN Path Extraction --------------------------------

def threshold_green_hsv(bgr: np.ndarray,
                        lower=(35, 40, 40),
                        upper=(85, 255, 255),
                        open_iters=1,
                        close_iters=2) -> np.ndarray:
    """
    Simple HSV threshold for green paint/marker.
    Returns a binary mask (uint8, 0/255).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if open_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))
    if close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    return mask


def skeletonize_morph(mask: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization using iterative erosion + opening.
    Returns a skeleton mask (uint8, 0/255).
    """
    skel = np.zeros_like(mask, dtype=np.uint8)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = (mask > 0).astype(np.uint8) * 255
    while True:
        eroded = cv2.erode(img, elem)
        temp = cv2.dilate(eroded, elem)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Select the largest connected component in a binary mask.
    Input:  mask can be 0/255 uint8 or any array; >0 is treated as foreground.
    Output: 0/255 uint8 mask of the largest component (or the original if none).
    """
    bin8 = (mask > 0).astype(np.uint8)          # <-- ensure CV_8U (OpenCV 4.12 fix)
    num, labels = cv2.connectedComponents(bin8, connectivity=8)
    if num <= 1:                                # only background
        return (bin8 * 255).astype(np.uint8)

    # count pixels for each label; label 0 is background
    counts = np.bincount(labels.ravel())
    best_label = 1 + int(np.argmax(counts[1:]))      # largest non-zero label

    out = np.zeros_like(bin8, dtype=np.uint8)
    out[labels == best_label] = 255
    return out


def extract_ordered_path_from_skeleton(skel: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a 1-pixel wide skeleton mask to an ordered list of (x, y) pixel coordinates.
    Works for paths with 0 or 2 endpoints (loops or simple curves). For loops, returns a loop.
    """
    pts = np.column_stack(np.where(skel > 0))  # (N, 2) as (y, x)
    if pts.shape[0] == 0:
        return []

    # Build a lookup of neighbors for each pixel
    pts_set = set(map(tuple, pts.tolist()))
    nbrs_map = {}
    for (y, x) in pts_set:
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in pts_set:
                    nbrs.append(q)
        nbrs_map[(y, x)] = nbrs

    # Find endpoints: degree == 1
    endpoints = [p for p, ns in nbrs_map.items() if len(ns) == 1]

    # Choose a starting point
    if len(endpoints) >= 1:
        start = endpoints[0]
    else:
        # Loop case: pick the lexicographically smallest pixel to break the loop deterministically
        start = min(pts_set)

    # Walk the skeleton
    ordered = []
    visited = set()
    prev = None
    curr = start
    while True:
        ordered.append((curr[1], curr[0]))  # append (x, y)
        visited.add(curr)
        # pick next neighbor not equal to prev and not visited
        candidates = [q for q in nbrs_map[curr] if q != prev and q not in visited]
        if not candidates:
            # if loop and we haven't visited all, jump to a new branch (unlikely for a single path)
            remaining = pts_set - visited
            if remaining:
                curr = min(remaining)
                prev = None
                continue
            break
        # choose the neighbor closest to current direction (prefer straightness)
        if prev is not None:
            v = (curr[0] - prev[0], curr[1] - prev[1])
            def score(q):
                w = (q[0] - curr[0], q[1] - curr[1])
                return -(v[0]*w[0] + v[1]*w[1])  # more negative is better (maximize dot)
            candidates.sort(key=score)
        nextp = candidates[0]
        prev, curr = curr, nextp
    return ordered


def extract_green_path_pixels(bgr: np.ndarray,
                              hsv_lower=(35, 40, 40),
                              hsv_upper=(85, 255, 255),
                              min_area_px=400,
                              open_iters=1,
                              close_iters=2,
                              save_debug=False,
                              debug_prefix="debug") -> List[Tuple[float, float]]:
    """
    Returns an ordered list of pixel (x, y) along the centerline path of the largest green component.
    """
    mask = threshold_green_hsv(bgr, hsv_lower, hsv_upper, open_iters, close_iters)
    # Keep only the largest blob
    comp = largest_component(mask)
    if save_debug:
        cv2.imwrite(f"{debug_prefix}_mask.png", mask)
        cv2.imwrite(f"{debug_prefix}_component.png", comp)

    if cv2.countNonZero(comp) < int(min_area_px):
        raise RuntimeError("Green area too small; adjust thresholds or min_area_px.")

    skel = skeletonize_morph(comp)
    if save_debug:
        cv2.imwrite(f"{debug_prefix}_skeleton.png", skel)

    path_xy = extract_ordered_path_from_skeleton(skel)
    if not path_xy:
        raise RuntimeError("Failed to extract path from skeleton; check image/thresholds.")

    # Create an overlay for visualization if requested
    if save_debug:
        vis = bgr.copy()
        for i, (x, y) in enumerate(path_xy):
            cv2.circle(vis, (int(x), int(y)), 1, (0, 0, 255), -1)
            if i % 25 == 0:
                cv2.putText(vis, str(i), (int(x)+2, int(y)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        cv2.imwrite(f"{debug_prefix}_path_overlay.png", vis)

    return path_xy


# ---------------------- CSV path loading -------------------------------------

def _parse_col_selector(sel: Optional[str], header: Optional[List[str]]) -> Optional[int]:
    """Return a column index for the selector; selector can be int string or header name. None -> None."""
    if sel is None:
        return None
    sel = str(sel).strip()
    # Try as integer index
    try:
        idx = int(sel)
        return idx
    except Exception:
        pass
    if header is None:
        return None
    names = [h.strip().lower() for h in header]
    target = sel.lower()
    if target in names:
        return names.index(target)
    return None


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def load_xy_from_csv(path: str,
                     colx: Optional[str] = None,
                     coly: Optional[str] = None) -> List[Tuple[float, float]]:
    """
    Load a list of (x, y) pairs from a CSV. Attempts to auto-detect columns if not specified.
    - If the file has a header, common names like X,Y,X_px,Y_px,X_mm,Y_mm,u,v will be used.
    - Otherwise, the first two numeric columns are used.
    You can override with --csv-x/--csv-y (name or 0-based index).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", newline="") as f:
        # Try DictReader for header
        peek = f.readline()
        f.seek(0)
        has_header = any(c.isalpha() for c in peek)
        if has_header:
            dr = csv.DictReader(f)
            header = dr.fieldnames or []
            # Resolve columns
            idx_x = _parse_col_selector(colx, header)
            idx_y = _parse_col_selector(coly, header)
            # Auto names
            if idx_x is None or idx_y is None:
                candidates = [
                    ("X_mm","Y_mm"), ("x_mm","y_mm"),
                    ("X","Y"), ("x","y"),
                    ("u","v"), ("U","V"),
                    ("X_px","Y_px"), ("x_px","y_px"),
                ]
                mapping = {h.strip().lower(): h for h in header}
                for (cx, cy) in candidates:
                    if cx.lower() in mapping and cy.lower() in mapping:
                        colx_name = mapping[cx.lower()]
                        coly_name = mapping[cy.lower()]
                        idx_x = header.index(colx_name)
                        idx_y = header.index(coly_name)
                        break
            if idx_x is None or idx_y is None:
                # Fallback: pick first two columns
                idx_x, idx_y = 0, 1
            # Read rows
            rows = []
            for row in dr:
                if row is None:
                    continue
                vals = [row.get(h, "") for h in header]
                if len(vals) <= max(idx_x, idx_y):
                    continue
                sx, sy = vals[idx_x], vals[idx_y]
                if _is_number(sx) and _is_number(sy):
                    rows.append((float(sx), float(sy)))
        else:
            rd = csv.reader(f)
            rows = []
            for row in rd:
                if not row:
                    continue
                # Find first two numeric entries
                nums = [float(v) for v in row if _is_number(v)]
                if len(nums) >= 2:
                    rows.append((nums[0], nums[1]))
    if not rows:
        raise RuntimeError(f"No numeric (x,y) pairs found in CSV: {path}")
    return rows


# ---------------------- Solver integration -----------------------------------

def import_solver_module(solver_path: str):
    spec = importlib.util.spec_from_file_location("user_solver", solver_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import solver from: {solver_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "solve"):
        raise RuntimeError(f"'{solver_path}' does not expose a solve(image_path, ...) function.")
    return mod


def run_solver_to_csv(solver_path: str,
                      image_path: str,
                      overlay_out: str = "solver_overlay.png",
                      csv_out: str = "solver_path_px.csv",
                      debug_dir: Optional[str] = None,
                      crop_out: Optional[str] = None,
                      click_select: bool = True,
                      gemini_first: bool = False,
                      gemini_key: str = "",
                      gemini_model: str = "gemini-1.5-flash",
                      gemini_temperature: float = 0.1,
                      gemini_max_points: int = 1600) -> str:
    """
    Import the user's solver module and run its solve(...), returning the path to the created CSV.
    The solver is expected to draw the green route onto overlay_out and write csv_out with (x_px,y_px).
    """
    mod = import_solver_module(solver_path)
    res = mod.solve(
        image_path,
        overlay_out,
        out_csv=csv_out,
        debug_dir=debug_dir,
        crop_out=crop_out,
        click_select=click_select,
        gemini_first=gemini_first,
        gemini_key=gemini_key,
        gemini_model=gemini_model,
        gemini_temperature=gemini_temperature,
        gemini_max_points=gemini_max_points
    )
    # The solver returns a dict; we can print a summary if desired.
    if isinstance(res, dict):
        print(f"[solver] n_points={res.get('n_points','?')} segments={res.get('segments','?')} "
              f"start={res.get('start_px','?')} end={res.get('end_px','?')} "
              f"solver={res.get('solver','?')}")
    if not os.path.exists(csv_out):
        raise RuntimeError("Solver did not produce the expected CSV.")
    return csv_out


# ---------------------- Dobot wrapper ----------------------------------------

class DobotDriver:
    def __init__(self, port="/dev/ttyACM0"):
        try:
            import pydobot
        except Exception as e:
            raise RuntimeError("pydobot is required to control the Dobot. Install with: pip install pydobot") from e
        self.dev = pydobot.Dobot(port=port)

    def set_speed(self, v=50.0, r=100.0):
        try:
            self.dev.speed(float(v), float(r))
        except Exception:
            # Some firmware uses different API; ignore if not supported
            pass

    def move_linear(self, x, y, z, r=0):
        self.dev.move_to(float(x), float(y), float(z), float(r))

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


# ---------------------- Main --------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Trace a path with a Dobot using a calibration YAML (homography), a solver output, a CSV path, or a solution overlay image of the GREEN line.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--image", type=str, help="Input image for the solver / legacy green extraction.")
    src.add_argument("--camera", type=int, help="Camera index to capture a single frame (e.g., 0).")
    src.add_argument("--solution-image", type=str, help="Use a pre-rendered solution overlay image with a GREEN path.")

    # Solver import mode
    ap.add_argument("--solver-file", type=str, default=None, help="Path to a solver .py (must expose solve(image_path, ...) and write CSV of (x_px,y_px)).")
    ap.add_argument("--solver-overlay", type=str, default="solver_overlay.png", help="Where to save the solver's overlay image.")
    ap.add_argument("--solver-csv-out", type=str, default="solver_path_px.csv", help="Where the solver should write pixels CSV.")
    ap.add_argument("--solver-debug", type=str, default="", help="Optional directory for solver debug artifacts.")
    ap.add_argument("--solver-crop", type=str, default="", help="Optional solver maze ROI crop output PNG.")
    ap.add_argument("--solver-no-click", action="store_true", help="Disable solver's interactive START/END selection.")
    ap.add_argument("--solver-gemini-first", action="store_true", help="Ask Gemini first inside the solver (requires API key).")
    ap.add_argument("--solver-gemini-key", type=str, default="", help="Gemini API key (or set GEMINI_API_KEY env).")
    ap.add_argument("--solver-gemini-model", type=str, default="gemini-1.5-flash")
    ap.add_argument("--solver-gemini-temp", type=float, default=0.1)
    ap.add_argument("--solver-gemini-max-points", type=int, default=1600)

    # CSV mode
    ap.add_argument("--path-csv", type=str, default=None, help="CSV file with path points (columns X,Y). If --csv-type=pixel, they are pixels and require --calib-yaml. If --csv-type=mm, they are millimeters.")
    ap.add_argument("--csv-type", choices=["mm","pixel"], default="mm", help="Unit/type of points in --path-csv.")
    ap.add_argument("--csv-x", type=str, default=None, help="Column name or 0-based index for X (optional).")
    ap.add_argument("--csv-y", type=str, default=None, help="Column name or 0-based index for Y (optional).")

    # Homography
    ap.add_argument("--calib-yaml", type=str, required=True,
                    help="Calibration homography YAML/JSON to map pixel→world before tracing "
                         "(required if using solver, image/camera, solution-image, or --csv-type=pixel).")
    ap.add_argument("--units-scale", type=float, default=1.0, help="Scale factor to convert world units to mm (e.g., 1000 if calibration is in meters).")

    # HSV green threshold & morphology
    ap.add_argument("--hsv-lower", type=str, default="35,40,40", help="Lower HSV for green (e.g., '35,40,40').")
    ap.add_argument("--hsv-upper", type=str, default="85,255,255", help="Upper HSV for green (e.g., '85,255,255').")
    ap.add_argument("--open-iters", type=int, default=1, help="Morphological open iterations.")
    ap.add_argument("--close-iters", type=int, default=2, help="Morphological close iterations.")
    ap.add_argument("--min-area-px", type=int, default=400, help="Minimum area (px) to accept as green path.")

    # Path post-processing
    ap.add_argument("--spacing-mm", type=float, default=5.0, help="Point spacing for the executed path in mm (<=0 keeps original spacing).")
    ap.add_argument("--dedupe-eps-mm", type=float, default=0.3, help="Minimum spacing to keep points distinct if spacing-mm<=0.")
    ap.add_argument("--swap-xy", action="store_true", help="Swap X/Y after mapping (frame alignment).")
    ap.add_argument("--flip-x", action="store_true", help="Mirror across a vertical line through the first point.")
    ap.add_argument("--flip-y", action="store_true", help="Mirror across a horizontal line through the first point.")
    ap.add_argument("--reverse", action="store_true", help="Reverse the path order.")
    ap.add_argument("--offset-x-mm", type=float, default=0.0, help="Add this X offset (mm) to all waypoints after mapping.")
    ap.add_argument("--offset-y-mm", type=float, default=0.0, help="Add this Y offset (mm) to all waypoints after mapping.")

    # Robot & execution
    ap.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port of the Dobot.")
    ap.add_argument("--safe-z", type=float, default=40.0, help="Approach/leave Z (mm).")
    ap.add_argument("--trace-z", type=float, default=12.0, help="Z height while tracing (mm).")
    ap.add_argument("--mm-per-sec", type=float, default=50.0, help="Linear motion speed (approx, firmware dependent).")
    ap.add_argument("--execute", action="store_true", help="If set, actually move the robot. Otherwise: dry run only.")
    ap.add_argument("--out-csv-mm", type=str, default="waypoints_mm.csv", help="CSV to write robot-space waypoints actually used.")

    # Debug (legacy green mode)
    ap.add_argument("--save-debug", action="store_true", help="Save debug images for GREEN extraction (mask, component, skeleton, overlay).")
    ap.add_argument("--debug-prefix", type=str, default="debug", help="Prefix for debug image files.")

    args = ap.parse_args()

    # Determine source of (x,y) path in pixels or mm
    path_mm: List[Tuple[float, float]] = []

    # Priority: CSV > solution-image > solver-file > legacy image/camera
    if args.path_csv:
        # Load from CSV
        pts = load_xy_from_csv(args.path_csv, colx=args.csv_x, coly=args.csv_y)
        if args.csv_type == "pixel":
            # Need homography
            H = load_homography(args.calib_yaml)
            world = apply_homography(H, pts)
            # Optional unit conversion
            world = [(x*args.units_scale, y*args.units_scale) for (x, y) in world]
        else:
            # Already mm
            world = [(float(x), float(y)) for (x, y) in pts]
        path_mm = world

    elif args.solution_image:
        # Directly extract the green path from a pre-rendered overlay image
        bgr = cv2.imread(args.solution_image, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"Could not read solution image: {args.solution_image}")

        def _parse_triplet(s):
            parts = [int(float(x.strip())) for x in s.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Bad HSV triplet: {s}")
            return tuple(np.clip(parts, 0, 255).tolist())
        hsv_lower = _parse_triplet(args.hsv_lower)
        hsv_upper = _parse_triplet(args.hsv_upper)

        # Extract green pixels as path
        path_px = extract_green_path_pixels(
            bgr,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            min_area_px=int(args.min_area_px),
            open_iters=int(args.open_iters),
            close_iters=int(args.close_iters),
            save_debug=bool(args.save_debug),
            debug_prefix=args.debug_prefix
        )

        # Pixels -> mm
        H = load_homography(args.calib_yaml)
        world = apply_homography(H, path_px)
        path_mm = [(x*args.units_scale, y*args.units_scale) for (x, y) in world]

    elif args.solver_file:
        # Prepare image for solver
        if args.image:
            solver_image = args.image
            if not os.path.exists(solver_image):
                raise SystemExit(f"Could not read image: {solver_image}")
        elif args.camera is not None:
            cap = cv2.VideoCapture(int(args.camera))
            if not cap.isOpened():
                raise SystemExit(f"Could not open camera index {args.camera}")
            for _ in range(6):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                raise SystemExit("Failed to capture a frame from camera.")
            solver_image = "frame_for_solver.jpg"
            cv2.imwrite(solver_image, frame)
            if args.save_debug:
                cv2.imwrite(f"{args.debug_prefix}_frame.png", frame)
        else:
            raise SystemExit("Provide --image or --camera when using --solver-file.")

        # Run solver to produce pixel CSV
        csv_px_path = run_solver_to_csv(
            solver_path=args.solver_file,
            image_path=solver_image,
            overlay_out=args.solver_overlay,
            csv_out=args.solver_csv_out,
            debug_dir=(args.solver_debug or None) if args.solver_debug else None,
            crop_out=(args.solver_crop or None) if args.solver_crop else None,
            click_select=(not args.solver_no_click),
            gemini_first=bool(args.solver_gemini_first),
            gemini_key=args.solver_gemini_key,
            gemini_model=args.solver_gemini_model,
            gemini_temperature=float(args.solver_gemini_temp),
            gemini_max_points=int(args.solver_gemini_max_points)
        )

        # Load the pixel path from the solver CSV
        pts_px = load_xy_from_csv(csv_px_path)
        if not pts_px:
            raise SystemExit("Solver CSV has no points.")

        # Map to mm via homography
        H = load_homography(args.calib_yaml)
        path_world = apply_homography(H, pts_px)
        path_mm = [(x*args.units_scale, y*args.units_scale) for (x, y) in path_world]

    else:
        # Legacy image/camera mode (detect green directly from a fresh frame)
        if args.image:
            bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
            if bgr is None:
                raise SystemExit(f"Could not read image: {args.image}")
        elif args.camera is not None:
            cap = cv2.VideoCapture(int(args.camera))
            if not cap.isOpened():
                raise SystemExit(f"Could not open camera index {args.camera}")
            for _ in range(6):
                cap.read()
            ok, frame = cap.read()
            cap.release()
            if not ok or frame is None:
                raise SystemExit("Failed to capture a frame from camera.")
            bgr = frame
            if args.save_debug:
                cv2.imwrite(f"{args.debug_prefix}_frame.png", bgr)
        else:
            raise SystemExit("Provide --path-csv OR --solution-image OR --solver-file OR (--image or --camera).")

        # Parse HSV thresholds
        def _parse_triplet(s):
            parts = [int(float(x.strip())) for x in s.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Bad HSV triplet: {s}")
            return tuple(np.clip(parts, 0, 255).tolist())
        hsv_lower = _parse_triplet(args.hsv_lower)
        hsv_upper = _parse_triplet(args.hsv_upper)

        # Need homography for image/camera mode (mapping pixels->mm)
        H = load_homography(args.calib_yaml)

        # 1) Extract green path (pixel coords, ordered)
        path_px = extract_green_path_pixels(
            bgr,
            hsv_lower=hsv_lower,
            hsv_upper=hsv_upper,
            min_area_px=int(args.min_area_px),
            open_iters=int(args.open_iters),
            close_iters=int(args.close_iters),
            save_debug=bool(args.save_debug),
            debug_prefix=args.debug_prefix
        )

        # 2) Pixel -> world (mm) via homography
        world = apply_homography(H, path_px)
        world = [(x*args.units_scale, y*args.units_scale) for (x, y) in world]
        path_mm = world

    # Optional axis adjustments & reverse
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

    # Apply offsets
    if args.offset_x_mm or args.offset_y_mm:
        dx, dy = float(args.offset_x_mm), float(args.offset_y_mm)
        path_mm = [(x+dx, y+dy) for (x, y) in path_mm]

    # Decimate / dedupe
    path_mm = decimate_path_mm(path_mm, spacing_mm=float(args.spacing_mm), dedupe_eps_mm=float(args.dedupe_eps_mm))

    # Write final used path
    with open(args.out_csv_mm, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["X_mm", "Y_mm"])
        for (x, y) in path_mm:
            w.writerow([f"{x:.3f}", f"{y:.3f}"])
    print(f"[i] Final robot path: {len(path_mm)} points -> {args.out_csv_mm}")
    if path_mm:
        print(f"    Start: ({path_mm[0][0]:.2f}, {path_mm[0][1]:.2f})  End: ({path_mm[-1][0]:.2f}, {path_mm[-1][1]:.2f})")

    # 6) Execute robot motion (safely) if requested
    if args.execute and path_mm:
        robot = DobotDriver(port=args.port)
        try:
            robot.set_speed(args.mm_per_sec, 100.0)
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
            print("[✓] Trace completed.")
        finally:
            try:
                robot.close()
            except Exception:
                pass
    else:
        print("[i] Dry run only (no --execute). Robot motion not executed.")

if __name__ == "__main__":
    main()
