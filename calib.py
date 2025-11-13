#!/usr/bin/env python3
"""
ArUco Homography Calibration — Pixel ↔ World (planar)

Given FOUR ArUco markers that define a plane (e.g., the corners of your field),
this script detects them in a single image (or a snapshot from a camera),
solves for a 3×3 homography H that maps pixel coordinates → real-world planar
coordinates, and saves a calibration file (YAML). The inverse H_inv is also
saved (world → pixel).

You can choose to define the four correspondences using:
  - the centers of the four markers ("centers" mode), or
  - the "outer" corner of each marker ("outer_corners" mode), where the corner
    pointing outward from the board centroid is selected automatically.

USAGE EXAMPLES
--------------

1) From a still image where markers with IDs 7, 3, 11, 42 are the four corners
   of a rectangle with real-world size 2.0 m × 1.0 m laid out as:
       id=7  → (0, 0)       id=3  → (2.0, 0)
       id=11 → (2.0, 1.0)   id=42 → (0, 1.0)

   python aruco_homography_calibrate.py \
       --image frame.jpg \
       --ids 7 3 11 42 \
       --world "0,0; 2.0,0; 2.0,1.0; 0,1.0" \
       --dict 4X4_50 \
       --mode outer_corners \
       --save homography.yml \
       --annotated annotated.jpg

2) Grab a single frame from camera index 0 and use marker CENTERS:
   python aruco_homography_calibrate.py \
       --camera 0 \
       --ids 25 8 14 19 \
       --world "0,0; 1.0,0; 1.0,1.0; 0,1.0" \
       --dict 5X5_100 \
       --mode centers \
       --save field_calib.yml

3) If you have camera intrinsics/distortion (highly recommended), pass them to
   undistort the image before solving H (YAML layout shown below):
   python aruco_homography_calibrate.py \
       --image frame.jpg \
       --camera-yaml intrinsics.yml \
       --ids 10 11 12 13 \
       --world "0,0; 3.5,0; 3.5,2.0; 0,2.0" \
       --mode outer_corners

INTRINSICS YAML (example)
-------------------------
camera_matrix:
  - [fx, 0, cx]
  - [0, fy, cy]
  - [0,  0,  1]
dist_coeffs: [k1, k2, p1, p2, k3]

NOTES
-----
• The order of --ids must correspond to the order of --world points.
• Units of --world are arbitrary (m, cm, mm, in) but must be consistent
  across your project.
• For best accuracy: the plane must be truly planar, markers should be well
  separated (forming a convex quad), and the image should be undistorted.
• This estimates a planar mapping; it does NOT by itself give height (Z).
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
import math
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple
import time

import numpy as np

try:
    import cv2
    from cv2 import aruco as aruco
except Exception as e:  # pragma: no cover
    print("ERROR: This script requires OpenCV with contrib modules.", file=sys.stderr)
    print("Install with: pip install opencv-contrib-python", file=sys.stderr)
    raise

try:
    import yaml
except Exception:
    yaml = None  # Will fallback to JSON if PyYAML not available.


# ---------------------------- Utilities -------------------------------------

def _parse_point_list(spec: str, expected_n: int | None = None) -> np.ndarray:
    """
    Parse a list of "x,y" pairs separated by ';' or whitespace.
    Example: "0,0; 2.0,0; 2.0,1.0; 0,1.0"
    Returns an (N,2) float64 array.
    """
    if spec is None or not str(spec).strip():
        raise ValueError("Point-list string is empty.")
    # Split on ';' or whitespace
    raw = []
    for token in spec.replace(';', ' ').split():
        if ',' not in token:
            raise ValueError(f"Bad point token '{token}', expected 'x,y'.")
        x_s, y_s = token.split(',', 1)
        raw.append((float(x_s), float(y_s)))
    pts = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
    if expected_n is not None and len(pts) != expected_n:
        raise ValueError(f"Expected {expected_n} points but got {len(pts)} in '{spec}'.")
    return pts


def _aruco_dict_from_name(name: str):
    """
    Map a human-friendly dictionary name to an OpenCV aruco dictionary.
    """
    name = (name or "").upper().replace("-", "_").strip()
    # Common predefined dictionaries
    mapping = {
        "4X4_50": aruco.DICT_4X4_50,
        "4X4_100": aruco.DICT_4X4_100,
        "4X4_250": aruco.DICT_4X4_250,
        "4X4_1000": aruco.DICT_4X4_1000,
        "5X5_50": aruco.DICT_5X5_50,
        "5X5_100": aruco.DICT_5X5_100,
        "5X5_250": aruco.DICT_5X5_250,
        "5X5_1000": aruco.DICT_5X5_1000,
        "6X6_50": aruco.DICT_6X6_50,
        "6X6_100": aruco.DICT_6X6_100,
        "6X6_250": aruco.DICT_6X6_250,
        "6X6_1000": aruco.DICT_6X6_1000,
        "7X7_50": aruco.DICT_7X7_50,
        "7X7_100": aruco.DICT_7X7_100,
        "7X7_250": aruco.DICT_7X7_250,
        "7X7_1000": aruco.DICT_7X7_1000,
        "ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
    }
    # AprilTag families (available in newer OpenCV builds)
    if hasattr(aruco, "DICT_APRILTAG_36h11"):
        mapping["APRILTAG_36H11"] = aruco.DICT_APRILTAG_36h11
    if name not in mapping:
        raise ValueError(f"Unknown aruco dictionary name '{name}'. Choices: {', '.join(mapping.keys())}")
    return aruco.getPredefinedDictionary(mapping[name])


def _load_intrinsics_yaml(path: str) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load camera intrinsics/distortion from YAML/JSON.
    Returns (K, dist) or (None, None) if not available.
    Accepts keys: camera_matrix or K, and dist_coeffs or D.
    """
    if not path:
        return None, None
    with open(path, "r") as f:
        if path.lower().endswith(".json"):
            data = json.load(f)
        else:
            if yaml is None:
                raise RuntimeError("PyYAML not installed; cannot read YAML intrinsics. Install with: pip install pyyaml")
            data = yaml.safe_load(f)
    def _as_mat(obj):
        arr = np.asarray(obj, dtype=np.float64)
        return arr
    K = None
    D = None
    if "camera_matrix" in data:
        K = _as_mat(data["camera_matrix"])
    elif "K" in data:
        K = _as_mat(data["K"])
    if "dist_coeffs" in data:
        D = _as_mat(data["dist_coeffs"])
    elif "D" in data:
        D = _as_mat(data["D"])
    if K is not None:
        K = K.reshape(3, 3)
    if D is not None:
        D = D.reshape(-1, 1)
    return K, D


def _undistort_if_needed(img: np.ndarray, K: np.ndarray | None, D: np.ndarray | None) -> np.ndarray:
    if K is None or D is None:
        return img
    return cv2.undistort(img, K, D, None, K)


def _detect_markers(img: np.ndarray, dictionary, adaptive_thresh: bool = True):
    # Create detector parameters
    params = aruco.DetectorParameters()
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(img)
    else:
        corners, ids, _ = aruco.detectMarkers(img, dictionary, parameters=params)
    # Normalize shapes
    ids = ids.flatten().astype(int) if ids is not None and len(ids) else np.array([], dtype=int)
    # corners: list of N arrays of shape (1,4,2) -> convert to (4,2)
    corners = [c.reshape(4, 2) for c in corners] if corners else []
    return corners, ids


def _gather_required(corners: List[np.ndarray], ids: np.ndarray, required_ids: Sequence[int]) -> Dict[int, np.ndarray]:
    """
    Return dict {id: corners(4,2)} for required IDs if found.
    Raises if not all present.
    """
    found = {}
    id_to_idx = {int(i): k for k, i in enumerate(ids)}
    missing = []
    for rid in required_ids:
        if rid in id_to_idx:
            found[rid] = corners[id_to_idx[rid]]
        else:
            missing.append(int(rid))
    if missing:
        raise RuntimeError(f"Did not detect all required markers. Missing IDs: {missing}. "
                           f"Detected IDs: {ids.tolist()}")
    return found


def _marker_centers(mark_corners: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    return {mid: c.mean(axis=0) for mid, c in mark_corners.items()}


def _choose_outer_corners(mark_corners: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    For each marker, select the corner that points 'outward' from the board centroid.
    Returns dict {id: (2,) ndarray} with one pixel point per marker.
    """
    centers = _marker_centers(mark_corners)
    board_center = np.mean(np.stack(list(centers.values()), axis=0), axis=0)
    chosen = {}
    for mid, c4 in mark_corners.items():
        c4 = np.asarray(c4, dtype=np.float64).reshape(4, 2)
        c_center = centers[mid]
        outward = c_center - board_center
        if np.linalg.norm(outward) < 1e-6:
            # Fallback: use farthest-from-board-center corner
            idx = int(np.argmax(np.linalg.norm(c4 - board_center, axis=1)))
        else:
            # Choose corner whose vector from marker center aligns most with outward
            vecs = c4 - c_center  # (4,2)
            dots = vecs @ outward.reshape(2, 1)  # (4,1)
            idx = int(np.argmax(dots))
        chosen[mid] = c4[idx]
    return chosen


def _compute_homography(pixels: np.ndarray, world: np.ndarray, ransac_thresh: float = 3.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute H using RANSAC and return (H, H_inv, rms_error).
    pixels: (N,2), world: (N,2)
    """
    assert pixels.shape == world.shape and pixels.shape[0] >= 4
    H, mask = cv2.findHomography(pixels.astype(np.float64), world.astype(np.float64),
                                 method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000, confidence=0.995)
    if H is None:
        raise RuntimeError("cv2.findHomography failed. Check your points.")
    H_inv = np.linalg.inv(H)
    # RMS reprojection error (pixel->world)
    reproj = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((reproj - world) ** 2, axis=1))))
    return H, H_inv, rms


def _save_calibration(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".json") or yaml is None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)


def _annotate_image(img: np.ndarray,
                    mark_corners: Dict[int, np.ndarray],
                    chosen_points: Dict[int, np.ndarray],
                    world_points: np.ndarray,
                    id_order: Sequence[int]) -> np.ndarray:
    out = img.copy()
    # Draw all markers and their outlines
    for mid, c4 in mark_corners.items():
        c4i = np.round(c4).astype(int)
        cv2.polylines(out, [c4i], isClosed=True, thickness=2, color=(0, 255, 0))
        cv2.putText(out, f"ID {mid}", tuple(c4i[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    # Draw chosen points & labels in the provided id/world order
    for k, mid in enumerate(id_order):
        p = chosen_points[mid]
        cv2.circle(out, tuple(np.round(p).astype(int)), 8, (0, 0, 255), -1)
        label = f"{mid} → ({world_points[k,0]:.3g},{world_points[k,1]:.3g})"
        cv2.putText(out, label, tuple((p + np.array([6, -10])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return out


# ------------------------------ Main ----------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Estimate planar homography using 4 ArUco markers and save calibration (YAML).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=str, help="Path to image file containing the 4 markers.")
    src.add_argument("--camera", type=int, help="Camera index to grab a single frame from (e.g., 0).")
    ap.add_argument("--ids", type=int, nargs=4, required=True,
                    help="Marker IDs in the SAME order as --world points (exactly four integers).")
    ap.add_argument("--world", type=str, required=True,
                    help='Real-world coords for the four points as "x,y; x,y; x,y; x,y".')
    ap.add_argument("--dict", type=str, default="4X4_50",
                    help="ArUco dictionary name (e.g., 4X4_50, 5X5_100, ARUCO_ORIGINAL).")
    ap.add_argument("--mode", type=str, choices=["centers", "outer_corners"], default="outer_corners",
                    help="Which pixel points to use per marker for the 4 correspondences.")
    ap.add_argument("--camera-yaml", type=str, default=None,
                    help="Optional camera intrinsics/distortion YAML/JSON to undistort before solving.")
    ap.add_argument("--ransac-thresh", type=float, default=3.0, help="RANSAC reprojection threshold (in world units).")
    ap.add_argument("--save", type=str, default="homography.yml", help="Output calibration file (.yml/.yaml/.json).")
    ap.add_argument("--annotated", type=str, default=None, help="Optional path to write an annotated debug image.")
    ap.add_argument("--print", dest="do_print", action="store_true",
                    help="Print the resulting H and H_inv matrices to stdout.")
    ap.add_argument("--test", type=str, default=None,
                    help='Optional list of pixel points to map, e.g. "100,200; 640,360".')

    args = ap.parse_args()

    # Load image
    if args.image:
        img = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Could not read image: {args.image}")
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if not cap.isOpened():
            raise SystemExit(f"Could not open camera index {args.camera}")
        time.sleep(10)
        ok, img = cap.read()
        cap.release()
        if not ok or img is None:
            raise SystemExit("Failed to capture a frame from camera.")

    # Undistort (optional)
    K, D = _load_intrinsics_yaml(args.camera_yaml) if args.camera_yaml else (None, None)
    img_u = _undistort_if_needed(img, K, D)

    # Detect markers
    dictionary = _aruco_dict_from_name(args.dict)
    corners, ids = _detect_markers(img_u, dictionary)

    if len(ids) < 4:
        raise SystemExit(f"Detected {len(ids)} markers, but need all four required IDs {args.ids}. "
                         f"Detected IDs: {ids.tolist()}")

    # Gather the required four markers in the order provided
    required_ids = [int(i) for i in args.ids]
    mark_corners = _gather_required(corners, ids, required_ids)

    # Select one pixel point per marker according to mode
    if args.mode == "centers":
        chosen = _marker_centers(mark_corners)  # dict id -> (2,)
    else:
        chosen = _choose_outer_corners(mark_corners)  # dict id -> (2,)

    # Build (4,2) arrays in the exact user order
    pixel_pts = np.stack([chosen[i] for i in required_ids], axis=0).astype(np.float64)
    world_pts = _parse_point_list(args.world, expected_n=4).astype(np.float64)

    # Solve homography
    H, H_inv, rms = _compute_homography(pixel_pts, world_pts, ransac_thresh=args.ransac_thresh)

    # Prepare payload
    now = _dt.datetime.now().isoformat(timespec="seconds")
    payload = {
        "created": now,
        "image_size": [int(img_u.shape[1]), int(img_u.shape[0])],  # [width, height]
        "aruco_dict": args.dict,
        "mode": args.mode,
        "required_ids": required_ids,
        "pixel_points_ordered": pixel_pts.tolist(),
        "world_points_ordered": world_pts.tolist(),
        "homography": H.tolist(),        # pixel -> world
        "homography_inv": H_inv.tolist(),# world -> pixel
        "rms_error_world_units": rms,
        "camera_intrinsics_used": bool(K is not None and D is not None),
    }
    if K is not None and D is not None:
        payload["camera_matrix"] = np.asarray(K, dtype=float).reshape(3, 3).tolist()
        payload["dist_coeffs"] = np.asarray(D, dtype=float).reshape(-1).tolist()

    _save_calibration(args.save, payload)

    print(f"[OK] Saved calibration to: {args.save}")
    print(f"     RMS reprojection error: {rms:.6f} (in world units)")

    if args.do_print:
        np.set_printoptions(precision=6, suppress=True)
        print("\nH (pixel → world):\n", H)
        print("\nH_inv (world → pixel):\n", H_inv)

    # Optional annotation
    if args.annotated:
        annotated = _annotate_image(img_u, mark_corners, chosen, world_pts, required_ids)
        cv2.imwrite(args.annotated, annotated)
        print(f"[OK] Wrote annotated image: {args.annotated}")

    # Optional test: map some pixel points
    if args.test:
        test_px = _parse_point_list(args.test, expected_n=None).astype(np.float64)
        test_world = cv2.perspectiveTransform(test_px.reshape(-1, 1, 2), H).reshape(-1, 2)
        print("\nTest mappings (pixel → world):")
        for (px, py), (wx, wy) in zip(test_px, test_world):
            print(f"  ({px:.3f}, {py:.3f})  →  ({wx:.6f}, {wy:.6f})")

    print("Done.")


if __name__ == "__main__":
    main()
