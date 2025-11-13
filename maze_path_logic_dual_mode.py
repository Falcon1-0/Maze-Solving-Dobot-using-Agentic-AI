#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
maze_path_logic_dual_mode.py
--------------------------------
Extracted, self-contained path-finding logic with dual input mode
(camera index OR image file) and a --start-color flag. The computed
path is rendered as a GREEN polyline, with the start dot GREEN and
the goal dot RED.

This module distills just the vision + planning pieces from a fuller
script (ROI isolation; corridor masks; center-biased A* with
skeleton/BFS fallbacks; clipped single green polyline drawing).

Usage examples:
  - From a saved image:
      python maze_path_logic_dual_mode.py --start-color green --image maze.jpg

  - From a webcam:
      python maze_path_logic_dual_mode.py --start-color red --camera-index 0

Outputs:
  - Saves the annotated image (default: extracted_centered_path.png).
  - Optionally previews a single window with the final path if --show is used.
  - Returns non-zero exit code on failure conditions (e.g., markers not found).

Dependencies: Python 3.8+, opencv-python, numpy
"""

import argparse, os, sys, time, math
from typing import List, Tuple, Optional
from collections import deque

import cv2
import numpy as np

# -------------------- Tunables / constants --------------------
AUTO_MARGIN_RATIO = 0.55
MIN_MARGIN_PX     = 2
CENTER_BIAS       = 40.0    # strong pull to corridor center
DRAW_THICKNESS_PX = 2
VISUAL_GAP_PX     = 2
THICK_FILTER_KERNEL  = 7
DENOISE_BLUR         = 3
TARGET_LONG_EDGE_PX  = 1200

# HSV ranges for marker detection (tuned for typical paper lighting)
HSV_GREEN_LO = (35, 40, 40);  HSV_GREEN_HI = (90, 255, 255)
HSV_RED1_LO  = (0,  60, 60);  HSV_RED1_HI  = (10, 255, 255)
HSV_RED2_LO  = (170,60, 60);  HSV_RED2_HI  = (180,255,255)
MIN_MARKER_AREA_RATIO = 0.001

# -------------------- Geometry & warping helpers --------------------
def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32).reshape(4,2)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def _auto_canny(gray):
    med = np.median(gray)
    lo = int(max(0, 0.66*med)); hi = int(min(255, 1.33*med))
    return cv2.Canny(gray, lo, hi)

def _largest_quad_from_edges(edges, min_area, prefer_square=False):
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = approx.reshape(-1,2).astype(np.float32)
            x,y,w,h = cv2.boundingRect(rect.astype(np.int32))
            ar = w / float(h) if h>0 else 1.0
            score = area * (1.0 + (1.0 - abs(ar-1.0)) * (1.0 if prefer_square else 0.0))
            if score > best_score:
                best, best_score = rect, score
    if best is None and cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area >= min_area:
            best = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    return best

def _warp_to_quad(img, src_quad, out_long_edge=TARGET_LONG_EDGE_PX):
    src = _order_quad(src_quad)
    w1 = np.linalg.norm(src[1]-src[0]); w2 = np.linalg.norm(src[2]-src[3])
    h1 = np.linalg.norm(src[3]-src[0]); h2 = np.linalg.norm(src[2]-src[1])
    w = max(int(max(w1,w2)),1); h = max(int(max(h1,h2)),1)
    if w >= h:
        out_w = out_long_edge; out_h = max(1, int(round(out_long_edge * h / w)))
    else:
        out_h = out_long_edge; out_w = max(1, int(round(out_long_edge * w / h)))
    dst = np.array([[0,0],[out_w-1,0],[out_w-1,out_h-1],[0,out_h-1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped, H, (out_w, out_h)

def isolate_maze_roi(orig_bgr):
    """
    Finds a page/board, rectifies it, then finds an inner square maze and warps to that.
    Returns: (maze_bgr, H_maze_to_orig) where H maps ROI coords -> original image coords.
    """
    H0, W0 = orig_bgr.shape[:2]
    gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
    edges = _auto_canny(cv2.GaussianBlur(gray, (5,5), 0))
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    page_quad = _largest_quad_from_edges(edges, min_area=0.20*H0*W0, prefer_square=False)
    if page_quad is None:
        return orig_bgr.copy(), np.eye(3, dtype=np.float32)

    doc_bgr, H_doc, _ = _warp_to_quad(orig_bgr, page_quad, out_long_edge=TARGET_LONG_EDGE_PX)
    doc_gray = cv2.cvtColor(doc_bgr, cv2.COLOR_BGR2GRAY)

    _, bin_bw = cv2.threshold(doc_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.bitwise_not(bin_bw)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (THICK_FILTER_KERNEL, THICK_FILTER_KERNEL))
    thick = cv2.erode(ink, k, 1); thick = cv2.dilate(thick, k, 1)

    maze_quad = _largest_quad_from_edges(cv2.Canny(thick,50,150),
                                         min_area=0.25*thick.size,
                                         prefer_square=True)
    if maze_quad is None:
        return doc_bgr, np.linalg.inv(H_doc).astype(np.float32)

    maze_bgr, H_maze, _ = _warp_to_quad(doc_bgr, maze_quad, out_long_edge=TARGET_LONG_EDGE_PX)
    H_total = np.linalg.inv(H_doc) @ np.linalg.inv(H_maze)
    return maze_bgr, H_total.astype(np.float32)

# -------------------- Binarization & corridor masks --------------------
def _invert_thresh(gray):
    gray = cv2.medianBlur(gray, DENOISE_BLUR)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 5)
    return th  # white = ink (walls/border)

def threshold_walls(maze_bgr: np.ndarray, color_mask_to_clear: Optional[np.ndarray]) -> np.ndarray:
    gray = cv2.cvtColor(maze_bgr, cv2.COLOR_BGR2GRAY)
    walls = _invert_thresh(gray)     # white = walls/border (ink)
    if color_mask_to_clear is not None:
        walls[color_mask_to_clear>0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    walls = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k, 1)
    walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, k, 1)
    return walls

def inside_open_mask(walls: np.ndarray, frame: int = 3) -> np.ndarray:
    H,W=walls.shape
    closed = walls.copy()
    cv2.rectangle(closed, (0,0), (W-1,H-1), 255, frame)  # close perimeter
    open_tmp = cv2.bitwise_not(closed)                    # white=open
    outside = open_tmp.copy()
    mask = np.zeros((H+2, W+2), np.uint8)
    cv2.floodFill(outside, mask, (0,0), 128)
    outside_mask = np.zeros_like(open_tmp)
    outside_mask[outside==128] = 255
    inside = cv2.subtract(open_tmp, outside_mask)
    return inside

def _erode_by_px(mask255: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return (mask255>0).astype(np.uint8)*255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*px+1, 2*px+1))
    return cv2.erode((mask255>0).astype(np.uint8)*255, k, iterations=1)

def _skeletonize(bin255: np.ndarray) -> np.ndarray:
    img = (bin255>0).astype(np.uint8)*255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel

# -------------------- Markers & utilities --------------------
def find_markers(bgr) -> Tuple[Optional[Tuple[int,int]], Optional[Tuple[int,int]], np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gmask = cv2.inRange(hsv, np.array(HSV_GREEN_LO), np.array(HSV_GREEN_HI))
    rmask1 = cv2.inRange(hsv, np.array(HSV_RED1_LO),  np.array(HSV_RED1_HI))
    rmask2 = cv2.inRange(hsv, np.array(HSV_RED2_LO),  np.array(HSV_RED2_HI))
    rmask = cv2.bitwise_or(rmask1, rmask2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_OPEN, k, 1)
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, k, 1)
    rmask = cv2.morphologyEx(rmask, cv2.MORPH_OPEN, k, 1)
    rmask = cv2.morphologyEx(rmask, cv2.MORPH_CLOSE, k, 1)

    H,W = bgr.shape[:2]
    min_area = MIN_MARKER_AREA_RATIO * (H*W)

    def _centroid(mask):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
        if not cnts: return None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0: return None
        cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
        return (cx,cy)

    gpt = _centroid(gmask)
    rpt = _centroid(rmask)
    color_union = cv2.bitwise_or(gmask, rmask)
        # Fallback: if no green is found, treat another non-red circle as "green"
    if gpt is None:
        # Consider any colored blob that is NOT red
        nonred = cv2.subtract(color_union, rmask)

        # Optional: clean up
        nonred = cv2.morphologyEx(nonred, cv2.MORPH_OPEN, k, 1)
        nonred = cv2.morphologyEx(nonred, cv2.MORPH_CLOSE, k, 1)

        cnts,_ = cv2.findContours(nonred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter by size and circularity (4πA/P^2 close to 1 is a circle)
        candidates = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4.0 * np.pi * area / (peri * peri)
            if circularity > 0.6:  # tweak threshold if needed (0.6–0.8)
                candidates.append((area, c))

        if candidates:
            _, c = max(candidates, key=lambda t: t[0])  # largest circular blob
            M = cv2.moments(c)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00']); cy = int(M['m01']/M['m00'])
                gpt = (cx, cy)

    return gpt, rpt, color_union

def nearest_nonzero(img, pt):
    x,y = map(int, pt)
    if 0<=x<img.shape[1] and 0<=y<img.shape[0] and img[y,x] > 0:
        return (x,y)
    H,W = img.shape
    for r in range(1, max(H,W), 2):
        xmin,xmax = max(0,x-r), min(W-1, x+r)
        ymin,ymax = max(0,y-r), min(H-1, y+r)
        roi = img[ymin:ymax+1, xmin:xmax+1]
        nz = cv2.findNonZero(roi)
        if nz is not None:
            nz = nz.reshape(-1,2)
            d2 = (nz[:,0]+xmin-x)**2 + (nz[:,1]+ymin-y)**2
            idx = int(np.argmin(d2))
            return (int(nz[idx,0]+xmin), int(nz[idx,1]+ymin))
    return (x,y)

def _bfs(binary255: np.ndarray, start: Tuple[int,int], end: Tuple[int,int], eight=True) -> List[Tuple[int,int]]:
    h,w=binary255.shape; sx,sy=start; ex,ey=end
    if not (0<=sx<w and 0<=sy<h and 0<=ex<w and 0<=ey<h): return []
    if binary255[sy,sx]==0 or binary255[ey,ex]==0: return []
    visited=np.zeros((h,w),np.uint8)
    parent=np.full((h,w,2),-1,dtype=np.int32)
    dq=deque([(sx,sy)]); visited[sy,sx]=1
    N4=[(0,-1),(0,1),(1,0),(-1,0)]
    N8=[(-1,-1),(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0)]
    neigh=N8 if eight else N4
    while dq:
        x,y=dq.popleft()
        if (x,y)==(ex,ey): break
        for dx,dy in neigh:
            nx,ny=x+dx,y+dy
            if 0<=nx<w and 0<=ny<h and visited[ny,nx]==0 and binary255[ny,nx]>0:
                visited[ny,nx]=1; parent[ny,nx]=[x,y]; dq.append((nx,ny))
    if parent[ey,ex,0]==-1 and (ex,ey)!=(sx,sy): return []
    path=[(ex,ey)]; x,y=ex,ey
    while (x,y)!=(sx,sy):
        px,py=parent[y,x] 
        if px==-1: break
        path.append((int(px),int(py))); x,y=int(px),int(py)
    path.append((sx,sy)); path.reverse(); return path

def _simplify_by_direction(points: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if len(points)<=2: return points
    out=[points[0]]
    dxp = points[1][0]-points[0][0]; dyp = points[1][1]-points[0][1]
    for i in range(2, len(points)):
        dx = points[i][0]-points[i-1][0]; dy = points[i][1]-points[i-1][1]
        if (dx,dy)!=(dxp,dyp): out.append(points[i-1])
        dxp,dyp = dx,dy
    out.append(points[-1]); return out

def _draw_polyline_clipped(base_bgr: np.ndarray, pts_xy: List[Tuple[int,int]],
                           allow_mask255: np.ndarray, color=(0,255,0),
                           thickness: int = DRAW_THICKNESS_PX):
    overlay = np.zeros_like(base_bgr)
    pts = np.array(pts_xy, dtype=np.int32).reshape((-1,1,2))
    cv2.polylines(overlay, [pts], isClosed=False, color=color,
                  thickness=thickness, lineType=cv2.LINE_AA)
    line_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
    allowed = cv2.bitwise_and(line_gray, allow_mask255)
    cv2.copyTo(overlay, allowed, base_bgr)

def _astar_centered(allow255: np.ndarray, dist_norm: np.ndarray, start, end,
                    *, len_w: float = 1.0, center_w: float = CENTER_BIAS) -> List[Tuple[int,int]]:
    # 4-connected A* with center penalty
    h,w = allow255.shape; sx,sy=start; ex,ey=end
    if not (0<=sx<w and 0<=sy<h and 0<=ex<w and 0<=ey<h): return []
    if allow255[sy,sx]==0 or allow255[ey,ex]==0: return []
    INF = 1e12
    g = np.full((h,w), INF, dtype=np.float32)
    parent = np.full((h,w,2), -1, dtype=np.int32)
    open_heap = []
    def hfun(x,y): return float(abs(x-ex)+abs(y-ey))
    import heapq
    g[sy,sx] = 0.0
    heapq.heappush(open_heap, (hfun(sx,sy), 0.0, sx, sy))
    closed = np.zeros((h,w), np.uint8)
    for_pop = [(0,-1),(0,1),(1,0),(-1,0)]
    while open_heap:
        f_curr, g_curr, x, y = heapq.heappop(open_heap)
        if closed[y,x]: continue
        closed[y,x] = 1
        if (x,y)==(ex,ey): break
        for dx,dy in for_pop:
            nx,ny = x+dx, y+dy
            if 0<=nx<w and 0<=ny<h and allow255[ny,nx]>0 and not closed[ny,nx]:
                step_cost = len_w + center_w*(1.0 - float(dist_norm[ny,nx]))
                tentative = g_curr + step_cost
                if tentative < g[ny,nx]:
                    g[ny,nx] = tentative
                    parent[ny,nx] = [x,y]
                    f = tentative + hfun(nx,ny)
                    heapq.heappush(open_heap, (f, tentative, nx, ny))
    if parent[ey,ex,0]==-1 and (ex,ey)!=(sx,sy): return []
    path=[(ex,ey)]
    x,y=ex,ey
    while (x,y)!=(sx,sy):
        px,py=parent[y,x]
        if px==-1: break
        path.append((int(px),int(py))); x,y=int(px),int(py)
    path.append((sx,sy)); path.reverse(); return path

# -------------------- Public solver --------------------
def solve_maze(orig_bgr: np.ndarray,
               start_color: str,
               thickness: int = 2,
               margin_px: Optional[int] = None,
               show: bool = False,
               output_path: str = "extracted_centered_path.png"
               ) -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    """
    Core end-to-end routine:
      - deskew/crop the maze ROI
      - detect red/green markers; choose start/end by `start_color`
      - build open-space corridor mask
      - auto-pick or apply safety margin; compute center field
      - try center-biased A* then skeleton-BFS then 4-connected BFS
      - draw a single GREEN polyline clipped to the corridor
      - project back to original coordinates; save image; optionally show

    Returns: (annotated_bgr_in_original_space, path_points_xy_in_original_space)
    Raises: RuntimeError with a descriptive message on failure.
    """
    if start_color not in ("red", "green"):
        raise RuntimeError("start_color must be 'red' or 'green'")

    # 1) Isolate the maze ROI
    maze_bgr, H_maze_to_orig = isolate_maze_roi(orig_bgr)

    # 2) Detect colored markers
    gpt, rpt, color_union = find_markers(maze_bgr)
    if gpt is None or rpt is None:
        raise RuntimeError("Could not detect both red and green markers in ROI.")
    start_roi = gpt if start_color == "green" else rpt
    end_roi   = rpt if start_color == "green" else gpt

    # 3) Corridor / inside mask
    walls = threshold_walls(maze_bgr, color_union)
    inside = inside_open_mask(walls, frame=3)     # white=open corridor

    # 4) Auto margin & center field
    draw_shrink = int(thickness//2 + max(0, VISUAL_GAP_PX))
    if margin_px is None:
        dist_tmp = cv2.distanceTransform(inside, cv2.DIST_L2, 3).astype(np.float32)
        skel_tmp = _skeletonize(inside)
        vals = dist_tmp[skel_tmp>0]
        r_med = float(np.median(vals)) if vals.size>0 else 3.0
        headroom = max(0.0, r_med - (draw_shrink + 1))
        margin_px = max(MIN_MARGIN_PX, min(int(round(AUTO_MARGIN_RATIO*r_med)), int(headroom)))
    else:
        margin_px = max(0, int(margin_px))

    dist = cv2.distanceTransform(inside, cv2.DIST_L2, 3).astype(np.float32)
    dmax = float(dist.max()) if float(dist.max())>0 else 1.0
    dist_norm = dist/(dmax+1e-6)

    # 5) Plan path with progressive fallbacks
    found_path = None
    safe_used = None
    for m in range(int(margin_px), -1, -1):
        safe = _erode_by_px(inside, m)
        if cv2.countNonZero(safe)==0: 
            continue
        s = nearest_nonzero(safe, start_roi)
        e = nearest_nonzero(safe, end_roi)
        path = _astar_centered(safe, dist_norm, s, e, center_w=CENTER_BIAS)
        if not path:
            sk = _skeletonize(safe)
            if cv2.countNonZero(sk)>0:
                s2 = nearest_nonzero(sk, s); e2 = nearest_nonzero(sk, e)
                path = _bfs(sk, s2, e2, eight=True)
        if not path:
            path = _bfs(safe, s, e, eight=False)
        if path:
            found_path = path; safe_used = safe; break
    if found_path is None:
        raise RuntimeError("Path Not Found. Check markers and maze continuity.")

    poly = _simplify_by_direction(found_path)

    # 6) Draw single GREEN polyline (clipped) in ROI, then project to original
    draw_allow = _erode_by_px(safe_used, draw_shrink)
    maze_vis = maze_bgr.copy()
    _draw_polyline_clipped(maze_vis, poly, draw_allow, color=(0,255,0), thickness=thickness)
    cv2.circle(maze_vis, tuple(map(int, poly[0])), 6, (0,255,0), -1)  # start
    cv2.circle(maze_vis, tuple(map(int, poly[-1])),6, (0,0,255), -1)  # goal

    # Project path + allow mask back into ORIGINAL coordinate system
    pts = np.array(poly, dtype=np.float32).reshape(-1,1,2)
    pts_h = cv2.perspectiveTransform(pts, H_maze_to_orig)
    poly_orig = [(int(round(p[0][0])), int(round(p[0][1]))) for p in pts_h]
    allow_full = cv2.warpPerspective(draw_allow, H_maze_to_orig,
                                     (orig_bgr.shape[1], orig_bgr.shape[0]))
    allow_full = (allow_full>0).astype(np.uint8)*255

    out_vis = orig_bgr.copy()
    _draw_polyline_clipped(out_vis, poly_orig, allow_full, color=(0,255,0), thickness=thickness)
    cv2.circle(out_vis, poly_orig[0], 6, (0,255,0), -1)
    cv2.circle(out_vis, poly_orig[-1],6, (0,0,255), -1)

    # Save + optional display
    if output_path:
        cv2.imwrite(output_path, out_vis)
    if show:
        cv2.imshow("Final Centered Path", out_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return out_vis, poly_orig

# -------------------- CLI (dual mode: image OR camera) --------------------
def _acquire_image(camera_index: int, image_path: str) -> np.ndarray:
    # camera takes precedence when index >= 0, otherwise falls back to image path
    if camera_index is not None and camera_index >= 0:
        cap = cv2.VideoCapture(camera_index)
        time.sleep(0.5)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Camera read failed. Try a different --camera-index or use --image.")
        return frame
    if not image_path:
        raise RuntimeError("Provide --image PATH or set --camera-index >= 0")
    if not os.path.exists(image_path):
        raise RuntimeError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {image_path}")
    return img

def main():
    ap = argparse.ArgumentParser(description="Extracted centered path solver (green polyline) with dual input mode")
    ap.add_argument("--start-color", required=True, choices=["red","green"],
                    help="Pick which colored marker is the START; the other color becomes the GOAL.")
    ap.add_argument("--camera-index", type=int, default=-1,
                    help="If >=0, capture a single frame from this camera; else use --image.")
    ap.add_argument("--image", type=str, default="",
                    help="Path to a saved maze image (used when --camera-index < 0).")
    ap.add_argument("--thickness", type=int, default=2, help="Green path thickness in pixels.")
    ap.add_argument("--margin", type=int, default=None,
                    help="Safety margin (px) from walls; if omitted, computed automatically.")
    ap.add_argument("--show", action="store_true", help="Show a window with the final result.")
    ap.add_argument("--output", type=str, default="extracted_centered_path.png",
                    help="Where to save the annotated image.")
    args = ap.parse_args()

    try:
        src = _acquire_image(args.camera_index, args.image)
        out, poly = solve_maze(src, start_color=args.start_color,
                               thickness=args.thickness,
                               margin_px=args.margin,
                               show=args.show,
                               output_path=args.output)
        print(f"[OK] Path points: {len(poly)}; saved: {args.output}")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
