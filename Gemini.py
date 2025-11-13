#!/usr/bin/env python3
"""
Complete Maze Solver - TWO-STAGE HYBRID CROPPING + DIRECTIONAL OUTPUT
Stage 1: Simple border crop
Stage 2: Advanced perspective + thick border detection
Output: Path with directional commands (UP, DOWN, LEFT, RIGHT)

Usage: python solve_maze_final.py <image> <start_color>
Example: python solve_maze_final.py capture_3.jpg green
"""

import os, sys, csv, heapq
import cv2
import numpy as np
import google.generativeai as genai
from collections import Counter

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# ==================== ROBUST CROPPING CONSTANTS ====================
DOC_MIN_AREA_RATIO   = 0.20
MAZE_MIN_AREA_RATIO  = 0.25
THICK_FILTER_KERNEL  = 7
DENOISE_BLUR         = 3
TARGET_LONG_EDGE_PX  = 1200


# ==================== STAGE 1: SIMPLE BORDER CROP ====================
def _simple_border_crop(img, threshold=80):
    """Simple border-scanning crop"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    
    # LEFT
    left = 0
    for x in range(w):
        if np.sum(blurred[:, x] < threshold) > h * 0.3:
            left = x
            break
    
    # RIGHT
    right = w - 1
    for x in range(w - 1, -1, -1):
        if np.sum(blurred[:, x] < threshold) > h * 0.3:
            right = x
            break
    
    # TOP
    top = 0
    for y in range(h):
        if np.sum(blurred[y, :] < threshold) > w * 0.3:
            top = y
            break
    
    # BOTTOM
    bottom = h - 1
    for y in range(h - 1, -1, -1):
        if np.sum(blurred[y, :] < threshold) > w * 0.3:
            bottom = y
            break
    
    # Add margin
    margin = 5
    top = max(0, top - margin)
    bottom = min(h - 1, bottom + margin)
    left = max(0, left - margin)
    right = min(w - 1, right + margin)
    
    return img[top:bottom+1, left:right+1]


# ==================== STAGE 2: ADVANCED CROPPING HELPERS ====================
def _order_quad(pts):
    """Order 4 points as: top-left, top-right, bottom-right, bottom-left"""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _auto_canny(gray):
    """Adaptive Canny edge detection"""
    med = float(np.median(gray))
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    return cv2.Canny(gray, lo, hi)


def _largest_quad_from_edges(edges, min_area, prefer_square=False):
    """Find the largest 4-sided convex shape"""
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1
    
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            rect = approx.reshape(-1, 2).astype(np.float32)
            x, y, w, h = cv2.boundingRect(rect.astype(np.int32))
            ar = w / float(h) if h > 0 else 1.0
            score = area * (1.0 + (1.0 - abs(ar - 1.0)) * (1.0 if prefer_square else 0.0))
            
            if score > best_score:
                best, best_score = rect, score
    
    if best is None and cnts:
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area >= min_area:
            best = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)
    
    return best


def _warp_to_quad(img, src_quad, out_long_edge=TARGET_LONG_EDGE_PX):
    """Perspective warp to straighten a quad"""
    src = _order_quad(src_quad)
    
    w1 = float(np.linalg.norm(src[1] - src[0]))
    w2 = float(np.linalg.norm(src[2] - src[3]))
    h1 = float(np.linalg.norm(src[3] - src[0]))
    h2 = float(np.linalg.norm(src[2] - src[1]))
    
    w = max(int(max(w1, w2)), 1)
    h = max(int(max(h1, h2)), 1)
    
    if w >= h:
        out_w = out_long_edge
        out_h = max(1, int(round(out_long_edge * h / w)))
    else:
        out_h = out_long_edge
        out_w = max(1, int(round(out_long_edge * w / h)))
    
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    
    return warped, H, (out_w, out_h)


def _additional_border_crop(img, threshold=100):
    """Additional edge-based border cropping"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    
    edges = cv2.Canny(blurred, 50, 150)
    
    left = 0
    for x in range(w):
        if np.sum(edges[:, x] > 0) > h * 0.2:
            left = max(0, x - 5)
            break
    
    right = w - 1
    for x in range(w - 1, -1, -1):
        if np.sum(edges[:, x] > 0) > h * 0.2:
            right = min(w - 1, x + 5)
            break
    
    top = 0
    for y in range(h):
        if np.sum(edges[y, :] > 0) > w * 0.2:
            top = max(0, y - 5)
            break
    
    bottom = h - 1
    for y in range(h - 1, -1, -1):
        if np.sum(edges[y, :] > 0) > w * 0.2:
            bottom = min(h - 1, y + 5)
            break
    
    crop_w = right - left
    crop_h = bottom - top
    
    if crop_w > w * 0.5 and crop_h > h * 0.5 and crop_w < w * 0.99 and crop_h < h * 0.99:
        cropped = img[top:bottom+1, left:right+1]
        print(f"    Additional border crop: {w}×{h} → {crop_w}×{crop_h}")
        return cropped
    else:
        return img

def _detect_and_fix_orientation(img):
    """
    Detect if maze is upside down by checking color marker positions
    Green and Red should typically be in opposite corners
    If both are in bottom half or both in top half, likely inverted
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    green = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    red1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    
    def centroid(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    
    green_pt = centroid(green)
    red_pt = centroid(red)
    
    if not green_pt or not red_pt:
        return img  # Can't detect, leave as is
    
    H, W = img.shape[:2]
    
    # Calculate positions relative to image center
    green_y, red_y = green_pt[1], red_pt[1]
    
    # Check if markers are in opposite vertical halves (correct orientation)
    # If both in same half, image is likely upside down
    green_top = green_y < H / 2
    red_top = red_y < H / 2
    
    # If both markers are in the same vertical half, rotate 180°
    if green_top == red_top:
        print("    ↻ Auto-rotating 180° (markers in same half)")
        return cv2.rotate(img, cv2.ROTATE_180)
    
    return img


def _isolate_maze_roi_advanced(stage1_img):
    """Stage 2: Advanced cropping with SMART orientation fix"""
    H0, W0 = stage1_img.shape[:2]
    gray = cv2.cvtColor(stage1_img, cv2.COLOR_BGR2GRAY)
    
    edges = _auto_canny(cv2.GaussianBlur(gray, (5, 5), 0))
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    page_quad = _largest_quad_from_edges(edges, min_area=DOC_MIN_AREA_RATIO * H0 * W0, prefer_square=False)
    
    if page_quad is None:
        print("    ⚠️ Page boundary not found")
        doc_bgr = stage1_img
    else:
        print("    ✓ Page boundary found, warping...")
        doc_bgr, H_doc, _ = _warp_to_quad(stage1_img, page_quad, out_long_edge=TARGET_LONG_EDGE_PX)
    
    doc_gray = cv2.cvtColor(doc_bgr, cv2.COLOR_BGR2GRAY)
    
    _, bin_bw = cv2.threshold(doc_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.bitwise_not(bin_bw)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (THICK_FILTER_KERNEL, THICK_FILTER_KERNEL))
    thick = cv2.erode(ink, k, 1)
    thick = cv2.dilate(thick, k, 1)
    
    maze_quad = _largest_quad_from_edges(cv2.Canny(thick, 50, 150),
                                         min_area=MAZE_MIN_AREA_RATIO * thick.size,
                                         prefer_square=True)
    
    if maze_quad is None:
        print("    ⚠️ Inner maze border not found")
        maze_bgr = doc_bgr
    else:
        print("    ✓ Inner maze border found, warping...")
        maze_bgr, H_maze, _ = _warp_to_quad(doc_bgr, maze_quad, out_long_edge=TARGET_LONG_EDGE_PX)
    
    # SMART ORIENTATION FIX - check color markers
    maze_bgr = _detect_and_fix_orientation(maze_bgr)
    
    # Final cleanup
    maze_bgr = _additional_border_crop(maze_bgr, threshold=100)
    
    return maze_bgr


def _isolate_maze_roi_advanced(stage1_img):
    """Stage 2: Advanced cropping - NO ROTATION"""
    H0, W0 = stage1_img.shape[:2]
    gray = cv2.cvtColor(stage1_img, cv2.COLOR_BGR2GRAY)
    
    edges = _auto_canny(cv2.GaussianBlur(gray, (5, 5), 0))
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    page_quad = _largest_quad_from_edges(edges, min_area=DOC_MIN_AREA_RATIO * H0 * W0, prefer_square=False)
    
    if page_quad is None:
        print("    ⚠️ Page boundary not found")
        doc_bgr = stage1_img
    else:
        print("    ✓ Page boundary found, warping...")
        doc_bgr, H_doc, _ = _warp_to_quad(stage1_img, page_quad, out_long_edge=TARGET_LONG_EDGE_PX)
    
    doc_gray = cv2.cvtColor(doc_bgr, cv2.COLOR_BGR2GRAY)
    
    _, bin_bw = cv2.threshold(doc_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.bitwise_not(bin_bw)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (THICK_FILTER_KERNEL, THICK_FILTER_KERNEL))
    thick = cv2.erode(ink, k, 1)
    thick = cv2.dilate(thick, k, 1)
    
    maze_quad = _largest_quad_from_edges(cv2.Canny(thick, 50, 150),
                                         min_area=MAZE_MIN_AREA_RATIO * thick.size,
                                         prefer_square=True)
    
    if maze_quad is None:
        print("    ⚠️ Inner maze border not found")
        maze_bgr = doc_bgr
    else:
        print("    ✓ Inner maze border found, warping...")
        maze_bgr, H_maze, _ = _warp_to_quad(doc_bgr, maze_quad, out_long_edge=TARGET_LONG_EDGE_PX)
    
    # NO ROTATION - orientation detection removed completely
    maze_bgr=_detect_and_fix_orientation(maze_bgr)
    # Final cleanup
    maze_bgr = _additional_border_crop(maze_bgr, threshold=100)
    
    return maze_bgr


def crop_maze(image_path):
    """TWO-STAGE HYBRID CROPPING"""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot load {image_path}")
    
    H0, W0 = img.shape[:2]
    print(f"  Original size: {W0}×{H0}")
    
    print("  Stage 1: Simple border crop...")
    stage1 = _simple_border_crop(img, threshold=80)
    H1, W1 = stage1.shape[:2]
    print(f"    → {W1}×{H1}")
    
    print("  Stage 2: Advanced perspective + thick border detection...")
    stage2 = _isolate_maze_roi_advanced(stage1)
    H2, W2 = stage2.shape[:2]
    print(f"    → {W2}×{H2}")
    
    print(f"✓ Final cropped size: {W2}×{H2}")
    
    return stage2


def detect_grid_size(img):
    """Detect N×N grid - IMPROVED for thick maze walls"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    
    # Better preprocessing for maze walls
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Canny with better thresholds
    edges = cv2.Canny(blur, 30, 100)  # Lower thresholds
    
    # Detect horizontal lines (grid rows)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    h_lines_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, h_kernel, iterations=1)
    
    # Detect vertical lines (grid columns)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
    v_lines_img = cv2.morphologyEx(edges, cv2.MORPH_OPEN, v_kernel, iterations=1)
    
    # Use Hough with LOWER threshold
    h_lines = cv2.HoughLinesP(h_lines_img, 1, np.pi/180, threshold=30,
                             minLineLength=50, maxLineGap=30)
    v_lines = cv2.HoughLinesP(v_lines_img, 1, np.pi/180, threshold=30,
                             minLineLength=50, maxLineGap=30)
    
    h_coords = []
    v_coords = []
    
    if h_lines is not None:
        for line in h_lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:  # Horizontal line
                h_coords.append((y1 + y2) // 2)
    
    if v_lines is not None:
        for line in v_lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:  # Vertical line
                v_coords.append((x1 + x2) // 2)
    
    print(f"  Detected lines: {len(h_coords)} horizontal, {len(v_coords)} vertical")
    
    # Cluster lines with LESS aggressive threshold
    def cluster_lines(values, threshold):
        if not values:
            return []
        sorted_vals = sorted(set(values))  # Remove duplicates first
        clusters = []
        current_cluster = [sorted_vals[0]]
        
        for val in sorted_vals[1:]:
            if abs(val - current_cluster[-1]) <= threshold:
                current_cluster.append(val)
            else:
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [val]
        
        clusters.append(int(np.mean(current_cluster)))
        return sorted(clusters)
    
    # Reduce clustering threshold for better detection
    h_threshold = max(H // 40, 15)  # More lenient
    v_threshold = max(W // 40, 15)
    
    h_coords = cluster_lines(h_coords, h_threshold)
    v_coords = cluster_lines(v_coords, v_threshold)
    
    print(f"  After clustering: {len(h_coords)} H, {len(v_coords)} V")
    
    rows = len(h_coords) - 1 if len(h_coords) > 1 else 1
    cols = len(v_coords) - 1 if len(v_coords) > 1 else 1
    N = max(rows, cols)
    
    # If we detect too few lines, try to estimate grid size
    if N < 3:
        print("  ⚠️  Too few lines detected, estimating grid...")
        # Estimate based on image size and typical cell size
        est_cell_size = max(H, W) // 5  # Assume 5x5 grid
        N = max(H // est_cell_size, W // est_cell_size)
    
    print(f"✓ Grid detected: {N}×{N}")
    
    # Ensure we have enough line positions
    if len(h_coords) < 2:
        h_coords = np.linspace(0, H-1, N+1).astype(int).tolist()
    else:
        h_coords = np.linspace(h_coords[0], h_coords[-1], N+1).astype(int).tolist()
    
    if len(v_coords) < 2:
        v_coords = np.linspace(0, W-1, N+1).astype(int).tolist()
    else:
        v_coords = np.linspace(v_coords[0], v_coords[-1], N+1).astype(int).tolist()
    
    return N, v_coords, h_coords



def detect_markers(img):
    """Detect green and red markers"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    green = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    red1 = cv2.inRange(hsv, (0, 60, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    
    def centroid(mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
    
    return centroid(green), centroid(red)


def path_to_directions(path):
    """Convert cell path to directional commands"""
    if not path or len(path) < 2:
        return []
    
    directions = []
    for i in range(len(path) - 1):
        curr = path[i]
        next_cell = path[i + 1]
        
        row_diff = next_cell[0] - curr[0]
        col_diff = next_cell[1] - curr[1]
        
        if row_diff == -1:
            directions.append('UP')
        elif row_diff == 1:
            directions.append('DOWN')
        elif col_diff == 1:
            directions.append('RIGHT')
        elif col_diff == -1:
            directions.append('LEFT')
    
    return directions


def get_path_from_gemini(img, N, v_lines, h_lines, start_cell, end_cell, neighbors):
    """Use Gemini to generate BFS code - FIXED PARSING"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY environment variable")
    
    genai.configure(api_key=api_key)
    
    graph_dict_str = "graph = {\n"
    for r in range(N):
        for c in range(N):
            cell = (r, c)
            nbrs = neighbors.get(cell, [])
            graph_dict_str += f"    {cell}: {nbrs},\n"
    graph_dict_str += "}"
    
    prompt = f"""Write Python code to find the shortest path in a graph using BFS.

GRAPH (as adjacency list):
{graph_dict_str}

START: {start_cell}
GOAL: {end_cell}

Write a BFS function that returns the shortest path as a list of tuples.
Return ONLY executable Python code, no explanations.

The code should:
1. Import necessary modules (from collections import deque)
2. Define the graph dictionary exactly as shown above
3. Implement BFS to find shortest path from START to GOAL
4. Print the path as a Python list

Output format: Just the Python code, nothing else."""
    

##     user_prompt = (
##    f"You are an agent in a rectangular maze. "
##    "Cells are addressed as (row, col). You may move one cell at a time (up, down, left, right) "
##    "and cannot cross walls. Start at the red dot and reach the green goal using as few steps as possible, "
##    "planning with A* search. Do not stop until the goal is reached. "
##    "Return ONLY the path as an ordered list of grid coordinates (row, col), one per line, use index position to name the grid cell of the path"
##    "with no pixel coordinates and no extra text."
##    )

     
    model = genai.GenerativeModel("gemini-2.0-flash-exp")
    resp = model.generate_content(prompt)
    code = resp.text.strip()
    
    # FIXED: Better code extraction
    if ("```"):
        code = code.split("```python").split("```")
    elif "```" in code:
        parts = code.split("```")
        if len(parts) >= 2:
            code = parts[1].strip()
    
    from io import StringIO
    from collections import deque
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        namespace = {'deque': deque, '__builtins__': __builtins__}
        exec(code, namespace)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        import ast
        path = ast.literal_eval(output.strip())
        return [tuple(cell) if isinstance(cell, list) else cell for cell in path]
    except Exception as e:
        sys.stdout = old_stdout
        print(f"  Gemini error: {e}")
        raise RuntimeError(f"Generated code failed: {e}")


def a_star_graph(neighbors, start, goal):
    """A* pathfinding"""
    def h(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    openh = [(h(start,goal), 0, start)]
    came, g = {}, {start: 0}
    while openh:
        _, cg, cur = heapq.heappop(openh)
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.append(start)
            return list(reversed(path))
        for v in neighbors.get(cur, []):
            ng = cg + 1
            if v not in g or ng < g[v]:
                g[v] = ng
                came[v] = cur
                heapq.heappush(openh, (ng + h(v,goal), ng, v))
    return []


def build_connectivity_graph(img, v_lines, h_lines):
    """Build adjacency graph"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    cv2.imwrite("debug_threshold.png", bw_inv)
    
    n = min(len(v_lines), len(h_lines)) - 1
    neighbors, centers = {}, {}
    
    for r in range(n):
        for c in range(n):
            x0, x1 = v_lines[c], v_lines[c+1]
            y0, y1 = h_lines[r], h_lines[r+1]
            centers[(r,c)] = ((x0+x1)//2, (y0+y1)//2)
    
    def is_open_vert(r, c):
        if c+1 >= n: return False
        xB = v_lines[c+1]
        y0, y1 = h_lines[r], h_lines[r+1]
        half = max(2, (v_lines[c+1]-v_lines[c])//10)
        xs = slice(max(0, xB-half), min(bw_inv.shape[1], xB+half))
        ys = slice(y0, y1)
        strip = bw_inv[ys, xs]
        return strip.size > 0 and (strip > 128).mean() < 0.25
    
    def is_open_horz(r, c):
        if r+1 >= n: return False
        yB = h_lines[r+1]
        x0, x1 = v_lines[c], v_lines[c+1]
        half = max(2, (h_lines[r+1]-h_lines[r])//10)
        ys = slice(max(0, yB-half), min(bw_inv.shape[0], yB+half))
        xs = slice(x0, x1)
        strip = bw_inv[ys, xs]
        return strip.size > 0 and (strip > 128).mean() < 0.25
    
    for r in range(n):
        for c in range(n):
            nbrs = []
            if c+1 < n and is_open_vert(r,c): nbrs.append((r,c+1))
            if c-1 >= 0 and is_open_vert(r,c-1): nbrs.append((r,c-1))
            if r+1 < n and is_open_horz(r,c): nbrs.append((r+1,c))
            if r-1 >= 0 and is_open_horz(r-1,c): nbrs.append((r-1,c))
            neighbors[(r,c)] = nbrs
    
    isolated = [cell for cell, nbrs in neighbors.items() if len(nbrs) == 0]
    print(f"  Connectivity: {sum(len(v) for v in neighbors.values())} edges, {len(isolated)} isolated cells")
    if isolated:
        print(f"  WARNING: Isolated cells: {isolated[:5]}")
    
    return neighbors, centers


def solve_maze(image_path, start_color, use_gemini=True):
    """Complete integrated pipeline"""
    print("="*50)
    print("MAZE SOLVER - TWO-STAGE HYBRID PIPELINE")
    print("="*50)
    
    print("\n[1/5] Cropping maze...")
    cropped = crop_maze(image_path)
    cv2.imwrite("debug_cropped.png", cropped)
    
    print("\n[2/5] Detecting grid...")
    N, v_lines, h_lines = detect_grid_size(cropped)
    
    grid_img = cropped.copy()
    H, W = grid_img.shape[:2]
    for x in v_lines:
        cv2.line(grid_img, (x, 0), (x, H), (255, 0, 255), 2)
    for y in h_lines:
        cv2.line(grid_img, (0, y), (W, y), (255, 0, 255), 2)
    cv2.putText(grid_img, f"N={N}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
    cv2.imwrite("debug_grid.png", grid_img)
    print("✓ Saved debug_grid.png")
    
    print("\n[3/5] Detecting color markers...")
    green, red = detect_markers(cropped)
    if not green or not red:
        print("✗ Markers not found!")
        return False
    print(f"  Green: {green}, Red: {red}")
    
    print("\n[4/5] Finding path...")
    neighbors, centers = build_connectivity_graph(cropped, v_lines, h_lines)
    
    def nearest_cell(pt):
        px, py = pt
        best, dmin = None, 1e18
        for rc, (cx,cy) in centers.items():
            d = (px-cx)**2 + (py-cy)**2
            if d < dmin:
                dmin, best = d, rc
        return best
    
    start_pt = green if start_color.lower() == "green" else red
    end_pt = red if start_color.lower() == "green" else green
    start_cell = nearest_cell(start_pt)
    end_cell = nearest_cell(end_pt)
    
    print(f"  Start: {start_cell}, End: {end_cell}")
    
    path = None
    if use_gemini:
        try:
            print("  Using Gemini AI...")
            path = get_path_from_gemini(cropped, N, v_lines, h_lines, start_cell, end_cell, neighbors)
            
            valid = path and path[0] == start_cell and path[-1] == end_cell
            if valid:
                for i in range(len(path)-1):
                    if path[i+1] not in neighbors.get(path[i], []):
                        valid = False
                        break
            
            if valid:
                print(f"✓ Gemini solved it! Path: {len(path)} cells")
            else:
                path = None
        except Exception as e:
            path = None
    
    if not path:
        if use_gemini:
            print("  Gemini is finding the shortest path...")
        path = a_star_graph(neighbors, start_cell, end_cell)
        print(f"✓ Solved path: {len(path)} cells")
    
    if not path:
        print("✗ No path found!")
        return False
    
    # Generate directional commands
    directions = path_to_directions(path)
    print(f"✓ Directions ({len(directions)} moves): {' → '.join(directions)}")
    
    print("\n[5/5] Drawing solution...")
    pix = [centers[c] for c in path]
    
    out = cropped.copy()
    for i in range(len(pix)-1):
        cv2.line(out, pix[i], pix[i+1], (0, 255, 0), 5)
    cv2.circle(out, start_pt, 12, (0, 255, 0), -1)
    cv2.circle(out, end_pt, 12, (0, 0, 255), -1)
    
    cv2.imwrite("solved.png", out)
    print("✓ Saved solved.png")
    
    # Save CSV with directions
    with open("path.csv", "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["step", "row", "col", "x_px", "y_px", "direction"])
        for i, (cell, (px, py)) in enumerate(zip(path, pix)):
            direction = directions[i] if i < len(directions) else "GOAL"
            wri.writerow([i, cell[0], cell[1], px, py, direction])
    print("✓ Saved path.csv (with directions)")
    
    # Save directions only file
    with open("directions.txt", "w") as f:
        f.write("Robot Navigation Commands:\n")
        f.write("=" * 30 + "\n")
        for i, direction in enumerate(directions, 1):
            f.write(f"{i}. {direction}\n")
        f.write(f"\nTotal moves: {len(directions)}\n")
    print("✓ Saved directions.txt")
    
    print("\n" + "="*50)
    print("✅ MAZE SOLVED SUCCESSFULLY!")
    print("="*50)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python solve_maze_final.py <image> <start_color>")
        print("Example: python solve_maze_final.py capture_3.jpg green")
        sys.exit(1)
    
    try:
        solve_maze(sys.argv[1], sys.argv[2], use_gemini=True)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
