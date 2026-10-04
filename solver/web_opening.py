"""Same steel I-beam section, same web opening at the same distance from a support, four spans:
how much earlier does global first yield come in each beam than in its own unholed twin?

2-D plane stress through the beam's depth with the THICKNESS of each triangle set by its height:
flange width inside the flanges, web thickness in the web (the standard 2-D model of an I-section
loaded in its plane). Steel is isotropic and first yield follows von Mises, so the load at which the
peak von Mises stress first reaches yield is a valid first-yield measure; linear elasticity gives it
exactly as (solid peak / holed peak) of the solid beam's first-yield load. No plasticity, no buckling,
no collapse load.

Supports carry no point loads: each end's reaction is applied as the beam-theory shear traction over
the end face (V Q(y) / I per unit height), so no singular stress sits anywhere in the read window.

"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import Delaunay

# ---- frozen section (IPE 300) and span --------------------------------------------------------
H_SEC, B_FL, TF, TW = 0.300, 0.150, 0.0107, 0.0071     # m
SPAN = 4.0                  # default only; the production run passes each span explicitly
NU = 0.3
E = 1.0                     # ratios only
Q_LOAD = 1.0                # N per m along the span, on the top flange (ratios only)


def thickness(y):
    return np.where((y < TF) | (y > H_SEC - TF), B_FL, TW)


def section_I():
    """Second moment of area of the idealised I (no root radii), for the end tractions."""
    hw = H_SEC - 2 * TF
    return B_FL * H_SEC ** 3 / 12 - (B_FL - TW) * hw ** 3 / 12


def first_moment(y):
    """Q(y): first moment of the area ABOVE height y about the neutral axis (mid-depth)."""
    c = H_SEC / 2
    y = np.asarray(y, float)
    def area_moment(a, b, t):                       # strip from a to b (heights), width t
        lo, hi = np.maximum(a, y), b
        return np.where(hi > lo, t * ((hi - c) ** 2 - (lo - c) ** 2) / 2, 0.0)
    return (area_moment(0.0, TF, B_FL) + area_moment(TF, H_SEC - TF, TW)
            + area_moment(H_SEC - TF, H_SEC, B_FL))


def make_mesh(holes, span=None, h_far=0.006, h_near=0.0012, n_ring=96):
    """Background grid with horizontal lines on both flange-web interfaces (so no triangle straddles
    a thickness jump), graded rings around each hole."""
    span = SPAN if span is None else span
    xs = np.linspace(0.0, span, int(round(span / h_far)) + 1)
    ys = np.unique(np.r_[np.linspace(0.0, TF, 4), np.linspace(TF, H_SEC - TF,
                                                             int(round((H_SEC - 2 * TF) / h_far)) + 1),
                         np.linspace(H_SEC - TF, H_SEC, 4)])
    pts = []
    for x in xs:
        for y in ys:
            if all(math.hypot(x - cx, y - cy) >= r + 3 * h_far for cx, cy, r in holes):
                pts.append((x, y))
    for cx, cy, r in holes:
        rad, dr = r, h_near
        while rad < r + 3 * h_far + 1e-12:
            n = max(int(round(2 * math.pi * rad / dr)), n_ring)
            for k in range(n):
                a = 2 * math.pi * k / n
                x, y = cx + rad * math.cos(a), cy + rad * math.sin(a)
                if TF + 1e-9 < y < H_SEC - TF - 1e-9:          # holes stay in the web
                    pts.append((x, y))
            rad += dr
            dr = min(dr * 1.25, h_far)
        for x in np.arange(cx - r - 3 * h_far, cx + r + 3 * h_far, h_near * 2):
            for y in (TF, H_SEC - TF):
                pts.append((x, y))
    P = np.unique(np.round(np.array(pts), 10), axis=0)
    T = Delaunay(P).simplices
    cent = P[T].mean(axis=1)
    keep = np.ones(len(T), bool)
    for cx, cy, r in holes:
        keep &= np.hypot(cent[:, 0] - cx, cent[:, 1] - cy) > r
    a = 0.5 * np.abs((P[T[:, 1], 0] - P[T[:, 0], 0]) * (P[T[:, 2], 1] - P[T[:, 0], 1])
                     - (P[T[:, 2], 0] - P[T[:, 0], 0]) * (P[T[:, 1], 1] - P[T[:, 0], 1]))
    keep &= a > 1e-14
    return P, T[keep]


def solve(P, T, q=Q_LOAD, support="traction", pad=0.10, span=None):
    span = SPAN if span is None else span
    n = len(P)
    D = E / (1 - NU ** 2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    rows, cols, vals, Bs = [], [], [], []
    cent = P[T].mean(axis=1)
    tk = thickness(cent[:, 1])
    for t, th in zip(T, tk):
        (x1, y1), (x2, y2), (x3, y3) = P[t]
        A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / A2
        c = np.array([x3 - x2, x1 - x3, x2 - x1]) / A2
        B = np.zeros((3, 6))
        B[0, 0::2], B[1, 1::2], B[2, 0::2], B[2, 1::2] = b, c, c, b
        dof = np.array([2 * t[0], 2 * t[0] + 1, 2 * t[1], 2 * t[1] + 1, 2 * t[2], 2 * t[2] + 1])
        rows.append(np.repeat(dof, 6)); cols.append(np.tile(dof, 6))
        vals.append((th * 0.5 * abs(A2) * B.T @ D @ B).ravel()); Bs.append((B, dof))
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * n, 2 * n)).tocsr()
    f = np.zeros(2 * n)
    top = np.where(np.abs(P[:, 1] - H_SEC) < 1e-12)[0]
    top = top[np.argsort(P[top, 0])]
    for a, b in zip(top[:-1], top[1:]):
        L = P[b, 0] - P[a, 0]
        f[2 * a + 1] -= q * L / 2
        f[2 * b + 1] -= q * L / 2
    I = section_I()
    V = q * span / 2
    ends = {}
    for side, x0 in (("left", 0.0), ("right", span)):
        e = np.where(np.abs(P[:, 0] - x0) < 1e-12)[0]
        ends[side] = e[np.argsort(P[e, 1])]
    if support == "traction":
        # end reactions as beam-theory shear tractions V Q(y) / I per unit height
        for side in ("left", "right"):
            e = ends[side]
            for a, b in zip(e[:-1], e[1:]):
                ym = np.linspace(P[a, 1], P[b, 1], 9)
                force = V * np.trapezoid(first_moment(ym), ym) / I
                f[2 * a + 1] += force / 2
                f[2 * b + 1] += force / 2
        lmid = ends["left"][np.argmin(np.abs(P[ends["left"], 1] - H_SEC / 2))]
        rmid = ends["right"][np.argmin(np.abs(P[ends["right"], 1] - H_SEC / 2))]
        fixed = [2 * lmid, 2 * lmid + 1, 2 * rmid + 1]
    elif support == "pad":
        # second plausible introduction: a uniform bearing pressure on the bottom flange over a
        # `pad`-long plate at each end (no stiffener), equal to the reaction
        bot = np.where(np.abs(P[:, 1]) < 1e-12)[0]
        bot = bot[np.argsort(P[bot, 0])]
        for lo, hi in ((0.0, pad), (span - pad, span)):
            seg = bot[(P[bot, 0] >= lo - 1e-12) & (P[bot, 0] <= hi + 1e-12)]
            w = np.zeros(len(seg))
            for i, (a, b) in enumerate(zip(seg[:-1], seg[1:])):
                L = P[b, 0] - P[a, 0]
                w[i] += L / 2
                w[i + 1] += L / 2
            # the pad nodes need not reach exactly `pad`; scale so the pad carries exactly V
            for node, wi in zip(seg, w / w.sum()):
                f[2 * node + 1] += V * wi
        lmid = ends["left"][np.argmin(np.abs(P[ends["left"], 1] - H_SEC / 2))]
        rmid = ends["right"][np.argmin(np.abs(P[ends["right"], 1] - H_SEC / 2))]
        fixed = [2 * lmid, 2 * lmid + 1, 2 * rmid + 1]
    else:
        raise ValueError(support)
    free = np.setdiff1d(np.arange(2 * n), fixed)
    u = np.zeros(2 * n)
    u[free] = spsolve(K[free][:, free].tocsc(), f[free])
    R = K @ u - f
    st = np.array([D @ (B @ u[dof]) for B, dof in Bs])
    sx, sy, txy = st[:, 0], st[:, 1], st[:, 2]
    vm = np.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3 * txy ** 2)
    return {"u": u, "vm": vm, "sx": sx, "txy": txy, "cent": cent, "thick": tk,
            "restraint_reactions": [float(R[d]) for d in fixed], "load_sum": float(f[1::2].sum())}


# ---- mesh geometry checks (registered) ----------------------------------------------------------
def boundary_edges(T):
    from collections import Counter
    c = Counter()
    for t in T:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            c[(min(a, b), max(a, b))] += 1
    return [e for e, k in c.items() if k == 1]


def mesh_checks(P, T, holes, span):
    """(1) every boundary edge lies on the outer rectangle or on a hole circle;
    (2) every hole-boundary node is on its circle to 1e-9; (3) no triangle straddles a flange-web
    interface (all its vertices on one side, or on the line)."""
    worst_circle, off_boundary = 0.0, 0
    for a, b in boundary_edges(T):
        pa, pb = P[a], P[b]
        on_rect = all(abs(v[0]) < 1e-9 or abs(v[0] - span) < 1e-9 or abs(v[1]) < 1e-9 or
                      abs(v[1] - H_SEC) < 1e-9 for v in (pa, pb))
        if on_rect:
            continue
        hit = False
        for cx, cy, r in holes:
            da = abs(math.hypot(pa[0] - cx, pa[1] - cy) - r)
            db = abs(math.hypot(pb[0] - cx, pb[1] - cy) - r)
            if max(da, db) < 1e-6:
                worst_circle = max(worst_circle, da, db)
                hit = True
        if not hit:
            off_boundary += 1
    straddle = 0
    for t in T:
        ys = P[t, 1]
        for yi in (TF, H_SEC - TF):
            if (ys < yi - 1e-12).any() and (ys > yi + 1e-12).any():
                straddle += 1
    return {"worst_hole_node_off_circle": worst_circle, "boundary_edges_off_geometry": off_boundary,
            "triangles_straddling_interface": straddle}


# ---- Kirsch / Heywood check (same CST) ----------------------------------------------------------
def kirsch(h_near=0.0003, W=1.0, Lx=2.0, d=0.15):
    """A d-diameter hole in a W-wide strip under unit end tension, uniform thickness."""
    xs = np.linspace(0.0, Lx, int(round(Lx / 0.02)) + 1)
    ys = np.linspace(0.0, W, int(round(W / 0.02)) + 1)
    pts = [(x, y) for x in xs for y in ys if math.hypot(x - Lx / 2, y - W / 2) >= d / 2 + 0.06]
    rad, dr = d / 2, h_near
    while rad < d / 2 + 0.06 + 1e-12:
        nn = max(int(round(2 * math.pi * rad / dr)), 96)
        pts += [(Lx / 2 + rad * math.cos(2 * math.pi * k / nn), W / 2 + rad * math.sin(2 * math.pi * k / nn))
                for k in range(nn)]
        rad += dr
        dr = min(dr * 1.25, 0.02)
    P = np.unique(np.round(np.array(pts), 10), axis=0)
    T = Delaunay(P).simplices
    c = P[T].mean(axis=1)
    keep = np.hypot(c[:, 0] - Lx / 2, c[:, 1] - W / 2) > d / 2
    T, c = T[keep], c[keep]
    n = len(P)
    D = E / (1 - NU ** 2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])
    rows, cols, vals, Bs = [], [], [], []
    for t in T:
        (x1, y1), (x2, y2), (x3, y3) = P[t]
        A2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        b = np.array([y2 - y3, y3 - y1, y1 - y2]) / A2
        cc = np.array([x3 - x2, x1 - x3, x2 - x1]) / A2
        B = np.zeros((3, 6))
        B[0, 0::2], B[1, 1::2], B[2, 0::2], B[2, 1::2] = b, cc, cc, b
        dof = np.array([2 * t[0], 2 * t[0] + 1, 2 * t[1], 2 * t[1] + 1, 2 * t[2], 2 * t[2] + 1])
        rows.append(np.repeat(dof, 6))
        cols.append(np.tile(dof, 6))
        vals.append((0.5 * abs(A2) * B.T @ D @ B).ravel())
        Bs.append((B, dof))
    K = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(2 * n, 2 * n)).tocsr()
    f = np.zeros(2 * n)
    right = np.where(np.abs(P[:, 0] - Lx) < 1e-12)[0]
    right = right[np.argsort(P[right, 1])]
    for a, b in zip(right[:-1], right[1:]):
        L = P[b, 1] - P[a, 1]
        f[2 * a] += L / 2
        f[2 * b] += L / 2
    left = np.where(np.abs(P[:, 0]) < 1e-12)[0]
    fixed = [2 * i for i in left] + [2 * left[np.argmin(P[left, 1])] + 1]
    free = np.setdiff1d(np.arange(2 * n), fixed)
    u = np.zeros(2 * n)
    u[free] = spsolve(K[free][:, free].tocsc(), f[free])
    sx = np.array([(D @ (B @ u[dof]))[0] for B, dof in Bs])
    near = np.hypot(c[:, 0] - Lx / 2, c[:, 1] - W / 2) < d / 2 + 3 * h_near
    ratio = d / W
    return float(sx[near].max()), (2 + (1 - ratio) ** 3) / (1 - ratio)
