"""Re-solve the four beams and their twins and check every published number.

    python reproduce.py [--full]

Exits non-zero, naming the quantity, if anything disagrees with reference/result.json or if the
solver's SHA-256 has changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "solver"))
import numpy as np  # noqa: E402
import web_opening as W  # noqa: E402

R = json.load(open(os.path.join(HERE, "reference", "result.json"), encoding="utf-8"))


def check(label, ok):
    if not ok:
        raise AssertionError(label)


def peak(span, holes, support, h_near):
    P, T = W.make_mesh(holes, span=span, h_near=h_near)
    s = W.solve(P, T, support=support, span=span)
    m = (s["cent"][:, 0] > R["read_window_inset"]) & (s["cent"][:, 0] < span - R["read_window_inset"])
    return float(s["vm"][m].max()), P, T


def main():
    with open(os.path.join(HERE, "solver", "web_opening.py"), "rb") as f:
        got = hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()
    check("solver module changed", got == R["solver_sha256"])
    k, ref = W.kirsch()
    check("Kirsch/Heywood %.4f vs %.4f" % (k, ref), abs(k / ref - 1) < 0.02)
    hole = (R["hole_centre_x"], R["h"] / 2, R["hole_diameter"] / 2)
    levels = [("0.0006", "ratio_at_0p6mm")] + ([("0.0003", "ratio_at_0p3mm")] if "--full" in sys.argv else [])
    for span_s in sorted(R["ratio_at_0p6mm"]):
        span = float(span_s)
        for hn_s, key in levels:
            hn = float(hn_s)
            solid, _, _ = peak(span, [], "traction", 0.0012)
            holed, P, T = peak(span, [hole], "traction", hn)
            mc = W.mesh_checks(P, T, [hole], span)
            check("%s m mesh geometry %r" % (span_s, mc), mc["worst_hole_node_off_circle"] < 1e-9 and
                  mc["boundary_edges_off_geometry"] == 0 and mc["triangles_straddling_interface"] == 0)
            ratio = holed / solid
            want = R[key][span_s]
            check("%s m ratio (%s mm) %r != %r" % (span_s, hn_s, ratio, want),
                  abs(ratio / want - 1) < R["tolerance_relative"])
            print("span %s m: holed first yield at %.1f %% of its twin (hole-edge mesh %.1f mm)"
                  % (span_s, 100 / ratio, 1000 * hn))
    print("EVERYTHING REPRODUCES")


if __name__ == "__main__":
    main()
