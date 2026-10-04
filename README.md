# openings

## Same steel beam, same web hole: how much earlier does first yield come?

An idealised **IPE 300** (catalogue h, b, tf, tw; root radii omitted) is simply supported under a
uniform load per metre. A 150 mm round web opening (half the depth) sits at mid-depth with its edge
0.60 m from the left support. Four beams differ only in span. Each holed beam is compared with its
own unholed twin: the load at which **global first yield** (peak von Mises anywhere) appears.

| span | span/depth | holed first yield, % of its twin (shear-traction ends) | with 100 mm bearing pads |
|---|---|---|---|
| 4.0 m | 13.3 | **77.7 %** | 74.9 % |
| 4.5 m | 15.0 | **83.4 %** | 80.7 % |
| 5.0 m | 16.7 | **89.3 %** | 86.8 % |
| 6.0 m | 20.0 | **100.0 %** | 99.4 % |

In the shorter beams the opening sits in a shear-heavy region and brings first yield forward. Bending
grows faster with span than shear does, and by span/depth 20 the opening barely changes first yield:
with shear-traction ends it moves to the midspan flange, with bearing pads it stays at the opening. The same load per metre
means the total load grows with span.

```sh
python -m pip install -r requirements.txt
python reproduce.py          # 0.6 mm hole-edge mesh, about a minute
python reproduce.py --full   # also the 0.3 mm production mesh
```

The solver is 2-D plane stress through the depth with each triangle's thickness set by its height
(flange width or web thickness), body-fitted rings on the hole edge, and the end reactions applied
as the beam-theory shear traction V·Q(y)/I (or, as a sensitivity, as bearing pressure on 100 mm pads).
`reproduce.py` re-solves every beam, checks the mesh geometry (hole nodes on the circle, no triangle
across a flange–web interface), the Kirsch/Heywood strip-with-hole value, and every ratio against
`reference/result.json` and the solver's SHA-256.

## What this is not

* **Not a design check.** Linear elastic, first yield only: no plasticity, web buckling, Vierendeel
  collapse, stiffeners, root radii, residual stress or composite action. Real openings follow design
  guides; the 0.60 m clearance is a fixed geometric choice here, not a rule this model checks.
