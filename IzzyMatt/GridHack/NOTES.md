# GridHack — TPSA Hackathon 2026: Efficiency in Ireland's Electrical Grid

## Layout
- `Hackathon_2026_Problem_Sheet-2.pdf` — the problem brief
- `repo/` — clone of https://github.com/farrencc/Hackathons
- `repo/grid_TF_Wind/participant-kit/` — **the kit you actually work in**
  - `.venv/` — Python 3.12 environment, deps installed, 22/22 tests passing

## Running things
```bash
cd repo/grid_TF_Wind/participant-kit
.venv/Scripts/python.exe examples/a_dc_power_flow.py     # or: source .venv/Scripts/activate
```
System Python is 3.9 (Visual Studio's) and cannot run PyPSA 1.3 — always use the venv.

## The kit in one screen
```python
import gridkit, flowmath
gridkit.quiet()
n = gridkit.load("WP2033", "north-west")   # scenarios: WP2024 SV2024 WP2033 SV2033
                                           # scopes: all-island | north-west
gridkit.solve(n)                           # HiGHS LOPF
gridkit.dispatch_down(n)                   # renewable energy the dispatch refused
gridkit.binding(n)                         # circuits at their limit, and for how long
gridkit.line_loading(n)
flowmath.ptdf(n)                           # power transfer distribution factors
flowmath.shift_factors(n, monitored="<line>")
flowmath.susceptibility(n, snapshot)       # edge-to-edge, for Braess-type effects
gridkit.add_battery(n, bus, p_nom, hours=4)
gridkit.set_rating(n, name, s_nom)         # DLR experiments
```
**Two gotchas:** call `gridkit.freeze_dispatch(n)` between `optimize` and any load flow;
use `gridkit.placed_buses(n)` when mapping (drops un-geocoded buses).

## Scenarios
| Case | Condition | Peak demand | Connected capacity |
|---|---|---|---|
| WP2024 | Winter peak, 2024 network | 7,325 MW | 22,650 MW |
| SV2024 | Summer valley, 2024 | 4,948 MW | 14,591 MW |
| WP2033 | Winter peak, 2033 | 8,792 MW | 42,574 MW |
| SV2033 | Summer valley, 2033 | 6,058 MW | 18,283 MW |

WP2033 has the headroom where constraint/curtailment lives (42.6 GW plant vs 8.8 GW peak).

## Examples map onto problem-sheet questions
| Script | Gives you |
|---|---|
| `a_dc_power_flow.py` | dispatch + DC flow + island map coloured by loading |
| `b_lopf_dispatch.py` | least-cost dispatch, dispatch-down accounting |
| `c_capacity_expansion.py` | battery/line siting & sizing (§3.1, §3.7) |
| `d_ptdf.py` | PTDF matrix (§3.2 correlation maps) |
| `e_braess_susceptibility.py` | edge-to-edge effects, non-local phenomena (§3.5) |
| `f_shift_factors.py` | shift factors — EirGrid's own grouping metric (§3.1, §3.2) |

## Caveats to state in any write-up
DC approximation. Not validated against real network parameters. 87% of transmission
buses geocoded (78% in the 2033 cases). Seven circuits carry no length — skip them in
per-km calculations rather than treating the gap as a zero.
