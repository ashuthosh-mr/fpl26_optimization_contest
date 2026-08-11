# Benchmark Results — mouna flow

Full-flow deterministic recipe
(`phys_opt → pblock → relocate → cell_replace → pinopt → reimpl` + LLM finisher),
run via `dcp_optimizer.py`. This is the version submitted to the **FPL'26 final round**
(`solution.zip`, md5 `1bf86a7d76cc78f35be129e4a18270d7`).

★ = benchmark that was scored in the **beta** round.

| Benchmark | Fmax (out) | Improvement | Runtime | Cost | Validation |
|---|---:|---:|---:|---:|:--|
| vexriscv_re-place (v1)        | 437.25 MHz | +127.08 | 338s  | $0.0066 | PASS |
| logicnets_jscl                | 524.11 MHz | +120.56 | 641s  | $0.0143 | PASS |
| amd_mini-isp ★                | 411.86 MHz | +104.74 | 193s  | $0.0067 | PASS |
| rosetta_digit-recognition     | 421.76 MHz |  +54.79 | 730s  | $0.0216 | PASS |
| finn_radioml                  | 333.56 MHz |  +48.66 | 1027s | n/a     | PASS |
| corescore_500_mod             | 385.21 MHz |  +40.97 | 1288s | $0.0232 | PASS |
| rosetta_optical-flow ★        | 346.38 MHz |  +21.49 | 1177s | $0.0103 | PASS |
| fir_systolic_transposed ★     | 374.25 MHz |  +18.76 | 622s  | $0.0047 | PASS |
| boom_soc (v2) ★               |  88.36 MHz |  +11.21 | ~52m  | n/a     | high roll |
| rosetta_3d-rendering          | 281.77 MHz |  +10.84 | 1709s | $0.0556 | PASS |
| vexriscv_re-place_v2          | 406.50 MHz |   +9.05 | 671s  | $0.0081 | PASS |
| vtr_mcml (v2) ★               |  71.12 MHz |   +1.79 | 2802s | $0.0165 | PASS |
| rosetta_spam-filter           | 437.45 MHz |   +0.00 | 571s  | $0.0093 | PASS (at ceiling) |

**Notes**
- `boom_soc` v2 is **placement-variance bound** (~80–88 MHz depending on the `place_design`
  roll); 88.36 is a high roll (near the ceiling). A single run can land anywhere in that band.
- `vtr_mcml` is **logic-depth / carry capped** — a low ceiling for any tool (placement/routing
  cannot shorten a combinational carry chain).
- `rosetta_spam-filter` is already at its timing ceiling (+0.00).
- All placement/routing-only results are functionally equivalent by construction; netlist-editing
  stages (`cell_replace`) are guarded by a final equivalence-simulation gate that falls back to the
  original design if a round-trip ever breaks equivalence.
- **Not run:** `ispd16_example2` (146 MB) — a known blind spot if it appears in the hidden final set.

## Final-round public preview (official harness)

Run by the contest harness on the 5 public benchmarks. **Total score 144.784**, every
validation gate PASS (`par_routed`, `par_drc_clean`, `hold`, `pulse_width`, `sim`).

| Benchmark | Fmax in→out | α (MHz) | β ($) | γ (h) | Score |
|---|---:|---:|---:|---:|---:|
| amd_mini-isp        | 307 → 412 | +104.74 | 0.086 | 0.18 | 102.00 |
| rosetta_optical-flow| 325 → 346 |  +21.49 | 0.025 | 0.52 |  20.32 |
| fir_systolic        | 355 → 374 |  +18.90 | 0.024 | 0.32 |  18.26 |
| boom_soc_v2         |  77 → 80  |   +2.76 | 0.000 | 1.00 |   2.49 |
| vtr_mcml_v2         |  69 → 71  |   +1.91 | 0.027 | 0.96 |   1.72 |

Score model: `score = α − 0.1·α·β − 0.1·α·γ` (α = Fmax improvement in MHz, β = OpenRouter
cost in USD, γ = runtime in hours, capped at 1.0). The preview's `boom` was an unlucky
low roll; the organizers re-run and take the best, so the scored `boom` can be higher.
