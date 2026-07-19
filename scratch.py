#!/usr/bin/env python3
"""
Lightweight DCP scratch runner.

This script keeps the useful MCP lifecycle from dcp_optimizer.py, but runs only
the first Vivado-side baseline steps so optimization ideas can be tried one by
one without invoking the full agent.


import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path



SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DCP = SCRIPT_DIR / "fpl26_contest_benchmarks" / "logicnets_jscl_2025.1.dcp"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a DCP through the lightweight MCP scratch flow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dcp",
        nargs="?",
        type=Path,
        default=DEFAULT_DCP,
        help="Input design checkpoint (.dcp)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("playground"),
        help="Directory for MCP logs and temporary files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show MCP server output in the console",
    )
    parser.add_argument(
        "--skip-timing",
        action="store_true",
        help="Only open the checkpoint; skip timing queries",
    )
    parser.add_argument(
        "--skip-fanout",
        action="store_true",
        help="Skip critical high-fanout net reporting",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=50,
        help="Number of timing paths used for fanout analysis",
    )
    parser.add_argument(
        "--min-fanout",
        type=int,
        default=100,
        help="Minimum fanout threshold for critical net reporting",
    )
    parser.add_argument(
        "--top-nets",
        type=int,
        default=10,
        help="Number of parsed high-fanout nets to print",
    )
    return parser


async def open_and_summarize_dcp(args: argparse.Namespace) -> int:
    input_dcp = args.input_dcp.resolve()
    if not input_dcp.exists():
        print(f"Error: input DCP not found: {input_dcp}", file=sys.stderr)
        return 1

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        from dcp_optimizer import FPGAOptimizerTest, parse_timing_summary_static
    except ModuleNotFoundError as exc:
        print(f"Error: missing Python dependency while importing dcp_optimizer.py: {exc}", file=sys.stderr)
        print("Hint: run this from the project environment after installing requirements.txt.", file=sys.stderr)
        return 1

    tester = FPGAOptimizerTest(debug=args.debug, run_dir=args.run_dir)
    started_at = time.time()

    try:
        await tester.start_servers()

        print("\n" + "=" * 70)
        print("SCRATCH DCP BASELINE")
        print("=" * 70)
        print(f"Input DCP: {input_dcp}")
        print(f"Run dir:   {tester.run_dir.resolve()}")
        print("=" * 70)

        print("\nSTEP 1: Open DCP in Vivado")
        open_result = await tester.call_vivado_tool(
            "open_checkpoint",
            {"dcp_path": str(input_dcp)},
            timeout=600.0,
        )
        print(open_result)

        if not args.skip_timing:
            print("\nSTEP 2: Baseline timing")
            timing_report = await tester.call_vivado_tool(
                "report_timing_summary",
                {},
                timeout=300.0,
            )
            timing_info = parse_timing_summary_static(timing_report)

            tester.clock_period = await tester.fetch_clock_period()
            target_wns = await tester.get_wns_for_target_clock(tester._call_vivado_for_clock)
            tester.initial_wns = target_wns if target_wns is not None else timing_info["wns"]
            tester.initial_tns = timing_info["tns"]
            tester.initial_failing_endpoints = timing_info["failing_endpoints"]

            tester.print_fmax_status("Initial", tester.initial_wns)
            if tester.initial_tns is not None:
                print(f"[SCRATCH] TNS: {tester.initial_tns:.3f} ns")
            if tester.initial_failing_endpoints is not None:
                print(f"[SCRATCH] Failing endpoints: {tester.initial_failing_endpoints}")

        if not args.skip_fanout:
            print("\nSTEP 3: Critical high-fanout nets")
            fanout_report = await tester.call_vivado_tool(
                "get_critical_high_fanout_nets",
                {
                    "num_paths": args.num_paths,
                    "min_fanout": args.min_fanout,
                    "exclude_clocks": True,
                },
                timeout=600.0,
            )
            tester.high_fanout_nets = tester.parse_high_fanout_nets(fanout_report)
            print(
                f"[SCRATCH] Parsed {len(tester.high_fanout_nets)} nets "
                f"with fanout >= {args.min_fanout}"
            )
            for idx, (net_name, fanout, path_count) in enumerate(
                tester.high_fanout_nets[: args.top_nets],
                start=1,
            ):
                print(f"  {idx:2d}. fanout={fanout:<5d} paths={path_count:<4d} {net_name}")

        elapsed = time.time() - started_at
        print("\n" + "=" * 70)
        print(f"Scratch flow completed in {elapsed:.2f}s")
        print(f"Run directory preserved at: {tester.run_dir.resolve()}")
        print("=" * 70)
        return 0

    except KeyboardInterrupt:
        print("\n[SCRATCH] Interrupted by user")
        return 130
    except Exception as exc:
        logging.exception("Scratch flow failed")
        print(f"\n[SCRATCH] Failed: {type(exc).__name__}: {exc}")
        print(f"[SCRATCH] Run directory: {tester.run_dir.resolve()}")
        return 1
    finally:
        await tester.cleanup()


async def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return await open_and_summarize_dcp(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
"""
import json
import os
import shutil
from pathlib import Path

# ------------------------------------------------------------------
# Find Java from Vivado if java is not on PATH
# ------------------------------------------------------------------

if shutil.which("java") is None:

    vivado = shutil.which("vivado")

    if vivado:
        vivado_root = Path(vivado).parent.parent

        java_candidates = list(
            vivado_root.glob("tps/lnx64/jre11*/bin/java")
        )

        if java_candidates:
            java_bin = java_candidates[0]

            java_home = java_bin.parent.parent

            os.environ["JAVA_HOME"] = str(java_home)
            os.environ["PATH"] = (
                str(java_bin.parent)
                + ":"
                + os.environ["PATH"]
            )

            print(f"Using Vivado Java: {java_bin}")

        else:
            print("WARNING: Could not find Vivado Java")

    else:
        print("WARNING: Vivado not found on PATH")

import asyncio
import logging
from pathlib import Path
import sys
from unittest import result

from dcp_optimizer import FPGAOptimizerTest


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def setup_logging(run_dir: Path) -> tuple[Path, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "scratch.log"
    log_file = open(log_path, "w")

    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(sys.__stdout__),
        ],
        force=True,
    )
    return log_path, log_file


async def main():
#    run_dir = Path("playground")
    run_dir = Path("playground").resolve()
    log_path, log_file = setup_logging(run_dir)
    logger = logging.getLogger(__name__)
    logger.info("Scratch log: %s", log_path.resolve())

    tester = FPGAOptimizerTest(
        debug=False,
        run_dir=run_dir,
    )

    try:
        await tester.start_servers()
        logger.info("Servers started")
        print("Servers started")

        # What RW tools exist?
        tools = await tester.rapidwright_session.list_tools()
        tool_names = [tool.name for tool in tools.tools]

        logger.info("RapidWright tools response: %s", tools)
        print(f"RapidWright tools ({len(tool_names)}):")
        for name in tool_names:
            print(f"  - {name}")
        print("\nInitializing RapidWright...")
        result = await tester.call_rapidwright_tool(
            "initialize_rapidwright",
            {
                "jvm_max_memory": "8G"
            }
        )
        print(result) 

        dcp = Path("fpl26_contest_benchmarks/vexriscv_re-place_2025.1_optimized-20260607_120755.dcp").resolve()
        print("\nReading checkpoint...")

        result = await tester.call_rapidwright_tool(
            "read_checkpoint",
            {
                "dcp_path": str(dcp.resolve())
            },
            timeout=600.0
            )
        print(result)

        print("\nDesign info...")

        result = await tester.call_rapidwright_tool(
            "get_design_info",
            {}
        )
        print(result)
########################################## analyse path spread
        print("Opening DCP in Vivado...")

        result = await tester.call_vivado_tool(
            "open_checkpoint",
            {
                "dcp_path": str(dcp.resolve())
            },
            timeout=600.0
        )

        print(result)

        critical_paths_file = run_dir / "critical_paths.json"

        result = await tester.call_vivado_tool(
            "extract_critical_path_cells",
            {
                "num_paths": 50,
                "output_file": str(critical_paths_file)
            },
            timeout=600.0
        )

        print(result)

        result = await tester.call_rapidwright_tool(
            "analyze_critical_path_spread",
            {
                "input_file": str(critical_paths_file)
            },
            timeout=300.0
        )
        print(result)

####################################### analyse detour

        critical_pins_file = run_dir / "critical_path_pins.json"

        result = await tester.call_vivado_tool(
            "extract_critical_path_pins",
            {
                "num_paths": 50,
                "output_file": str(critical_pins_file)
            },
            timeout=600.0
        )
        print(result)
        import json

        result = await tester.call_rapidwright_tool(
            "analyze_net_detour",
            {
                "input_file": str(critical_pins_file)
            }
        )
        detour_result = json.loads(result)
        print(json.dumps(json.loads(result), indent=2)[:10000])

        from collections import Counter, defaultdict

        counts = Counter()
        ratios = defaultdict(float)

        for c in detour_result["candidates"]:
            counts[c["cell"]] += 1
            ratios[c["cell"]] += c["max_detour_ratio"]

        print("\nRecurring detour hotspots:\n")

        for cell, count in counts.most_common(20):
            avg_ratio = ratios[cell] / count
            print(
                f"{count:3d} paths | "
                f"avg detour {avg_ratio:6.2f} | "
                f"{cell}"
            )
        import json

        # ------------------------------------------------------------------
        # STEP 3 (updated): target the exact cells from the posted path.
        # This keeps the flow focused on the logic that is currently being
        # pulled into the X3Y0 region instead of staying near the intended
        # X4Y0/X11Y13-style critical path neighborhood.
        # ------------------------------------------------------------------
        critical_path_cells = [
            "memory_to_writeBack_MUL_LOW_reg[43]",
            "dataCache_1/HazardSimplePlugin_writeBackBuffer_payload_data[14]_i_7",
            "dataCache_1/HazardSimplePlugin_writeBackBuffer_payload_data_reg[14]_i_2",
            "dataCache_1/HazardSimplePlugin_writeBackBuffer_payload_data_reg[22]_i_2",
            "dataCache_1/HazardSimplePlugin_writeBackBuffer_payload_data[18]_i_1",
            "dataCache_1/RegFilePlugin_regFile_reg_r1_0_31_14_27_i_6",
            "RegFilePlugin_regFile_reg_r2_0_31_14_27",
        ]

        print("\n" + "="*80)
        print("CRITICAL PATH CELL INVESTIGATION")
        print("="*80)
        for cell in critical_path_cells:
            result = await tester.call_rapidwright_tool(
                "search_cells",
                {
                    "pattern": cell
                }
            )

            print("\n" + "-"*60)
            print(cell)
            print(result)

        result = await tester.call_rapidwright_tool(
            "optimize_cell_placement",
            {
                "cell_names": critical_path_cells
            }
        )

        print(result)

        # After optimize_cell_placement()

        rw_dcp = run_dir / "temp_rw.dcp"

        await tester.call_rapidwright_tool(
            "write_checkpoint",
            {
                "dcp_path": str(rw_dcp)
            },
            timeout=600.0
        )

        # Open in Vivado
        await tester.call_vivado_tool(
            "open_checkpoint",
            {
                "dcp_path": str(rw_dcp)
            },
            timeout=600.0
        )

        # Re-route
        await tester.call_vivado_tool(
            "run_tcl",
            {
                "command": "route_design"
            },
            timeout=3600.0
        )

        # Save routed design
        final_dcp = run_dir / "411mhz_input.dcp"

        await tester.call_vivado_tool(
            "run_tcl",
            {
                "command": f"write_checkpoint -force {final_dcp}"
            },
            timeout=600.0
        )
    except Exception:
        logger.exception("Scratch run failed")
        raise
    finally:
        logger.info("Cleaning up MCP servers")
        await tester.cleanup()
        logger.info("Cleanup complete")
        log_file.close()



asyncio.run(main())
