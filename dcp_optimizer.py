#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions of this file consist of AI-generated content.
# SPDX-License-Identifier: Apache 2.0

"""
FPGA Design Optimization Agent

An autonomous AI agent that analyzes FPGA designs and applies optimizations
using RapidWright and Vivado via MCP servers.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"


def parse_timing_summary_static(timing_report: str) -> dict:
    """
    Parse timing summary report to extract WNS, TNS, and failing endpoints.
    Returns dict with keys: wns, tns, failing_endpoints
    
    Parses the Design Timing Summary table:
        WNS(ns)      TNS(ns)  TNS Failing Endpoints  ...
        -------      -------  ---------------------  ...
         -0.099       -1.449                     42  ...
    
    This is a shared utility function used by both FPGAOptimizer and FPGAOptimizerTest.
    """
    result = {
        "wns": None,
        "tns": None,
        "failing_endpoints": None
    }
    
    lines = timing_report.split('\n')
    
    # Find the line with "WNS(ns)" header
    header_idx = -1
    for i, line in enumerate(lines):
        if 'WNS(ns)' in line and 'TNS(ns)' in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return result
    
    # The data line should be 2 lines after the header (skipping the dashes line)
    # Format: whitespace + values separated by whitespace
    data_idx = header_idx + 2
    if data_idx >= len(lines):
        return result
    
    data_line = lines[data_idx].strip()
    if not data_line:
        return result
    
    # Split by whitespace and extract first 3 values: WNS, TNS, TNS Failing Endpoints
    parts = data_line.split()
    if len(parts) >= 3:
        try:
            result["wns"] = float(parts[0])
            result["tns"] = float(parts[1])
            result["failing_endpoints"] = int(parts[2])
        except (ValueError, IndexError):
            # If parsing fails, leave as None
            pass
    
    return result


def load_system_prompt() -> str:
    """Load system prompt from SYSTEM_PROMPT.TXT file."""
    script_dir = Path(__file__).parent.resolve()
    prompt_file = script_dir / "SYSTEM_PROMPT.TXT"
    
    try:
        with open(prompt_file, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"System prompt file not found: {prompt_file}")
        raise
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        raise


def load_planner_prompt() -> Optional[str]:
    """Load the LLM-planner prompt from PLANNER_PROMPT.TXT. Returns None if missing
    (the caller then falls back to the default deterministic recipe)."""
    prompt_file = Path(__file__).parent.resolve() / "PLANNER_PROMPT.TXT"
    try:
        with open(prompt_file, 'r') as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Planner prompt not loadable ({e}); planner will fall back to default recipe.")
        return None


def convert_mcp_tool_to_openai(tool, server_prefix: str) -> dict:
    """Convert MCP tool definition to OpenAI-compatible format with server prefix."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": f"{server_prefix}_{tool.name}",
            "description": tool.description or "",
            "parameters": {
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", [])
            }
        }
    }


class DCPOptimizerBase:
    """Base class with shared functionality for FPGA optimization."""
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        self.debug = debug
        
        # Create run directory if not provided
        if run_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
            self.run_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created run directory: {self.run_dir}")
        else:
            self.run_dir = run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.exit_stack = AsyncExitStack()
        self.rapidwright_session: Optional[ClientSession] = None
        self.vivado_session: Optional[ClientSession] = None
        
        # Use run directory for all temporary files
        self.temp_dir = self.run_dir
        logger.info(f"Working directory: {self.temp_dir}")
        
        # Timing tracking
        self.initial_wns = None
        self.initial_tns = None
        self.initial_failing_endpoints = None
        self.high_fanout_nets = []
        self.clock_period = None
        self.target_clock = None  # Set to clock name (e.g. "clk_fpl26contest") for clock-specific Fmax
        # Critical-path spread (drives whether a pblock re-placement is worthwhile)
        self.critical_path_spread_info = None
        self.pblock_recommended = False
        
        # Log file handles
        self._rw_log_file = None
        self._v_log_file = None
    
    async def start_servers(self, log_prefix: str = ""):
        """Start and connect to both MCP servers."""
        script_dir = Path(__file__).parent.resolve()
        
        # Create log files in run directory
        rapidwright_log = self.run_dir / "rapidwright.log"
        rapidwright_mcp_log = self.run_dir / "rapidwright-mcp.log"
        vivado_log = self.run_dir / "vivado.log"
        vivado_journal = self.run_dir / "vivado.jou"
        vivado_mcp_log = self.run_dir / "vivado-mcp.log"
        
        # Open log files (if not in debug mode, redirect stderr to log)
        if self.debug:
            self._rw_log_file = None
            self._v_log_file = None
            logger.info("Debug mode: MCP server output will be shown in console")
            if log_prefix:
                print(f"{log_prefix} Debug mode: MCP server output will be shown in console")
        else:
            self._rw_log_file = open(rapidwright_mcp_log, 'w')
            self._v_log_file = open(vivado_mcp_log, 'w')
            logger.info(f"RapidWright Java output: {rapidwright_log}")
            logger.info(f"RapidWright MCP output: {rapidwright_mcp_log}")
            logger.info(f"Vivado output: {vivado_log}")
            logger.info(f"Vivado journal: {vivado_journal}")
            logger.info(f"Vivado MCP output: {vivado_mcp_log}")
            print(f"Log files in {self.run_dir.name}/: {rapidwright_log.name}, {rapidwright_mcp_log.name}, {vivado_log.name}, {vivado_journal.name}, {vivado_mcp_log.name}")
        
        # RapidWright MCP server config
        rapidwright_args = [str(script_dir / "RapidWrightMCP" / "server.py")]
        if not self.debug:
            rapidwright_args.extend([
                "--java-log", str(rapidwright_log),
                "--mcp-log", str(rapidwright_mcp_log)
            ])
        
        env = {**os.environ}
        rapidwright_submodule = script_dir / "RapidWright"
        if rapidwright_submodule.is_dir() and "RAPIDWRIGHT_PATH" not in env:
            env["RAPIDWRIGHT_PATH"] = str(rapidwright_submodule)
            env["CLASSPATH"] = f"{rapidwright_submodule}/bin:{rapidwright_submodule}/jars/*"
        
        rapidwright_config = {
            "command": sys.executable,
            "args": rapidwright_args,
            "cwd": str(self.run_dir),
            "env": env
        }
        
        # Vivado MCP server config
        vivado_args = [str(script_dir / "VivadoMCP" / "vivado_mcp_server.py")]
        if not self.debug:
            vivado_args.extend([
                "--vivado-log", str(vivado_log),
                "--vivado-journal", str(vivado_journal)
            ])
        
        vivado_config = {
            "command": sys.executable,
            "args": vivado_args,
            "cwd": str(self.run_dir),
            "env": {**os.environ}
        }
        
        # Start RapidWright MCP
        logger.info("Starting RapidWright MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting RapidWright MCP server...")
        start_time = time.time()
        
        rw_params = StdioServerParameters(**rapidwright_config)
        rw_transport = await self.exit_stack.enter_async_context(
            stdio_client(rw_params, errlog=self._rw_log_file)
        )
        rw_read, rw_write = rw_transport
        self.rapidwright_session = await self.exit_stack.enter_async_context(
            ClientSession(rw_read, rw_write)
        )
        await self.rapidwright_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"RapidWright MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} RapidWright MCP server started in {elapsed:.2f}s")
        
        # Start Vivado MCP
        logger.info("Starting Vivado MCP server...")
        if log_prefix:
            print(f"{log_prefix} Starting Vivado MCP server...")
        start_time = time.time()
        
        vivado_params = StdioServerParameters(**vivado_config)
        vivado_transport = await self.exit_stack.enter_async_context(
            stdio_client(vivado_params, errlog=self._v_log_file)
        )
        v_read, v_write = vivado_transport
        self.vivado_session = await self.exit_stack.enter_async_context(
            ClientSession(v_read, v_write)
        )
        await self.vivado_session.initialize()
        
        elapsed = time.time() - start_time
        logger.info(f"Vivado MCP server started in {elapsed:.2f}s")
        if log_prefix:
            print(f"{log_prefix} Vivado MCP server started in {elapsed:.2f}s")
        
        logger.info("Both MCP servers connected")
        if log_prefix:
            print(f"{log_prefix} Both MCP servers connected successfully")
    
    async def cleanup(self):
        """Clean up resources."""
        await self.exit_stack.aclose()
        
        if self._rw_log_file:
            self._rw_log_file.close()
        if self._v_log_file:
            self._v_log_file.close()
        
        logger.info(f"Run directory preserved at: {self.run_dir}")
    
    def calculate_fmax(self, wns: Optional[float], clock_period: Optional[float]) -> Optional[float]:
        """
        Calculate achievable fmax in MHz based on WNS and clock period.
        
        fmax = 1 / (clock_period - WNS) when WNS < 0 (timing violation)
        fmax = 1 / clock_period when WNS >= 0 (timing met)
        
        Returns fmax in MHz, or None if cannot be calculated.
        """
        if clock_period is None or clock_period <= 0:
            return None
        if wns is None:
            return None
        
        achievable_period_ns = clock_period - wns
        if achievable_period_ns <= 0:
            return None
        
        return 1000.0 / achievable_period_ns
    
    async def get_clock_period(self, call_tool_fn) -> Optional[float]:
        """
        Query the clock period of the target clock from Vivado in nanoseconds.
        
        First checks for the contest clock 'clk_fpl26contest'. If found, uses its
        period and sets self.target_clock. Otherwise falls back to the endpoint clock
        of the worst setup timing path.
        
        Args:
            call_tool_fn: Function to call Vivado tools, should accept (tool_name, arguments)
        
        Returns the period of the target clock, or None if no clocks found.
        """
        tcl_cmd = (
            "set contest_clk [get_clocks -quiet clk_fpl26contest]; "
            "if {$contest_clk ne {}} { "
            "  puts \"CLOCK:clk_fpl26contest\"; "
            "  puts [get_property PERIOD $contest_clk]; "
            "} else { "
            "  set tp [get_timing_paths -max_paths 1 -setup]; "
            "  if {$tp ne {}} { "
            "    set clk [get_property ENDPOINT_CLOCK $tp]; "
            "    if {$clk ne {}} { "
            "      puts \"CLOCK:$clk\"; "
            "      puts [get_property PERIOD [get_clocks $clk]]; "
            "    } "
            "  } "
            "}"
        )
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            
            clock_name = None
            for token in result.strip().split():
                if token.startswith('CLOCK:'):
                    clock_name = token[len('CLOCK:'):]
                    continue
                if token.startswith('ERROR') or token.startswith('WARNING'):
                    continue
                try:
                    period = float(token)
                    if period > 0:
                        if clock_name:
                            self.target_clock = clock_name
                            logger.info(f"Target clock: {clock_name}, period: {period:.3f} ns")
                        else:
                            logger.info(f"Critical clock period: {period:.3f} ns")
                        return period
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to get clock period: {e}")
        
        logger.warning("Could not determine clock period from Vivado")
        return None
    
    async def get_wns_for_target_clock(self, call_tool_fn) -> Optional[float]:
        """
        Get WNS specifically for the target clock domain.
        
        When target_clock is set (e.g. 'clk_fpl26contest'), queries WNS filtered
        to that clock's timing paths. Falls back to overall WNS if no target clock.
        
        Args:
            call_tool_fn: Function to call Vivado tools, should accept (tool_name, arguments)
        
        Returns WNS in nanoseconds, or None if query fails.
        """
        if self.target_clock:
            tcl_cmd = (
                f"set clk_obj [get_clocks -quiet {{{self.target_clock}}}]; "
                f"if {{$clk_obj ne {{}}}} {{ "
                f"  set tp [get_timing_paths -max_paths 1 -setup -to $clk_obj]; "
                f"  if {{[llength $tp] > 0}} {{get_property SLACK $tp}} else {{puts 0.0}} "
                f"}} else {{ "
                f"  set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                f"  if {{[llength $tp] > 0}} {{get_property SLACK $tp}} else {{puts 0.0}} "
                f"}}"
            )
        else:
            tcl_cmd = (
                "set tp [get_timing_paths -max_paths 1 -slack_lesser_than 999]; "
                "if {[llength $tp] > 0} {get_property SLACK $tp} else {puts 0.0}"
            )
        
        try:
            result = await call_tool_fn("run_tcl", {"command": tcl_cmd})
            for token in result.strip().split('\n'):
                token = token.strip()
                if not token or token.startswith('ERROR') or token.startswith('WARNING'):
                    continue
                try:
                    wns = float(token)
                    clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
                    logger.info(f"WNS{clock_info}: {wns:.3f} ns")
                    return wns
                except ValueError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to get WNS for target clock: {e}")
        
        return None
    
    def parse_high_fanout_nets(self, report: str) -> list[tuple[str, int, int]]:
        """
        Parse high fanout nets report and return list of (net_name, fanout, path_count).
        """
        nets = []
        lines = report.split('\n')
        in_net_section = False
        
        for line in lines:
            if 'Paths' in line and 'Fanout' in line and 'Parent Net Name' in line:
                in_net_section = True
                continue
            
            if in_net_section:
                if line.startswith('---') or not line.strip():
                    continue
                if line.startswith('==='):
                    break
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        path_count = int(parts[0])
                        fanout = int(parts[1])
                        net_name = parts[2]
                        
                        if (net_name and 
                            '/' in net_name and
                            not net_name.startswith('get_') and
                            not net_name.startswith('ERROR') and
                            not net_name.startswith('WARNING')):
                            nets.append((net_name, fanout, path_count))
                    except ValueError:
                        continue
        
        return nets

    def _format_fmax_results(
        self,
        clock_period: Optional[float],
        initial_wns: Optional[float],
        result_wns: Optional[float],
        result_label: str = "Final",
    ) -> list[str]:
        """Format Fmax/WNS results block as a list of lines.
        
        """
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        result_fmax = self.calculate_fmax(result_wns, clock_period)
        result_fmax_label = f"{result_label} Fmax:"
        result_wns_label = f"{result_label} WNS:"
        
        lines: list[str] = []
        if initial_fmax is not None and result_fmax is not None:
            target_fmax = 1000.0 / clock_period
            fmax_change = result_fmax - initial_fmax
            lines.append(f"  {'Target Fmax:':<21s}{target_fmax:8.2f} MHz  (clock period: {clock_period:.3f} ns)")
            lines.append(f"  {'Initial Fmax:':<21s}{initial_fmax:8.2f} MHz  (WNS: {initial_wns:.3f} ns)")
            lines.append(f"  {result_fmax_label:<21s}{result_fmax:8.2f} MHz  (WNS: {result_wns:.3f} ns)")
            lines.append(f"  {'Fmax Improvement:':<21s}{fmax_change:+8.2f} MHz  (WNS: {result_wns - initial_wns:+.3f} ns)")
        else:
            if clock_period is not None:
                target_fmax = 1000.0 / clock_period
                lines.append(f"  {'Clock period:':<21s}{clock_period:8.3f} ns (target: {target_fmax:.2f} MHz)")
            if initial_wns is not None:
                fmax_str = f"  (fmax: {initial_fmax:.2f} MHz)" if initial_fmax else ""
                lines.append(f"  {'Initial WNS:':<21s}{initial_wns:8.3f} ns{fmax_str}")
            if result_wns is not None:
                fmax_str = f"  (fmax: {result_fmax:.2f} MHz)" if result_fmax else ""
                lines.append(f"  {result_wns_label:<21s}{result_wns:8.3f} ns{fmax_str}")
            if initial_wns is not None and result_wns is not None:
                lines.append(f"  {'WNS Improvement:':<21s}{result_wns - initial_wns:+8.3f} ns")
        
        return lines
    
    
    def print_wns_change(
        self,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float]
    ):
        """Print Fmax/WNS change comparison with improvement/regression status."""
        if final_wns is None or initial_wns is None:
            return
        
        initial_fmax = self.calculate_fmax(initial_wns, clock_period)
        final_fmax = self.calculate_fmax(final_wns, clock_period)
        
        if initial_fmax is not None and final_fmax is not None:
            fmax_improvement = final_fmax - initial_fmax
            pct = (fmax_improvement / initial_fmax) * 100 if initial_fmax else 0
            print(f"\n*** Fmax: {initial_fmax:.2f} -> {final_fmax:.2f} MHz ({fmax_improvement:+.2f} MHz, {pct:+.1f}%) ***")
            print(f"*** WNS:  {initial_wns:.3f} -> {final_wns:.3f} ns ***")
            if fmax_improvement > 0:
                print(f"IMPROVEMENT: Fmax improved by {fmax_improvement:.2f} MHz")
            elif fmax_improvement < 0:
                print(f"REGRESSION: Fmax got worse by {-fmax_improvement:.2f} MHz")
            else:
                print("NO CHANGE: Fmax is the same")
        else:
            wns_improvement = final_wns - initial_wns
            print(f"\n*** WNS: {initial_wns:.3f} -> {final_wns:.3f} ns ({wns_improvement:+.3f} ns) ***")
            if wns_improvement > 0:
                print(f"IMPROVEMENT: WNS improved by {wns_improvement:.3f} ns")
            elif wns_improvement < 0:
                print(f"REGRESSION: WNS got worse by {-wns_improvement:.3f} ns")
            else:
                print("NO CHANGE")
    
    def print_fmax_status(self, label: str, wns: Optional[float]):
        """Print Fmax (primary) and WNS (secondary) for a given measurement point."""
        if wns is None:
            print(f"*** {label}: WNS unknown ***")
            return
        fmax = self.calculate_fmax(wns, self.clock_period)
        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        if fmax is not None:
            print(f"*** {label} Fmax{clock_info}: {fmax:.2f} MHz (WNS: {wns:.3f} ns) ***")
        else:
            print(f"*** {label} WNS{clock_info}: {wns:.3f} ns ***")
    
    def print_test_summary(
        self,
        title: str,
        elapsed_seconds: float,
        initial_wns: Optional[float],
        final_wns: Optional[float],
        clock_period: Optional[float],
        extra_info: str = ""
    ):
        """Print formatted test summary."""
        print("\n" + "="*70)
        print(title)
        print("="*70)
        print(f"Total runtime: {elapsed_seconds:.2f} seconds ({elapsed_seconds/60:.2f} minutes)")
        
        result_lines = self._format_fmax_results(clock_period, initial_wns, final_wns)
        if result_lines:
            print(f"\nFmax Results:")
            print("\n".join(result_lines))
        
        if extra_info:
            print(f"\n{extra_info}")
        print("="*70)


class DCPOptimizer(DCPOptimizerBase):
    """FPGA Design Optimization Agent using RapidWright and Vivado MCPs."""
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        debug: bool = False,
        run_dir: Optional[Path] = None,
        pre_opt: str = "phys_opt,pblock,relocate,pinopt,reimpl,routeopt",
        physopt_directive: str = "",
        pblock_mode: str = "always",
        cell_replace_mode: str = "auto",
        relocate_mode: str = "always",
        retime_mode: str = "always",
        reimpl_mode: str = "always",
        reimpl_place_directive: str = "ExtraTimingOpt",
        reimpl_route_directive: str = "AggressiveExplore",
        reimpl_skip_if_recovered: float = 0.30,
        skip_llm: bool = False,
        use_planner: bool = True,
        phys_opt_timeout: int = 1200,
        manual_timeout: int = 1200,
        reimpl_timeout: int = 2400,
        llm_timeout: int = 1200,
        total_timeout: int = 3600,
        hard_deadline: int = 3420,
        cost_cap: float = 1.0
    ):
        super().__init__(debug=debug, run_dir=run_dir)

        self.api_key = api_key
        self.model = model
        # Deterministic pre-LLM optimization pipeline, run in order before the LLM.
        # e.g. "phys_opt,pblock" -> phys_opt baseline, then pblock re-placement.
        self.pre_opt_steps = [s.strip() for s in (pre_opt or "").split(",")
                              if s.strip() and s.strip() != "none"]
        self.physopt_directive = physopt_directive  # optional phys_opt directive
        self.pblock_mode = pblock_mode            # auto | always | never
        self.cell_replace_mode = cell_replace_mode  # auto | always | never
        self.relocate_mode = relocate_mode        # auto | always | never
        self.retime_mode = retime_mode            # auto | always | never
        self.reimpl_mode = reimpl_mode            # auto | always | never (final fallback stage)
        self.reimpl_place_directive = reimpl_place_directive  # place_design directive for re-impl
        self.reimpl_route_directive = reimpl_route_directive  # route_design directive for re-impl
        # Skip re-impl once the incremental stages have recovered this fraction of the
        # initial slack deficit (re-impl only wins from a weak incremental result).
        self.reimpl_skip_if_recovered = reimpl_skip_if_recovered
        self.skip_llm = skip_llm                  # stop after deterministic baseline
        # LLM PLANNER (front): one up-front call reads the design diagnosis and picks the
        # ordered recipe of OPTIONAL stages that run AFTER the mandatory phys_opt. Advisory
        # only -- any failure falls back to the full default recipe, so the floor is never
        # worse than the proven deterministic flow. Disabled by --skip-llm or --no-planner.
        self.use_planner = use_planner
        self._planner_ran = False
        # Wall-clock budgets (seconds) and cost cap ($). Contest limit: 1 hr + $1/benchmark.
        # Phased 20/20/20: phys_opt | manual (pblock + cell_replace SHARE this) | LLM.
        self.phys_opt_timeout = phys_opt_timeout  # phase-1 cap for phys_opt
        self.manual_timeout = manual_timeout      # phase-2 cap SHARED by pblock + cell_replace
        self.reimpl_timeout = reimpl_timeout      # dedicated cap for the re-impl fallback stage
        self.llm_timeout = llm_timeout            # phase-3 cap for the LLM stage
        self.total_timeout = total_timeout        # soft cap: gates STAGE STARTS only
        # HARD wall-clock kill. The stage budgets above are SOFT -- a running Vivado
        # place/route cannot be interrupted (the MCP server waits up to 3600s for it),
        # so on huge designs (e.g. boom) a stage entered just under total_timeout can
        # overrun to 2.5h. This watchdog GUARANTEES we stop under the contest's 1-hour
        # limit: at hard_deadline it force-writes the protected-best DCP as the scored
        # output and kills the process. Because the protected best is saved (and
        # equivalence-checked) after every successful stage, the output is always valid.
        self.hard_deadline = hard_deadline
        self._watchdog: Optional[threading.Timer] = None
        self.cost_cap = cost_cap                  # stop LLM before spending more than this
        # Output-DCP protection: the contest scores the most-recently-modified
        # *_optimized*.dcp, so we keep a protected copy of the best design and
        # guarantee the final written output is never worse than it.
        self.output_dcp: Optional[Path] = None
        self._golden_input: Optional[Path] = None  # original input DCP, for equivalence gating
        self.protected_best_dcp: Optional[Path] = None
        self.protected_best_wns = float('-inf')
        # Set True only when a stage EDITS the netlist (cell_replace's RapidWright
        # round-trip, retime, or the LLM). Placement/routing-only stages (phys_opt,
        # pblock, relocate, pinopt, reimpl) leave the netlist untouched and are provably
        # equivalent, so finalize skips the expensive simulation check when this stays False.
        self._netlist_touched = False
        self.tools: list[dict] = []
        self.messages: list[dict] = []
        
        # LLM client is only needed when the LLM stage runs; skip when baseline-only.
        self.openai = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        ) if api_key else None
        
        # Track optimization progress
        self.iteration = 0
        self.best_wns = float('-inf')
        self.no_improvement_count = 0
        self.llm_call_count = 0
        
        # Track token usage and costs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_call_details = []
        
        # Track all tool calls with timing and WNS
        self.tool_call_details = []
        
        # Track total runtime
        self.start_time = None
        self.end_time = None
    
    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers()
        await self._collect_tools()
        logger.info(f"Connected to servers with {len(self.tools)} tools available")
    
    async def _collect_tools(self):
        """Collect and convert tools from both MCP servers."""
        self.tools = []
        
        rw_response = await self.rapidwright_session.list_tools()
        for tool in rw_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "rapidwright"))
        
        v_response = await self.vivado_session.list_tools()
        for tool in v_response.tools:
            self.tools.append(convert_mcp_tool_to_openai(tool, "vivado"))
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call on the appropriate MCP server."""
        # Parse server prefix from tool name
        if tool_name.startswith("rapidwright_"):
            session = self.rapidwright_session
            actual_name = tool_name[len("rapidwright_"):]
        elif tool_name.startswith("vivado_"):
            session = self.vivado_session
            actual_name = tool_name[len("vivado_"):]
        else:
            return json.dumps({"error": f"Unknown tool prefix in: {tool_name}"})
        
        # Track timing for this tool call
        start_time = time.time()
        wns_measured = None
        error_occurred = False
        
        try:
            logger.info(f"Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
            result = await session.call_tool(actual_name, arguments)
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                result_text = "\n".join(text_parts)
            else:
                result_text = "(no output)"
            
            # Track WNS from timing reports and get_wns calls
            if tool_name == "vivado_report_timing_summary":
                # If target clock is set, get clock-specific WNS instead of overall
                if self.target_clock:
                    try:
                        clock_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
                        if clock_wns is not None:
                            current_wns = clock_wns
                            wns_measured = current_wns
                            current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                            fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                            if current_wns > self.best_wns:
                                logger.info(f"New best WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                                self.best_wns = current_wns
                            else:
                                logger.info(f"Current WNS (clock: {self.target_clock}): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                    except Exception as e:
                        logger.warning(f"Failed to get clock-specific WNS, falling back to overall: {e}")
                        self.target_clock = None  # Fall through to overall WNS parsing
                
                if not self.target_clock or wns_measured is None:
                    timing_info = parse_timing_summary_static(result_text)
                    if timing_info["wns"] is not None:
                        current_wns = timing_info["wns"]
                        wns_measured = current_wns
                        current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                        fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                        if current_wns > self.best_wns:
                            logger.info(f"New best WNS: {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                            self.best_wns = current_wns
                        else:
                            logger.info(f"Current WNS: {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
            
            # Also track WNS from get_wns tool (returns just the numeric WNS value)
            elif tool_name == "vivado_get_wns":
                try:
                    current_wns = float(result_text.strip())
                    wns_measured = current_wns
                    current_fmax = self.calculate_fmax(current_wns, self.clock_period)
                    fmax_str = f", fmax: {current_fmax:.2f} MHz" if current_fmax is not None else ""
                    if current_wns > self.best_wns:
                        logger.info(f"New best WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (improved from {self.best_wns:.3f} ns)")
                        self.best_wns = current_wns
                    else:
                        logger.info(f"Current WNS (from get_wns): {current_wns:.3f} ns{fmax_str} (best is still {self.best_wns:.3f} ns)")
                except (ValueError, AttributeError):
                    logger.warning(f"Could not parse WNS from get_wns output: {result_text[:100]}")
            
            elapsed_time = time.time() - start_time
            
            # Record tool call details
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": wns_measured,
                "error": False
            })
            
            return result_text
            
        except Exception as e:
            error_occurred = True
            elapsed_time = time.time() - start_time
            
            # Record failed tool call
            self.tool_call_details.append({
                "tool_name": tool_name,
                "iteration": self.iteration,
                "elapsed_time": elapsed_time,
                "wns": None,
                "error": True,
                "error_message": str(e)
            })
            
            logger.error(f"Tool call failed: {e}")
            return json.dumps({"error": str(e)})
    
    async def _call_vivado_tool(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools (for use with base class methods)."""
        return await self.call_tool(f"vivado_{tool_name}", arguments)
    
    async def process_response(self, response) -> tuple[str, bool]:
        """Process LLM response, execute tool calls, return final text and done flag."""
        # Validate response structure with detailed logging
        try:
            if not response:
                raise ValueError("Response is None")
            if not hasattr(response, 'choices'):
                raise ValueError(f"Response has no 'choices' attribute. Response type: {type(response)}, Response: {response}")
            if response.choices is None:
                raise ValueError("Response.choices is None")
            if len(response.choices) == 0:
                raise ValueError("Response choices list is empty")
            
            message = response.choices[0].message
            if not message:
                raise ValueError("Message is None")
        except Exception as e:
            logger.error(f"Failed to parse response structure: {e}")
            logger.error(f"Response object: {response}")
            raise
        
        # Convert message to dict, excluding None values which can cause issues
        message_dict = message.model_dump(exclude_none=True)
        self.messages.append(message_dict)
        
        if self.debug:
            logger.debug(f"Added message to conversation: {json.dumps(message_dict, indent=2)[:500]}...")
        
        # Check for tool calls
        if message.tool_calls:
            tool_results = []
            
            for tool_call in message.tool_calls:
                # Validate tool_call structure
                if not tool_call or not hasattr(tool_call, 'function') or not tool_call.function:
                    logger.warning(f"Invalid tool_call structure: {tool_call}")
                    continue
                
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                except json.JSONDecodeError:
                    tool_args = {}
                
                result = await self.call_tool(tool_name, tool_args)
                
                # Truncate very long results to avoid API issues
                MAX_RESULT_LENGTH = 50000  # characters
                if len(result) > MAX_RESULT_LENGTH:
                    logger.warning(f"Tool result from {tool_name} is {len(result)} chars, truncating to {MAX_RESULT_LENGTH}")
                    result = result[:MAX_RESULT_LENGTH] + f"\n...[truncated {len(result) - MAX_RESULT_LENGTH} characters]"
                
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": result
                })
                
                # Debug logging
                if self.debug:
                    logger.debug(f"Tool {tool_name} result: {result[:500]}...")
            
            # Add tool results to messages
            self.messages.extend(tool_results)
            
            # Continue conversation
            return await self.get_completion()
        
        # No tool calls - check if we're done
        content = message.content or ""
        
        # Check for completion indicators
        is_done = any(phrase in content.lower() for phrase in [
            "optimization complete",
            "timing is met",
            "wns >= 0",
            "no more optimizations",
            "design meets timing",
            "successfully saved",
            "final design saved"
        ])
        
        return content, is_done
    
    async def perform_initial_analysis(self, input_dcp: Path) -> str:
        """
        Perform initial analysis without LLM:
        1. Initialize RapidWright
        2. Open checkpoint in Vivado
        3. Report timing summary
        4. Get critical high fanout nets
        
        Returns a formatted summary of the analysis.
        """
        logger.info("Performing initial design analysis...")
        print("\n=== Initial Design Analysis ===\n")
        
        # Step 1: Initialize RapidWright
        logger.info("Initializing RapidWright...")
        print("Initializing RapidWright...")
        result = await self.call_tool("rapidwright_initialize_rapidwright", {})
        if "error" in result.lower() and "success" not in result.lower():
            raise RuntimeError(f"Failed to initialize RapidWright: {result}")
        print("✓ RapidWright initialized\n")
        
        # Step 2: Open checkpoint in Vivado
        logger.info(f"Opening checkpoint: {input_dcp}")
        print(f"Opening checkpoint: {input_dcp.name}")
        result = await self.call_tool("vivado_open_checkpoint", {
            "dcp_path": str(input_dcp.resolve())
        })
        if "error" in result.lower() and "opened successfully" not in result.lower():
            raise RuntimeError(f"Failed to open checkpoint: {result}")
        print("✓ Checkpoint opened in Vivado\n")
        
        # Step 3: Report timing summary
        logger.info("Analyzing timing...")
        print("Analyzing timing...")
        timing_report = await self.call_tool("vivado_report_timing_summary", {})
        
        # Parse timing
        timing_info = parse_timing_summary_static(timing_report)
        self.initial_tns = timing_info["tns"]
        self.initial_failing_endpoints = timing_info["failing_endpoints"]
        
        # Get clock period for fmax calculation (also detects target clock)
        self.clock_period = await super().get_clock_period(self._call_vivado_tool)
        
        # Get WNS for the target clock domain
        target_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
        if target_wns is not None:
            self.initial_wns = target_wns
        else:
            self.initial_wns = timing_info["wns"]
        self.best_wns = self.initial_wns if self.initial_wns is not None else float('-inf')
        
        clock_info = f" (clock: {self.target_clock})" if self.target_clock else ""
        print(f"✓ Timing analyzed:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            print(f"  - Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.target_clock:
            print(f"  - Target clock: {self.target_clock}")
        if self.initial_wns is not None:
            print(f"  - WNS{clock_info}: {self.initial_wns:.3f} ns")
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                print(f"  - Achievable fmax: {initial_fmax:.2f} MHz")
        if self.initial_tns is not None:
            print(f"  - TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            print(f"  - Failing endpoints: {self.initial_failing_endpoints}")
        print()
        
        # Step 4: Get critical high fanout nets
        logger.info("Identifying critical high fanout nets...")
        print("Identifying critical high fanout nets...")
        nets_report = await self.call_tool("vivado_get_critical_high_fanout_nets", {
            "num_paths": 50,
            "min_fanout": 100
        })
        
        # Parse high fanout nets
        self.high_fanout_nets = self.parse_high_fanout_nets(nets_report)
        print(f"✓ Found {len(self.high_fanout_nets)} high fanout nets (>100 fanout)\n")
        
        # Step 5: Load design in RapidWright for spread analysis
        critical_path_spread_info = None  # Initialize
        
        logger.info("Loading design in RapidWright...")
        print("Loading design in RapidWright for spread analysis...")
        result = await self.call_tool("rapidwright_read_checkpoint", {
            "dcp_path": str(input_dcp.resolve())
        })
        if "error" in result.lower() and "success" not in result.lower():
            print(f"⚠ Warning: Could not load design in RapidWright: {result}")
        else:
            print("✓ Design loaded in RapidWright\n")
            
            # Step 6: Extract critical path cells and analyze spread
            logger.info("Extracting and analyzing critical path spread...")
            print("Analyzing critical path spread...")
            
            # Extract critical path cells from Vivado
            temp_path = Path(self.temp_dir) / "initial_critical_paths.json"
            cells_json = await self.call_tool("vivado_extract_critical_path_cells", {
                "num_paths": 50,
                "output_file": str(temp_path)
            })
            
            # Analyze spread in RapidWright
            spread_result = await self.call_tool("rapidwright_analyze_critical_path_spread", {
                "input_file": str(temp_path)
            })
            
            # Parse spread results
            import json
            try:
                spread_data = json.loads(spread_result)
                critical_path_spread_info = {
                    "max_distance": spread_data.get("max_distance_found", 0),
                    "avg_distance": spread_data.get("avg_max_distance", 0),
                    "paths_analyzed": spread_data.get("paths_analyzed", 0)
                }
                print(f"✓ Critical path spread analyzed:")
                print(f"  - Max distance: {critical_path_spread_info['max_distance']} tiles")
                print(f"  - Avg distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
                print(f"  - Paths analyzed: {critical_path_spread_info['paths_analyzed']}")
                print()
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠ Warning: Could not parse spread results: {e}")
                critical_path_spread_info = None
        
        # Create concise summary for LLM
        summary = []
        summary.append("=== Initial Design Analysis ===\n")
        
        # Timing status
        summary.append("TIMING STATUS:")
        if self.clock_period is not None:
            target_fmax = 1000.0 / self.clock_period
            summary.append(f"  Clock period: {self.clock_period:.3f} ns (target fmax: {target_fmax:.2f} MHz)")
        if self.initial_wns is not None:
            if self.initial_wns >= 0:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING MET ✓")
            else:
                summary.append(f"  WNS: {self.initial_wns:.3f} ns - TIMING VIOLATED")
            # Add fmax information
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                summary.append(f"  Achievable fmax: {initial_fmax:.2f} MHz")
        if self.initial_tns is not None:
            summary.append(f"  TNS: {self.initial_tns:.3f} ns")
        if self.initial_failing_endpoints is not None:
            summary.append(f"  Failing endpoints: {self.initial_failing_endpoints}")
        summary.append("")
        
        # Critical path spread analysis
        self.critical_path_spread_info = critical_path_spread_info
        self.pblock_recommended = bool(
            critical_path_spread_info
            and critical_path_spread_info.get('avg_distance', 0) > 70
            and critical_path_spread_info.get('paths_analyzed', 0) >= 5
        )
        if critical_path_spread_info:
            summary.append("CRITICAL PATH SPREAD ANALYSIS:")
            summary.append(f"  Max cell distance: {critical_path_spread_info['max_distance']} tiles")
            summary.append(f"  Avg cell distance: {critical_path_spread_info['avg_distance']:.1f} tiles")
            summary.append(f"  Paths analyzed: {critical_path_spread_info['paths_analyzed']}")

            # Recommendation based on spread
            if self.pblock_recommended:
                summary.append(f"  ⚠ RECOMMENDATION: Use PBLOCK strategy (high spread detected)")
            summary.append("")
        
        # High fanout nets (show top 10)
        if self.high_fanout_nets:
            summary.append("CRITICAL HIGH FANOUT NETS (top 10):")
            for i, (net_name, fanout, path_count) in enumerate(self.high_fanout_nets[:10]):
                summary.append(f"  {i+1}. {net_name}")
                summary.append(f"     Fanout: {fanout}, Critical paths: {path_count}")
            if len(self.high_fanout_nets) > 10:
                summary.append(f"  ... and {len(self.high_fanout_nets) - 10} more nets")
        else:
            summary.append("CRITICAL HIGH FANOUT NETS: None found")
        
        summary.append("")
        summary.append(f"Total nets available for optimization: {len(self.high_fanout_nets)}")
        
        summary_text = "\n".join(summary)
        print(summary_text)
        print()
        
        return summary_text
    
    async def get_completion(self) -> tuple[str, bool]:
        """Get LLM completion and process it."""
        try:
            self.llm_call_count += 1
            logger.info(f"LLM API call #{self.llm_call_count}")
            
            # Request usage accounting from OpenRouter
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=4096,
                extra_body={
                    "usage": {
                        "include": True
                    }
                }
            )
            
            # Validate response immediately
            if response is None:
                raise ValueError("API returned None response")
            
            # Extract token usage information from OpenRouter
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                
                # Update cumulative totals
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_tokens += total_tokens
                
                # Get actual cost from OpenRouter (in credits/dollars)
                call_cost = 0.0
                if hasattr(response.usage, 'cost') and response.usage.cost is not None:
                    call_cost = float(response.usage.cost)
                    self.total_cost += call_cost
                else:
                    logger.warning("OpenRouter did not provide cost information")
                
                # Extract additional usage details if available
                cached_tokens = 0
                reasoning_tokens = 0
                if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                    if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                        cached_tokens = response.usage.prompt_tokens_details.cached_tokens or 0
                if hasattr(response.usage, 'completion_tokens_details') and response.usage.completion_tokens_details:
                    if hasattr(response.usage.completion_tokens_details, 'reasoning_tokens'):
                        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
                
                # Store details for this call
                call_detail = {
                    "call_number": self.llm_call_count,
                    "iteration": self.iteration,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": call_cost,
                    "cached_tokens": cached_tokens,
                    "reasoning_tokens": reasoning_tokens
                }
                self.api_call_details.append(call_detail)
                
                # Log token usage
                cache_info = f", Cached: {cached_tokens:,}" if cached_tokens > 0 else ""
                reasoning_info = f", Reasoning: {reasoning_tokens:,}" if reasoning_tokens > 0 else ""
                cost_info = f" | Cost: ${call_cost:.4f}" if call_cost > 0 else ""
                
                logger.info(f"API call #{self.llm_call_count} - Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total{cost_info}{cache_info}{reasoning_info}")
                print(f"[API Call #{self.llm_call_count}] Tokens: {total_tokens:,} (Prompt: {prompt_tokens:,}, Completion: {completion_tokens:,}{cache_info}{reasoning_info}){cost_info}")
            else:
                logger.warning("No usage information in API response")
            
            # Debug logging
            if self.debug:
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response: {response}")
            
            # Check if response has error
            if hasattr(response, 'error') and response.error:
                raise ValueError(f"API returned error: {response.error}")
            
            return await self.process_response(response)
            
        except Exception as e:
            logger.error(f"Error in get_completion: {e}")
            logger.error(f"Number of messages in conversation: {len(self.messages)}")
            if self.messages:
                logger.error(f"Last message: {self.messages[-1]}")
            raise

    def _elapsed(self) -> float:
        """Seconds since optimize() started."""
        return time.time() - self.start_time if self.start_time else 0.0

    def _remaining_total(self) -> float:
        """Seconds left in the overall wall-clock budget."""
        return self.total_timeout - self._elapsed()

    def _arm_watchdog(self, output_dcp: Path):
        """Arm the HARD wall-clock kill. Fires once at hard_deadline seconds.

        Soft stage budgets cannot interrupt a running Vivado op, so this is the only
        guaranteed way to finish under the contest's 1-hour limit. On fire it writes
        the protected-best DCP (already equivalence-checked) as the scored output and
        force-exits, so a valid, improved submission always exists under the deadline.
        """
        if self.hard_deadline is None or self.hard_deadline <= 0:
            return

        def _fire():
            try:
                if (self.protected_best_dcp is not None
                        and self.protected_best_dcp.exists()):
                    shutil.copy2(self.protected_best_dcp, output_dcp)
                    os.utime(output_dcp, None)  # newest matching *_optimized*.dcp
                    msg = f"protected best (WNS {self.protected_best_wns:.3f} ns)"
                else:
                    msg = "current output_dcp on disk"
            except Exception as e:
                msg = f"could not refresh output ({e})"
            line = (f"\n⏰ HARD DEADLINE {self.hard_deadline}s reached "
                    f"({self._elapsed():.0f}s elapsed) -- force-stopping under the "
                    f"1-hour contest limit. Scored output = {msg}.\n")
            try:
                logger.error(line.strip())
                print(line, flush=True)
            except Exception:
                pass
            # Kill Vivado / MCP children, then this process. os._exit avoids running
            # finalize (which would need the now-killed Vivado); the output DCP on disk
            # is already the best valid result.
            try:
                os.system("pkill -9 -f VivadoMCP/vivado_mcp_server.py 2>/dev/null; "
                          "pkill -9 -f 'vivado -mode tcl' 2>/dev/null")
            except Exception:
                pass
            os._exit(0)

        # Anchor the fire time to optimize() start (self.start_time), so hard_deadline
        # is measured the same way as _elapsed()/_remaining_total() -- i.e. it includes
        # the initial analysis time. Subtract time already spent so the ABSOLUTE fire
        # point is hard_deadline seconds after optimize() began, not after arming.
        delay = max(1.0, self.hard_deadline - self._elapsed())
        self._watchdog = threading.Timer(delay, _fire)
        self._watchdog.daemon = True
        self._watchdog.start()
        logger.info(f"Hard-deadline watchdog armed: fires in {delay:.0f}s "
                    f"(hard_deadline {self.hard_deadline}s from optimize start; "
                    f"total_timeout {self.total_timeout}s).")

    def _disarm_watchdog(self):
        """Cancel the watchdog after a normal (in-budget) completion."""
        if self._watchdog is not None:
            try:
                self._watchdog.cancel()
            except Exception:
                pass
            self._watchdog = None

    async def run_physopt_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """Deterministic pre-LLM optimization: run phys_opt_design and save a
        guaranteed baseline DCP.

        phys_opt_design only commits changes that improve WNS, keeps the design
        placed-and-routed, and preserves functional equivalence, so the saved
        result is always a safe, valid submission that will pass validate_dcps.py.
        The design must already be open in Vivado (from perform_initial_analysis).
        Returns the post-optimization WNS (or None if it could not be measured)."""
        print("\n=== Deterministic Pre-LLM Optimization: phys_opt_design ===\n")
        wns_before = self.best_wns if self.best_wns > float('-inf') else self.initial_wns

        directive = (self.physopt_directive or "").strip()
        physopt_args = {"directive": directive} if directive else {}
        if timeout:
            physopt_args["timeout"] = int(timeout)
        label = f"directive={directive}" if directive else "default optimizations"
        tstr = f", timeout {timeout}s" if timeout else ""
        logger.info(f"Running phys_opt_design ({label}{tstr})...")
        print(f"Running phys_opt_design ({label}{tstr})... this may take a few minutes")
        result = await self.call_tool("vivado_phys_opt_design", physopt_args)
        if "error" in result.lower() and "complete" not in result.lower():
            print(f"⚠ phys_opt_design reported an issue: {result[:300]}")

        # Re-measure timing (call_tool auto-updates self.best_wns for the target clock)
        print("Re-checking timing after phys_opt_design...")
        await self.call_tool("vivado_report_timing_summary", {})
        wns_after = self.best_wns if self.best_wns > float('-inf') else None

        # Adopt only if the result is at least as good as the protected fallback (it
        # normally is, since phys_opt does not regress setup WNS). This guards against
        # a timed-out/interrupted phys_opt leaving a worse or broken in-memory design.
        if wns_after is not None and wns_after >= self.protected_best_wns:
            print(f"Saving baseline DCP to: {output_dcp}")
            await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()), "force": True})
            if self.protected_best_dcp is None:
                self.protected_best_dcp = self.run_dir / "best_protected.dcp"
            try:
                shutil.copy2(output_dcp, self.protected_best_dcp)
                self.protected_best_wns = wns_after
            except Exception as e:
                logger.warning(f"Could not update protected baseline copy: {e}")
            delta = (wns_after - wns_before) if wns_before is not None else 0.0
            fmax_after = self.calculate_fmax(wns_after, self.clock_period)
            fmax_str = f", fmax: {fmax_after:.2f} MHz" if fmax_after is not None else ""
            print(f"✓ Baseline saved. WNS {wns_before:.3f} → {wns_after:.3f} ns "
                  f"(Δ {delta:+.3f} ns{fmax_str})\n")
            return wns_after

        # phys_opt did not help (or could not be measured / timed out) -> keep fallback.
        cur = f"{wns_after:.3f} ns" if wns_after is not None else "unmeasurable"
        print(f"⚠ phys_opt result ({cur}) not better than fallback "
              f"({self.protected_best_wns:.3f} ns); keeping fallback.")
        if (self.protected_best_dcp is not None and self.protected_best_dcp.exists()
                and (wns_after is None or wns_after < self.protected_best_wns)):
            await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(self.protected_best_dcp.resolve())})
            print("✓ Restored fallback into Vivado.\n")
        return None

    async def run_pinopt_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """CHEAP deterministic optimization: phys_opt_design -critical_pin_opt.

        Runs ONLY the critical-pin-swap slice of phys_opt -- it remaps critical LUT
        input pins to faster pins (and adjusts the LUT INIT accordingly), which is
        function-preserving and keeps the design placed-and-routed incrementally. No
        full place, no full route. Costs ~1-2 min even on large designs (measured ~68s
        on VTR/8k SLICEs), so it fits designs where a full phys_opt is too expensive,
        and it re-optimizes pins on whatever critical path the earlier stages produced.
        Adopts only if it beats the protected baseline. Returns new WNS or None."""
        print("\n=== Deterministic Optimization: critical pin opt (cheap) ===\n")
        wns_before = self.best_wns if self.best_wns > float('-inf') else self.initial_wns
        rt = int(timeout) if timeout else 300
        print(f"Running phys_opt_design -critical_pin_opt (timeout {rt}s)...")
        await self.call_tool("vivado_run_tcl", {
            "command": "phys_opt_design -critical_pin_opt", "timeout": rt})

        await self.call_tool("vivado_report_timing_summary", {})
        wns_after = self.best_wns if self.best_wns > float('-inf') else None

        # Adopt only if strictly better than the protected fallback (pin swaps are
        # function-preserving and shouldn't regress; keep-if-better guards anyway).
        if wns_after is not None and wns_after > self.protected_best_wns:
            await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()), "force": True})
            if self.protected_best_dcp is None:
                self.protected_best_dcp = self.run_dir / "best_protected.dcp"
            try:
                shutil.copy2(output_dcp, self.protected_best_dcp)
                self.protected_best_wns = wns_after
            except Exception as e:
                logger.warning(f"Could not update protected baseline copy: {e}")
            if wns_after > self.best_wns:
                self.best_wns = wns_after
            delta = (wns_after - wns_before) if wns_before is not None else 0.0
            fmax_after = self.calculate_fmax(wns_after, self.clock_period)
            fmax_str = f", fmax: {fmax_after:.2f} MHz" if fmax_after is not None else ""
            print(f"✓ critical_pin_opt improved WNS {wns_before:.3f} → {wns_after:.3f} ns "
                  f"(Δ {delta:+.3f} ns{fmax_str}). Saving as new baseline.\n")
            return wns_after

        cur = f"{wns_after:.3f} ns" if wns_after is not None else "unmeasurable"
        print(f"⚠ critical_pin_opt ({cur}) did not beat baseline "
              f"({self.protected_best_wns:.3f} ns); keeping baseline.")
        if (self.protected_best_dcp is not None and self.protected_best_dcp.exists()
                and (wns_after is None or wns_after < self.protected_best_wns)):
            await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(self.protected_best_dcp.resolve())})
            print("✓ Restored baseline into Vivado.\n")
        return None

    async def run_routeopt_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """CHEAP deterministic finisher: re-route the current best with a higher-effort
        router directive (route_design -directive Explore).

        Routing-only -- no placement move, no netlist edit -> provably functionally
        equivalent (validate-safe). It re-runs the router with more effort on whatever
        placement the earlier stages produced, which can find a shorter-delay routing of
        the same nets. This is exactly the move the LLM finisher stumbled onto on
        logicnets (route_design -directive Explore: WNS -0.417 -> -0.408, +2.46 MHz); we
        bake it in deterministically so every design gets it, not just when the LLM
        happens to try it. Must run AFTER a fully-routed best exists. Adopts only if it
        beats the protected baseline; otherwise the protected baseline is reloaded."""
        print("\n=== Deterministic Optimization: best-of-N explore-route polish ===\n")
        if getattr(self, "routeopt_mode", "always") == "never":
            print("routeopt disabled (--routeopt-mode never). Skipping.\n")
            return None
        if self.protected_best_dcp is None or not self.protected_best_dcp.exists():
            print("No protected baseline to polish; skipping routeopt.\n")
            return None
        # route_design -directive Explore is HIGH-VARIANCE: on logicnets one roll gives
        # WNS -0.417, another -0.408 (+2.46 MHz). A single route often misses the good
        # roll (this was the old LLM finisher's real edge -- it ran Explore repeatedly and
        # kept the best). So re-route the SAME placement N times (unroute -> Explore) and
        # keep the best. Routing-only -> functionally equivalent. N scales DOWN with design
        # size (a route is cheap on small designs, expensive on big ones) -- same tiers as
        # the pblock sweep.
        try:
            occ_txt = await self.call_tool("vivado_run_tcl", {
                "command": r'puts "ROCC [llength [get_sites -quiet -filter {SITE_TYPE =~ SLICE* && IS_USED} SLICE_*]]"',
                "timeout": 60})
            occ = int(re.search(r"ROCC (\d+)", occ_txt).group(1))
        except Exception:
            occ = 99999
        n_rolls = 3 if occ <= 6000 else (2 if occ <= 12000 else 1)
        wns_before = self.protected_best_wns
        rt = max(300, int((timeout or 1200) / n_rolls))
        best_wns = None
        best_dcp = Path(self.temp_dir) / "routeopt_best.dcp"
        print(f"best-of-{n_rolls} explore-route (occ={occ}, ≤{rt}s/route)...")

        for i in range(n_rolls):
            if self._remaining_total() < rt + 60:
                print(f"⏱ Out of budget; stopping routeopt at roll {i}.\n"); break
            # INCREMENTAL re-route (route -Explore ON the already-routed baseline) -- this is
            # the high-variance pass that reaches the good roll (-0.408 / 524.11 on logicnets).
            # A from-scratch unroute+route is deterministic and MISSES it. Reload the routed
            # baseline before each roll so every roll re-routes the SAME -0.417 starting point.
            if i > 0:
                await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(self.protected_best_dcp.resolve())})
            await self.call_tool("vivado_route_design", {"directive": "Explore", "timeout": rt})
            rs = await self.call_tool("vivado_report_route_status", {})
            routed = ("fully routed" in rs.lower()) or ("100.00%" in rs)
            await self.call_tool("vivado_report_timing_summary", {})
            w = self.best_wns if (routed and self.best_wns > float('-inf')) else None
            fx = self.calculate_fmax(w, self.clock_period) if w is not None else None
            print(f"  [roll {i+1}/{n_rolls}] " + (f"WNS {w:.3f} ns"
                  + (f", {fx:.2f} MHz" if fx is not None else "") if w is not None
                  else "not fully routed -- discarded"))
            if w is not None and (best_wns is None or w > best_wns):
                best_wns = w
                await self.call_tool("vivado_write_checkpoint", {
                    "dcp_path": str(best_dcp.resolve()), "force": True})

        # Adopt the best roll iff it beats the incoming baseline; else restore baseline.
        if best_wns is not None and best_wns > wns_before and best_dcp.exists():
            try:
                shutil.copy2(best_dcp, output_dcp)
                shutil.copy2(best_dcp, self.protected_best_dcp)
            except Exception as e:
                logger.warning(f"routeopt: could not save best roll ({e})")
            self.protected_best_wns = best_wns
            if best_wns > self.best_wns:
                self.best_wns = best_wns
            await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(output_dcp.resolve())})
            fx = self.calculate_fmax(best_wns, self.clock_period)
            fstr = f", fmax: {fx:.2f} MHz" if fx is not None else ""
            print(f"✓ best-of-{n_rolls} explore-route: WNS {wns_before:.3f} → {best_wns:.3f} ns "
                  f"(Δ {best_wns - wns_before:+.3f} ns{fstr}). Saving as new baseline.\n")
            return best_wns

        cur = f"{best_wns:.3f} ns" if best_wns is not None else "no clean roll"
        print(f"⚠ best-of-N explore-route ({cur}) did not beat baseline "
              f"({wns_before:.3f} ns); reverting to the protected baseline.")
        await self.call_tool("vivado_open_checkpoint", {
            "dcp_path": str(self.protected_best_dcp.resolve())})
        print("✓ Restored baseline into Vivado.\n")
        return None

    @staticmethod
    def _parse_pblock_targets(util_text: str) -> Optional[dict]:
        """Parse the '1.5x Multiplier' section of report_utilization_for_pblock."""
        section = util_text.split("1.5x Multiplier")[-1] if "1.5x Multiplier" in util_text else util_text
        targets = {}
        for key in ("LUT", "FF", "DSP", "BRAM", "URAM"):
            m = re.search(rf"{key}s?:\s*([\d,]+)", section)
            if m:
                targets[key] = int(m.group(1).replace(",", ""))
        return targets or None

    @staticmethod
    def _parse_json_field(text: str, field: str):
        """Extract a field from a JSON tool result; None on error/invalid."""
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or data.get("status") == "error" or "error" in data:
            return None
        return data.get(field)

    async def run_pblock_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """Deterministic pblock re-placement: consolidate a spread-out design into a
        contiguous fabric region to cut wire delay, then re-place and re-route.

        Functionally equivalent (same netlist; only placement/routing change). This
        DESTRUCTIVELY re-places the design, so the result is adopted ONLY if it routes
        cleanly AND beats the protected best; otherwise the protected baseline (e.g. the
        phys_opt result) is reloaded into Vivado and kept. Returns the new WNS if adopted."""
        print("\n=== Deterministic Pre-LLM Optimization: pblock re-placement ===\n")

        # Gate: pblock helps spread-out designs; it is slow/pointless on compact ones.
        if self.pblock_mode == "never":
            print("pblock step disabled (--pblock-mode never). Skipping.\n")
            return None
        if self.pblock_mode == "auto" and not self.pblock_recommended:
            print("pblock not recommended by spread analysis (design is compact); skipping. "
                  "Use --pblock-mode always to force.\n")
            return None

        # 1-3. Derive a compact confinement box directly from the design's SLICE
        # footprint. The proven geometry -- occupied_slices x 1.5, tall-narrow
        # (H/W ~= 3.4), centered on the die -- captures the global route-length win
        # (validated +118 MHz on logicnets: 403->521 MHz). This replaces the
        # RapidWright fabric analysis, which under-sized the region ~8x on this
        # UltraScale+ part (picked ~740 SLICE sites for a 5759-SLICE design).
        # DSP/BRAM are left unconstrained on purpose: Vivado re-places them next to
        # the confined logic, so a pure-SLICE box still works on DSP designs, and
        # the keep-if-better + revert gate below protects designs where confine loses.
        HEADROOM = 1.5
        ASPECT = 3.4  # H/W; a tall-narrow box beat a square one by ~70 MHz in tuning
        stats_tcl = (
            r'set occ [llength [get_sites -quiet -filter {SITE_TYPE =~ SLICE* && IS_USED} SLICE_*]]; '
            r'set ndsp [llength [get_sites -quiet -filter {IS_USED} DSP*]]; '
            r'set nbram [llength [get_sites -quiet -filter {IS_USED} RAMB*]]; '
            r'set xmin 1000000; set xmax -1; set ymin 1000000; set ymax -1; '
            r'foreach s [get_sites -quiet SLICE_*] { regexp {SLICE_X(\d+)Y(\d+)} $s -> x y; '
            r'if {$x<$xmin} {set xmin $x}; if {$x>$xmax} {set xmax $x}; '
            r'if {$y<$ymin} {set ymin $y}; if {$y>$ymax} {set ymax $y} }; '
            r'puts "CONFINE_STATS occ $occ dsp $ndsp bram $nbram xmin $xmin xmax $xmax ymin $ymin ymax $ymax"'
        )
        stats = await self.call_tool("vivado_run_tcl", {"command": stats_tcl})
        m = re.search(r"CONFINE_STATS occ (\d+) dsp (\d+) bram (\d+) xmin (\d+) xmax (\d+) ymin (\d+) ymax (\d+)", stats)
        if not m:
            print(f"⚠ Could not compute SLICE footprint for confine box; skipping pblock.\n  {stats[:300]}\n")
            return None
        occ, ndsp, nbram, xmin, xmax, ymin, ymax = (int(g) for g in m.groups())
        if occ <= 0:
            print("⚠ No occupied SLICEs found; skipping pblock.\n")
            return None

        # Gate: global confinement WINS on logic-heavy designs (logicnets +118,
        # vexriscv +137) but TRAPS DSP-dominated datapath designs in a worse basin
        # (amd: confine is kept by the greedy per-stage WNS gate, yet the incremental
        # stages reach a better result WITHOUT it -> net -7.6 MHz). DSP/BRAM density
        # is the clean empirical discriminator: amd (40 DSP + 14 BRAM)/688 = 0.078,
        # rosetta 107/6221 = 0.017, logicnets/vexriscv ~0. Skip confine when the
        # design is DSP-dominated. This is a HARD safety gate: it applies in BOTH
        # 'auto' and 'always' modes (only 'never' skips earlier, before stats). The
        # default pblock_mode is 'always', which is meant to override the conservative
        # spread heuristic -- NOT this safety check, since forcing confine on a DSP
        # design is a known net loss (amd: +104.7 without -> +97.08 with).
        DSP_DENSITY_LIMIT = 0.04
        dsp_density = (ndsp + nbram) / occ
        if dsp_density > DSP_DENSITY_LIMIT:
            print(f"pblock (confine) skipped: DSP/BRAM-dominated design "
                  f"({ndsp} DSP + {nbram} BRAM over {occ} SLICEs = density {dsp_density:.3f} "
                  f"> {DSP_DENSITY_LIMIT}). Global confinement traps datapath designs; the "
                  f"incremental stages do better here.\n")
            return None
        # 4. SWEEP the confine box ratio. The box geometry is high-leverage and the best
        # headroom varies by design (logicnets got a big improvement at one ratio). On
        # SMALL designs a full place+route is cheap, so try a few headrooms and keep the
        # best; on large designs a single box (sweep would be too slow). 1.5 goes FIRST
        # (the proven default) so the primary path is always exercised. Each candidate
        # restarts from the SAME pre-pblock baseline (reloaded from the protected DCP),
        # so ratios are compared fairly. Adopt the overall best iff it beats the baseline;
        # otherwise revert -- the keep-if-better + protected-best guarantee is preserved.
        # SIZE-GRADUATED ratio count: each ratio is a full confined place+route whose cost
        # scales with design size, so the runtime we spend sweeping must scale DOWN as the
        # design grows. Big designs -> a single box (1.5x proven default); the sweep runtime
        # would otherwise dwarf any ratio gain. Small designs -> full 3-ratio sweep (cheap).
        # (Sweeping needs a protected baseline to restore between ratios; without one, 1 box.)
        have_baseline = (self.protected_best_dcp is not None and self.protected_best_dcp.exists())
        if not have_baseline or occ > 12000:
            headrooms = [HEADROOM]            # BIG (or no baseline): single box, no sweep
        elif occ > 6000:
            headrooms = [1.5, 1.3]            # MEDIUM: 2-ratio sweep
        else:
            headrooms = [1.5, 1.3, 1.8]       # SMALL: full 3-ratio sweep
        # Per-op timeout: confined place_design is SLOWER than unconstrained (the placer has
        # less freedom), so a too-tight cap makes it soft-timeout and churn. Floor it at 420s
        # so each place+route completes cleanly; the per-ratio budget guard below still bounds
        # the total, and the hard-deadline watchdog is the ultimate cap.
        pr_timeout = max(420, int(timeout / (2 * len(headrooms)))) if timeout else 480
        pre_pblock_wns = self.protected_best_wns
        best_sweep_wns = None
        best_sweep_dcp = Path(self.temp_dir) / "pblock_best.dcp"
        _tier = "single box (big design)" if len(headrooms) == 1 else f"{len(headrooms)}-ratio sweep"
        print(f"pblock: {_tier} (occ={occ}, headrooms {headrooms}), aspect {ASPECT}\n")

        for i, hr in enumerate(headrooms):
            if self._remaining_total() < 2 * pr_timeout + 90:
                print(f"⏱ Not enough budget for a clean place+route ({self._remaining_total():.0f}s "
                      f"left, need ≥{2*pr_timeout+90}s); stopping pblock sweep at ratio {i}.\n"); break
            if i > 0:  # restore the pre-pblock baseline before trying the next ratio
                await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(self.protected_best_dcp.resolve())})
            W = max(1, int(round((occ * hr / ASPECT) ** 0.5)))
            H = max(1, int(round(ASPECT * W)))
            xc = (xmin + xmax) // 2; yc = (ymin + ymax) // 2
            x0 = max(xmin, xc - W // 2); x1 = min(xmax, x0 + W - 1)
            y0 = max(ymin, yc - H // 2); y1 = min(ymax, y0 + H - 1)
            pranges = f"SLICE_X{x0}Y{y0}:SLICE_X{x1}Y{y1}"
            print(f"[sweep {i+1}/{len(headrooms)}] {hr}x box -> {pranges} (W {x1-x0+1} H {y1-y0+1}); "
                  f"unplace -> confine -> place -> route (each ≤{pr_timeout}s)...")
            await self.call_tool("vivado_run_tcl", {
                "command": "if {[llength [get_pblocks -quiet pblock_opt]]} "
                           "{delete_pblocks [get_pblocks pblock_opt]}; place_design -unplace"})
            await self.call_tool("vivado_create_and_apply_pblock", {
                "pblock_name": "pblock_opt", "ranges": pranges,
                "apply_to": "current_design", "is_soft": False})
            await self.call_tool("vivado_place_design", {"directive": "Default", "timeout": pr_timeout})
            await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": pr_timeout})
            rs = await self.call_tool("vivado_report_route_status", {})
            routed = ("fully routed" in rs.lower()) or ("100.00%" in rs)
            await self.call_tool("vivado_report_timing_summary", {})
            try:
                w = await super().get_wns_for_target_clock(self._call_vivado_tool)
            except Exception:
                w = None
            if routed and w is not None:
                fx = self.calculate_fmax(w, self.clock_period)
                print(f"   -> WNS {w:.3f} ns" + (f", fmax {fx:.2f} MHz" if fx is not None else ""))
                if best_sweep_wns is None or w > best_sweep_wns:
                    best_sweep_wns = w
                    await self.call_tool("vivado_write_checkpoint", {
                        "dcp_path": str(best_sweep_dcp.resolve()), "force": True})
            else:
                print(f"   -> {'not fully routed' if not routed else 'WNS unmeasurable'}; discarding this ratio")

        # 5. Adopt the best sweep result iff it beats the pre-pblock baseline.
        if best_sweep_wns is not None and best_sweep_wns > pre_pblock_wns and best_sweep_dcp.exists():
            fmax = self.calculate_fmax(best_sweep_wns, self.clock_period)
            fmax_str = f", fmax: {fmax:.2f} MHz" if fmax is not None else ""
            print(f"✓ pblock best (of {len(headrooms)} ratio(s)) WNS {best_sweep_wns:.3f} ns "
                  f"(was {pre_pblock_wns:.3f} ns{fmax_str}). Saving as new baseline.\n")
            try:
                shutil.copy2(best_sweep_dcp, output_dcp)
                if self.protected_best_dcp is not None:
                    shutil.copy2(best_sweep_dcp, self.protected_best_dcp)
            except Exception as e:
                logger.warning(f"pblock: could not save best sweep DCP ({e})")
            self.protected_best_wns = best_sweep_wns
            if best_sweep_wns > self.best_wns:
                self.best_wns = best_sweep_wns
            # reload the adopted best into Vivado for the next stage
            await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(output_dcp.resolve())})
            return best_sweep_wns

        print(f"⚠ pblock sweep did not beat baseline ({pre_pblock_wns:.3f} ns); reverting.")
        if self.protected_best_dcp is not None and self.protected_best_dcp.exists():
            await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(self.protected_best_dcp.resolve())})
            print("✓ Restored protected baseline into Vivado.\n")
        return None

    # Fixed-column hard primitives that anchor a critical path (cannot be freely relocated).
    _HARD_ANCHOR_RE = "DSP|RAMB|URAM|FIFO"

    async def run_relocate_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """Deterministic targeted relocation for a route-bound critical path.

        When the worst path runs between a FIXED hard macro (DSP/BRAM/URAM — which can
        only sit in specific fabric columns) and a movable register bank that is placed
        far away, the recoverable delay is the long route between them. Since the macro
        can't move, this pulls the movable side (the register bus + its feeder LUTs)
        next to the anchor via a small pblock, then re-places ONLY those cells and
        re-routes. Purely placement (no netlist edits) -> functionally equivalent
        (validate-safe).

        Unlike run_pblock_baseline (which re-places the WHOLE design and usually fails to
        route), this touches only the ~tens of cells on the critical path, so it is fast
        and low-risk. It self-gates (skips logic-bound paths with no far hard anchor) and
        self-reverts: adopted only if it routes cleanly AND beats the protected best;
        otherwise the protected baseline is reloaded. Returns the new WNS if adopted.

        Proven on amd_mini-isp: a DSP multiply feeding a distant output register bank,
        WNS -1.094 -> -0.850 (Fmax 375 -> 413 MHz), passing structural + simulation."""
        print("\n=== Deterministic Pre-LLM Optimization: targeted relocation ===\n")
        if getattr(self, "relocate_mode", "always") == "never":
            print("relocation disabled (--relocate-mode never). Skipping.\n")
            return None
        if self.protected_best_dcp is None or not self.protected_best_dcp.exists():
            print("No protected baseline to operate on; skipping relocation.\n")
            return None

        # Analyze the worst path, pick anchor + movable bank, pblock them together, and
        # unplace the bank -- all in one Vivado pass. The planning Tcl WRITES its result
        # marker to a file (robust: run_tcl's pexpect stdout capture is unreliable for
        # multi-line procs), which we then read. Side effects (pblock/unplace) persist in
        # the open design.
        plan_tcl = Path(self.temp_dir) / "relocate_plan.tcl"
        result_file = Path(self.temp_dir) / "relocate_result.txt"
        if result_file.exists():
            try:
                result_file.unlink()
            except Exception:
                pass
        tcl_body = r'''
proc _sitetype {cell} { set s [get_sites -quiet -of_objects $cell]; if {$s eq ""} {return ""}; return [get_property SITE_TYPE $s] }
proc _tile {cell} { return [get_tiles -quiet -of_objects [get_sites -quiet -of_objects $cell]] }
# Compact SLICE pblock range holding ~ncells, made of the slices CLOSEST to (acol,arow)
# in clock region cr. Hugging the target minimizes wire length (recovers more delay
# than a loose box). Returns "" if too few slices.
proc _compact_range {cr acol arow ncells} {
    # Capacity: ~ncells cells, FFs pack 16/slice; x4 headroom for occupied slices.
    set need [expr {int(ceil($ncells/8.0))*4}]
    if {$need < 16} { set need 16 }
    # Collect this clock region's slices with tile coords + SLICE X/Y indices.
    set slices {}
    foreach s [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ SLICE*}] {
        set t [get_tiles -of_objects $s]
        regexp {SLICE_X(\d+)Y(\d+)} $s -> xx yy
        lappend slices [list [get_property COLUMN $t] [get_property ROW $t] $xx $yy]
    }
    if {[llength $slices] < 4} { return "" }
    # Rank SLICE columns by tile-col distance to the target; keep the 2 nearest.
    # A 2-column band is the empirical sweet spot: 1 col over-congests, wide boxes let
    # cells drift from the anchor -- both cost delay.
    set coldist [dict create]
    foreach e $slices { set x [lindex $e 2]; set d [expr {abs([lindex $e 0]-$acol)}]
        if {![dict exists $coldist $x] || $d < [dict get $coldist $x]} { dict set coldist $x $d } }
    set collist {}
    dict for {x d} $coldist { lappend collist [list $d $x] }
    set keepx {}
    foreach pair [lrange [lsort -integer -index 0 $collist] 0 1] { lappend keepx [lindex $pair 1] }
    # Within the kept columns, take the rows nearest the target row, up to capacity.
    set inband {}
    foreach e $slices {
        if {[lsearch $keepx [lindex $e 2]] >= 0} { lappend inband [list [expr {abs([lindex $e 1]-$arow)}] [lindex $e 2] [lindex $e 3]] }
    }
    set take [lrange [lsort -integer -index 0 $inband] 0 [expr {$need-1}]]
    if {[llength $take] < 4} { return "" }
    set xs {}; set ys {}
    foreach e $take { lappend xs [lindex $e 1]; lappend ys [lindex $e 2] }
    set xmin [lindex [lsort -integer $xs] 0]; set xmax [lindex [lsort -integer $xs] end]
    set ymin [lindex [lsort -integer $ys] 0]; set ymax [lindex [lsort -integer $ys] end]
    return "SLICE_X${xmin}Y${ymin}:SLICE_X${xmax}Y${ymax}"
}
proc _apply {targets range label slack} {
    catch {delete_pblocks pb_relocate}
    create_pblock pb_relocate
    add_cells_to_pblock pb_relocate $targets
    resize_pblock pb_relocate -add $range
    # Clear fixed flags and unroute the targets' nets first, so unplace_cell cannot fail
    # with "Cannot unplace / routing contention" on an already-routed design (this bites
    # 2nd+ iterations). All placement/routing-only -> functionally equivalent.
    catch {set_property IS_BEL_FIXED 0 $targets}
    catch {set_property IS_LOC_FIXED 0 $targets}
    set tnets [get_nets -quiet -of_objects $targets]
    if {$tnets ne ""} { catch {route_design -unroute -nets $tnets} }
    unplace_cell $targets
    return "PLAN ok $label ncells=[llength $targets] range=$range slack=$slack"
}
# Case A: worst path runs between a FIXED hard macro and a distant register bank.
# Pull the bank (bus + feeder LUTs) tight against the anchor.
proc _plan_anchor {anchor mov hard_re slack} {
    set at [_tile $anchor]; set mt [_tile $mov]
    set dist [expr {abs([get_property COLUMN $at]-[get_property COLUMN $mt])+abs([get_property ROW $at]-[get_property ROW $mt])}]
    if {$dist < 6} { return "SKIP anchor_already_close_dist_$dist" }
    set base [get_property NAME $mov]; regsub {\[\d+\]$} $base {} base
    set pat $base ; append pat {[*]}
    set ffs [get_cells -quiet $pat]
    if {[llength $ffs] == 0} { set ffs $mov }
    set inpins [get_pins -quiet -of_objects $ffs -filter {DIRECTION==IN && REF_PIN_NAME!=C}]
    set drv [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects [get_nets -quiet -of_objects $inpins] -filter {DIRECTION==OUT}]]
    set luts {}
    foreach c $drv { set st [_sitetype $c]; if {[regexp {SLICE} $st] && ![regexp $hard_re $st]} { lappend luts [get_property NAME $c] } }
    set targets [get_cells -quiet [lsort -unique [concat [get_property NAME $ffs] $luts]]]
    if {[llength $targets] == 0} { return "SKIP no_targets" }
    if {[llength $targets] > 400} { return "SKIP too_many_cells_[llength $targets]" }
    set cr [get_clock_regions -of_objects [get_sites -of_objects $anchor]]
    set range [_compact_range $cr [get_property COLUMN $at] [get_property ROW $at] [llength $targets]]
    if {$range eq ""} { return "SKIP no_slices_near_anchor" }
    return [_apply $targets $range "anchor=[get_property NAME $anchor] dist=$dist" $slack]
}
# Case B: no hard anchor (FF -> logic cone -> FF). The top paths share one spread-out
# cone; consolidate ALL its SLICE cells into a compact region around their centroid.
proc _plan_cone {hard_re slack} {
    set paths [get_timing_paths -max_paths 12 -nworst 3 -delay_type max]
    set names {}
    foreach p $paths {
        foreach c [get_cells -quiet -of_objects [get_pins -quiet -of_objects $p]] {
            set st [_sitetype $c]
            if {[regexp {SLICE} $st] && ![regexp $hard_re $st]} { lappend names [get_property NAME $c] }
        }
    }
    set targets [get_cells -quiet [lsort -unique $names]]
    if {[llength $targets] < 4} { return "SKIP cone_too_small_[llength $targets]" }
    if {[llength $targets] > 600} { return "SKIP cone_too_large_[llength $targets]" }
    set sc 0; set sr 0; set n 0; set cols {}; set rows {}; set crcount [dict create]
    foreach c $targets {
        set t [_tile $c]
        if {$t ne ""} {
            set cc [get_property COLUMN $t]; set rr [get_property ROW $t]
            set sc [expr {$sc+$cc}]; set sr [expr {$sr+$rr}]; incr n
            lappend cols $cc; lappend rows $rr
            set creg [get_clock_regions -quiet -of_objects [get_sites -quiet -of_objects $c]]
            if {$creg ne ""} { dict incr crcount $creg }
        }
    }
    if {$n == 0} { return "SKIP cone_unplaced" }
    set spread [expr {([lindex [lsort -integer $cols] end]-[lindex [lsort -integer $cols] 0])+([lindex [lsort -integer $rows] end]-[lindex [lsort -integer $rows] 0])}]
    if {$spread < 12} { return "SKIP cone_already_compact_spread_$spread" }
    set cr ""; set best 0
    dict for {k v} $crcount { if {$v > $best} { set best $v; set cr $k } }
    if {$cr eq ""} { return "SKIP cone_no_region" }
    set range [_compact_range $cr [expr {$sc/$n}] [expr {$sr/$n}] [llength $targets]]
    if {$range eq ""} { return "SKIP no_slices_for_cone" }
    return [_apply $targets $range "cone spread=$spread region=$cr" $slack]
}
proc do_relocate_plan {hard_re} {
    set p [lindex [get_timing_paths -max_paths 1 -nworst 1 -delay_type max] 0]
    if {$p eq ""} { return "SKIP no_path" }
    set slack [get_property SLACK $p]
    if {$slack >= 0} { return "SKIP timing_met" }
    set scell [get_cells -quiet -of_objects [get_pins -quiet [get_property STARTPOINT_PIN $p]]]
    set ecell [get_cells -quiet -of_objects [get_pins -quiet [get_property ENDPOINT_PIN $p]]]
    if {$scell eq "" || $ecell eq ""} { return "SKIP no_endpoints" }
    set stype [_sitetype $scell]; set etype [_sitetype $ecell]
    if {[regexp $hard_re $stype] && ![regexp $hard_re $etype]} {
        return [_plan_anchor $scell $ecell $hard_re $slack]
    } elseif {[regexp $hard_re $etype] && ![regexp $hard_re $stype]} {
        return [_plan_anchor $ecell $scell $hard_re $slack]
    } else {
        return [_plan_cone $hard_re $slack]
    }
}
if {[catch {do_relocate_plan {%HARD%}} r]} { set r "SKIP tcl_error:$r" }
set fh [open {%RESULT%} w]; puts $fh $r; close $fh
'''.replace("%HARD%", self._HARD_ANCHOR_RE).replace("%RESULT%", str(result_file.resolve()))
        plan_tcl.write_text(tcl_body)

        await self.call_tool("vivado_run_tcl", {"command": f"source {{{plan_tcl.resolve()}}}"})
        detail = result_file.read_text().strip() if result_file.exists() else ""
        if not detail.startswith("PLAN"):
            print(f"Relocation not applicable ({detail or 'no result returned'}); skipping.\n")
            return None
        print(f"Relocation plan: {detail}")

        # Re-place only the relocated cells (rest stays fixed), then re-route.
        pr_timeout = int(timeout / 2) if timeout else 3600
        print(f"Re-placing relocated cells (timeout {pr_timeout}s)...")
        await self.call_tool("vivado_place_design", {"directive": "Default", "timeout": pr_timeout})
        print(f"Re-routing (timeout {pr_timeout}s)...")
        await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": pr_timeout})
        # Drop the temporary pblock so it does not persist in the output constraints.
        await self.call_tool("vivado_run_tcl", {"command": "catch {delete_pblocks pb_relocate}"})

        # Check route status + timing
        route_status = await self.call_tool("vivado_report_route_status", {})
        fully_routed = ("fully routed" in route_status.lower()) or ("100.00%" in route_status)
        await self.call_tool("vivado_report_timing_summary", {})
        try:
            wns_after = await super().get_wns_for_target_clock(self._call_vivado_tool)
        except Exception:
            wns_after = None
        if wns_after is None and self.best_wns > float('-inf'):
            wns_after = self.best_wns

        # Adopt only if routed AND improved; otherwise restore the protected baseline.
        if fully_routed and wns_after is not None and wns_after > self.protected_best_wns:
            fmax = self.calculate_fmax(wns_after, self.clock_period)
            fmax_str = f", fmax: {fmax:.2f} MHz" if fmax is not None else ""
            print(f"✓ relocation improved WNS to {wns_after:.3f} ns (was {self.protected_best_wns:.3f} ns"
                  f"{fmax_str}). Saving as new baseline.\n")
            await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()), "force": True})
            if self.protected_best_dcp is not None:
                try:
                    shutil.copy2(output_dcp, self.protected_best_dcp)
                except Exception as e:
                    logger.warning(f"Could not update protected baseline copy: {e}")
            self.protected_best_wns = wns_after
            if wns_after > self.best_wns:
                self.best_wns = wns_after
            return wns_after

        if not fully_routed:
            reason = "did not route cleanly"
        elif wns_after is not None:
            reason = f"WNS {wns_after:.3f} ns did not beat baseline {self.protected_best_wns:.3f} ns"
        else:
            reason = "WNS could not be measured"
        print(f"⚠ relocation {reason}; reverting to the protected baseline.")
        if self.protected_best_dcp is not None and self.protected_best_dcp.exists():
            await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(self.protected_best_dcp.resolve())})
            print("✓ Restored protected baseline into Vivado.\n")
        return None

    async def run_relocate_rw_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """Group relocation using RapidWright for the UNPLACE step (iteration-friendly).

        Same critical-cone / hard-anchor selection as run_relocate_baseline, but the
        cells are unplaced+unrouted in RapidWright (fullyUnplaceCell + Net.unroute), which
        -- unlike Vivado's unplace_cell -- does NOT fail with "routing contention at pips"
        on an already-routed design. Vivado then re-places the group under a pblock and
        routes. Because the RapidWright unplace never contends, this stage can be listed
        multiple times in pre_opt to iteratively consolidate successive critical cones.
        Placement/routing-only -> functionally equivalent; adopt only if routed + improved."""
        print("\n=== Deterministic Pre-LLM Optimization: RapidWright group relocation ===\n")
        if getattr(self, "relocate_mode", "always") == "never":
            print("relocation disabled; skipping RW relocation.\n")
            return None
        if self.protected_best_dcp is None or not self.protected_best_dcp.exists():
            print("No protected baseline; skipping RW relocation.\n")
            return None

        targets_file = Path(self.temp_dir) / "rw_targets.txt"
        result_file = Path(self.temp_dir) / "rw_result.txt"
        plan_tcl = Path(self.temp_dir) / "rw_plan.tcl"
        for f in (targets_file, result_file):
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass

        # Same selection procs as run_relocate_baseline, but the final step writes the
        # target cell names to a file instead of doing the pblock/unplace in Vivado.
        tcl_body = r'''
proc _sitetype {cell} { set s [get_sites -quiet -of_objects $cell]; if {$s eq ""} {return ""}; return [get_property SITE_TYPE $s] }
proc _tile {cell} { return [get_tiles -quiet -of_objects [get_sites -quiet -of_objects $cell]] }
proc _compact_range {cr acol arow ncells} {
    set need [expr {int(ceil($ncells/8.0))*4}]
    if {$need < 16} { set need 16 }
    set slices {}
    foreach s [get_sites -quiet -of_objects $cr -filter {SITE_TYPE =~ SLICE*}] {
        set t [get_tiles -of_objects $s]
        regexp {SLICE_X(\d+)Y(\d+)} $s -> xx yy
        lappend slices [list [get_property COLUMN $t] [get_property ROW $t] $xx $yy]
    }
    if {[llength $slices] < 4} { return "" }
    set coldist [dict create]
    foreach e $slices { set x [lindex $e 2]; set d [expr {abs([lindex $e 0]-$acol)}]
        if {![dict exists $coldist $x] || $d < [dict get $coldist $x]} { dict set coldist $x $d } }
    set collist {}
    dict for {x d} $coldist { lappend collist [list $d $x] }
    set keepx {}
    foreach pair [lrange [lsort -integer -index 0 $collist] 0 1] { lappend keepx [lindex $pair 1] }
    set inband {}
    foreach e $slices {
        if {[lsearch $keepx [lindex $e 2]] >= 0} { lappend inband [list [expr {abs([lindex $e 1]-$arow)}] [lindex $e 2] [lindex $e 3]] }
    }
    set take [lrange [lsort -integer -index 0 $inband] 0 [expr {$need-1}]]
    if {[llength $take] < 4} { return "" }
    set xs {}; set ys {}
    foreach e $take { lappend xs [lindex $e 1]; lappend ys [lindex $e 2] }
    set xmin [lindex [lsort -integer $xs] 0]; set xmax [lindex [lsort -integer $xs] end]
    set ymin [lindex [lsort -integer $ys] 0]; set ymax [lindex [lsort -integer $ys] end]
    return "SLICE_X${xmin}Y${ymin}:SLICE_X${xmax}Y${ymax}"
}
proc _write_targets {targets} {
    global TFILE
    set fh [open $TFILE w]
    foreach c $targets { puts $fh [get_property NAME $c] }
    close $fh
}
proc _plan_anchor {anchor mov hard_re slack} {
    set at [_tile $anchor]; set mt [_tile $mov]
    set dist [expr {abs([get_property COLUMN $at]-[get_property COLUMN $mt])+abs([get_property ROW $at]-[get_property ROW $mt])}]
    if {$dist < 6} { return "SKIP anchor_already_close_dist_$dist" }
    set base [get_property NAME $mov]; regsub {\[\d+\]$} $base {} base
    set pat $base ; append pat {[*]}
    set ffs [get_cells -quiet $pat]
    if {[llength $ffs] == 0} { set ffs $mov }
    set inpins [get_pins -quiet -of_objects $ffs -filter {DIRECTION==IN && REF_PIN_NAME!=C}]
    set drv [get_cells -quiet -of_objects [get_pins -quiet -leaf -of_objects [get_nets -quiet -of_objects $inpins] -filter {DIRECTION==OUT}]]
    set luts {}
    foreach c $drv { set st [_sitetype $c]; if {[regexp {SLICE} $st] && ![regexp $hard_re $st]} { lappend luts [get_property NAME $c] } }
    set targets [get_cells -quiet [lsort -unique [concat [get_property NAME $ffs] $luts]]]
    if {[llength $targets] == 0} { return "SKIP no_targets" }
    if {[llength $targets] > 400} { return "SKIP too_many_cells_[llength $targets]" }
    set cr [get_clock_regions -of_objects [get_sites -of_objects $anchor]]
    set range [_compact_range $cr [get_property COLUMN $at] [get_property ROW $at] [llength $targets]]
    if {$range eq ""} { return "SKIP no_slices_near_anchor" }
    _write_targets $targets
    return "PLAN ok anchor=[get_property NAME $anchor] dist=$dist ncells=[llength $targets] range=$range slack=$slack"
}
proc _plan_cone {hard_re slack} {
    set paths [get_timing_paths -max_paths 12 -nworst 3 -delay_type max]
    set names {}
    foreach p $paths {
        foreach c [get_cells -quiet -of_objects [get_pins -quiet -of_objects $p]] {
            set st [_sitetype $c]
            if {[regexp {SLICE} $st] && ![regexp $hard_re $st]} { lappend names [get_property NAME $c] }
        }
    }
    set targets [get_cells -quiet [lsort -unique $names]]
    if {[llength $targets] < 4} { return "SKIP cone_too_small_[llength $targets]" }
    if {[llength $targets] > 600} { return "SKIP cone_too_large_[llength $targets]" }
    set sc 0; set sr 0; set n 0; set cols {}; set rows {}; set crcount [dict create]
    foreach c $targets {
        set t [_tile $c]
        if {$t ne ""} {
            set cc [get_property COLUMN $t]; set rr [get_property ROW $t]
            set sc [expr {$sc+$cc}]; set sr [expr {$sr+$rr}]; incr n
            lappend cols $cc; lappend rows $rr
            set creg [get_clock_regions -quiet -of_objects [get_sites -quiet -of_objects $c]]
            if {$creg ne ""} { dict incr crcount $creg }
        }
    }
    if {$n == 0} { return "SKIP cone_unplaced" }
    set spread [expr {([lindex [lsort -integer $cols] end]-[lindex [lsort -integer $cols] 0])+([lindex [lsort -integer $rows] end]-[lindex [lsort -integer $rows] 0])}]
    if {$spread < 12} { return "SKIP cone_already_compact_spread_$spread" }
    set cr ""; set best 0
    dict for {k v} $crcount { if {$v > $best} { set best $v; set cr $k } }
    if {$cr eq ""} { return "SKIP cone_no_region" }
    set range [_compact_range $cr [expr {$sc/$n}] [expr {$sr/$n}] [llength $targets]]
    if {$range eq ""} { return "SKIP no_slices_for_cone" }
    _write_targets $targets
    return "PLAN ok cone spread=$spread region=$cr ncells=[llength $targets] range=$range slack=$slack"
}
proc do_relocate_plan {hard_re} {
    set p [lindex [get_timing_paths -max_paths 1 -nworst 1 -delay_type max] 0]
    if {$p eq ""} { return "SKIP no_path" }
    set slack [get_property SLACK $p]
    if {$slack >= 0} { return "SKIP timing_met" }
    set scell [get_cells -quiet -of_objects [get_pins -quiet [get_property STARTPOINT_PIN $p]]]
    set ecell [get_cells -quiet -of_objects [get_pins -quiet [get_property ENDPOINT_PIN $p]]]
    if {$scell eq "" || $ecell eq ""} { return "SKIP no_endpoints" }
    set stype [_sitetype $scell]; set etype [_sitetype $ecell]
    if {[regexp $hard_re $stype] && ![regexp $hard_re $etype]} {
        return [_plan_anchor $scell $ecell $hard_re $slack]
    } elseif {[regexp $hard_re $etype] && ![regexp $hard_re $stype]} {
        return [_plan_anchor $ecell $scell $hard_re $slack]
    } else {
        return [_plan_cone $hard_re $slack]
    }
}
set TFILE "%TARGETS%"
if {[catch {do_relocate_plan {%HARD%}} r]} { set r "SKIP tcl_error:$r" }
set fh [open {%RESULT%} w]; puts $fh $r; close $fh
'''.replace("%HARD%", self._HARD_ANCHOR_RE).replace("%RESULT%", str(result_file.resolve())).replace("%TARGETS%", str(targets_file.resolve()))
        plan_tcl.write_text(tcl_body)

        await self.call_tool("vivado_run_tcl", {"command": f"source {{{plan_tcl.resolve()}}}"})
        detail = result_file.read_text().strip() if result_file.exists() else ""
        if not detail.startswith("PLAN") or not targets_file.exists():
            print(f"RW relocation not applicable ({detail or 'no result'}); skipping.\n")
            return None
        targets = [ln.strip() for ln in targets_file.read_text().splitlines() if ln.strip()]
        m = re.search(r"range=(\S+)", detail)
        prange = m.group(1) if m else None
        if not targets or not prange:
            print(f"RW relocation: no targets/range parsed ({detail}); skipping.\n")
            return None
        print(f"RW relocation plan: {detail} ({len(targets)} cells)")

        # Hand the current design to RapidWright, unplace the group there (no contention),
        # write it back, then let Vivado place under the pblock and route.
        stage_in = Path(self.temp_dir) / "rw_stage_in.dcp"
        stage_out = Path(self.temp_dir) / "rw_stage_out.dcp"
        await self.call_tool("vivado_write_checkpoint", {"dcp_path": str(stage_in.resolve()), "force": True})
        rd = await self.call_tool("rapidwright_read_checkpoint", {"dcp_path": str(stage_in.resolve())})
        if "error" in rd.lower() and "success" not in rd.lower():
            print(f"⚠ RapidWright could not read DCP; skipping.\n  {rd[:200]}\n")
            return None
        up = await self.call_tool("rapidwright_unplace_cells", {"cell_names": targets, "unroute": True})
        logger.info(f"RW unplace result: {up[:200]}")
        await self.call_tool("rapidwright_write_checkpoint", {"dcp_path": str(stage_out.resolve())})
        if not stage_out.exists():
            print("⚠ RapidWright produced no DCP; reverting.\n")
            await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(self.protected_best_dcp.resolve())})
            return None

        # Vivado: constrain the unplaced group to the compact pblock, place + route.
        await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(stage_out.resolve())})
        pb_cmd = ("catch {delete_pblocks pb_relocate}; create_pblock pb_relocate; "
                  "set _fh [open {%T} r]; set _cells [read $_fh]; close $_fh; "
                  "add_cells_to_pblock pb_relocate [get_cells $_cells]; "
                  "resize_pblock pb_relocate -add %R").replace("%T", str(targets_file.resolve())).replace("%R", prange)
        await self.call_tool("vivado_run_tcl", {"command": pb_cmd})
        pr_timeout = int(timeout / 2) if timeout else 1800
        print(f"Re-placing relocated group (timeout {pr_timeout}s)...")
        await self.call_tool("vivado_place_design", {"directive": "Default", "timeout": pr_timeout})
        print(f"Re-routing (timeout {pr_timeout}s)...")
        await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": pr_timeout})
        await self.call_tool("vivado_run_tcl", {"command": "catch {delete_pblocks pb_relocate}"})

        route_status = await self.call_tool("vivado_report_route_status", {})
        fully_routed = ("fully routed" in route_status.lower()) or ("100.00%" in route_status)
        await self.call_tool("vivado_report_timing_summary", {})
        try:
            wns_after = await super().get_wns_for_target_clock(self._call_vivado_tool)
        except Exception:
            wns_after = None
        if wns_after is None and self.best_wns > float('-inf'):
            wns_after = self.best_wns

        if fully_routed and wns_after is not None and wns_after > self.protected_best_wns:
            fmax = self.calculate_fmax(wns_after, self.clock_period)
            fmax_str = f", fmax: {fmax:.2f} MHz" if fmax is not None else ""
            print(f"✓ RW relocation improved WNS to {wns_after:.3f} ns (was {self.protected_best_wns:.3f} ns"
                  f"{fmax_str}). Saving as new baseline.\n")
            await self.call_tool("vivado_write_checkpoint", {"dcp_path": str(output_dcp.resolve()), "force": True})
            if self.protected_best_dcp is not None:
                try:
                    shutil.copy2(output_dcp, self.protected_best_dcp)
                except Exception as e:
                    logger.warning(f"Could not update protected baseline copy: {e}")
            self.protected_best_wns = wns_after
            if wns_after > self.best_wns:
                self.best_wns = wns_after
            return wns_after

        if not fully_routed:
            reason = "did not route cleanly"
        elif wns_after is not None:
            reason = f"WNS {wns_after:.3f} ns did not beat baseline {self.protected_best_wns:.3f} ns"
        else:
            reason = "WNS could not be measured"
        print(f"⚠ RW relocation {reason}; reverting to the protected baseline.")
        if self.protected_best_dcp is not None and self.protected_best_dcp.exists():
            await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(self.protected_best_dcp.resolve())})
            print("✓ Restored protected baseline into Vivado.\n")
        return None

    @staticmethod
    def _rapidwright_env(script_dir: Path) -> dict:
        """Env for the validate_dcps subprocess with JAVA_HOME/RAPIDWRIGHT_PATH guaranteed.

        The RapidWright JVM needs libjvm.so; without JAVA_HOME the validator aborts Phase 1
        with 'RapidWright not initialized' and reports FAILED with 0 checks -- which would
        make good stages silently self-revert. We inherit the current env and, only if
        JAVA_HOME is missing/broken, fall back to Vivado's bundled JRE11 (same as Makefile)."""
        env = dict(os.environ)
        env.setdefault("RAPIDWRIGHT_PATH", str(script_dir / "RapidWright"))

        def _valid_jh(jh: str) -> bool:
            return bool(jh) and (Path(jh) / "lib" / "server" / "libjvm.so").exists()

        if not _valid_jh(env.get("JAVA_HOME", "")):
            candidates = []
            # Derive Vivado root from `vivado` on PATH, then its bundled jre11.
            vivado = shutil.which("vivado")
            if vivado:
                vroot = Path(vivado).resolve().parent.parent  # <root>/bin/vivado
                candidates += sorted(vroot.glob("tps/lnx64/jre11*"))
                candidates += sorted((vroot.parent).glob("*/tps/lnx64/jre11*"))
            # Common Xilinx install locations as a last resort.
            for base in ("/mnt/tools/Xilinx", "/opt/Xilinx", "/tools/Xilinx"):
                candidates += sorted(Path(base).glob("**/tps/lnx64/jre11*")) if Path(base).exists() else []
            for c in candidates:
                if _valid_jh(str(c)):
                    env["JAVA_HOME"] = str(c)
                    env["PATH"] = f"{c}/bin:" + env.get("PATH", "")
                    break
            else:
                logger.warning("Could not locate a JRE11 for RapidWright; equivalence gate may fail.")
        return env

    async def _validate_equivalence(self, golden: Path, revised: Path, vectors: int = 100) -> bool:
        """Run validate_dcps.py as a subprocess; True iff 'Overall Result: PASSED'.
        Gates netlist-editing stages (retiming) before adoption so we never keep a
        result that would score 0 on the official validator."""
        script_dir = Path(__file__).parent.resolve()
        py = script_dir / "venv" / "bin" / "python"
        if not py.exists():
            py = Path(sys.executable)
        cmd = [str(py), str(script_dir / "validate_dcps.py"),
               str(golden.resolve()), str(revised.resolve()), "--vectors", str(vectors)]
        env = self._rapidwright_env(script_dir)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(script_dir), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await proc.communicate()
            text = out.decode(errors="replace")
            passed = "Overall Result: PASSED" in text
            if not passed:
                logger.warning(f"Equivalence check did not pass. Tail:\n{text[-600:]}")
            return passed
        except Exception as e:
            logger.warning(f"Equivalence subprocess failed: {e}")
            return False

    async def run_retime_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """Deterministic register retiming for logic-depth-bound critical paths.

        Runs `phys_opt_design -retime`, which rebalances registers ACROSS combinational
        logic to shorten deep paths (long carry chains / LUT cones) WITHOUT changing
        latency or cycle-by-cycle I/O behavior -- retiming is latency-preserving. This is
        the one lever that helps logic-depth paths that placement (relocate/pblock) cannot.

        Because retiming EDITS the netlist (unlike the placement-only stages), it is not
        guaranteed equivalent in every corner (registers with INIT / async set-reset / CE
        can shift post-reset behavior). So this stage is safe-by-construction: it adopts
        the retimed result ONLY if it (a) beats the protected best WNS AND (b) PASSES the
        equivalence validator against the golden input; otherwise it reverts. Returns the
        new WNS if adopted."""
        print("\n=== Deterministic Pre-LLM Optimization: register retiming ===\n")
        if getattr(self, "retime_mode", "always") == "never":
            print("retiming disabled (--retime-mode never). Skipping.\n")
            return None
        if self.protected_best_dcp is None or not self.protected_best_dcp.exists():
            print("No protected baseline to operate on; skipping retiming.\n")
            return None
        if self._golden_input is None or not Path(self._golden_input).exists():
            print("No golden input available for equivalence gate; skipping retiming.\n")
            return None

        rt = int(timeout) if timeout else 1200
        print(f"Running phys_opt_design -retime (timeout {rt}s)...")
        self._netlist_touched = True  # retiming edits the netlist (register rebalancing)
        await self.call_tool("vivado_run_tcl", {"command": "phys_opt_design -retime", "timeout": rt})

        await self.call_tool("vivado_report_timing_summary", {})
        try:
            wns_after = await super().get_wns_for_target_clock(self._call_vivado_tool)
        except Exception:
            wns_after = None
        if wns_after is None and self.best_wns > float('-inf'):
            wns_after = self.best_wns

        if wns_after is None or wns_after <= self.protected_best_wns:
            cur = f"{wns_after:.3f} ns" if wns_after is not None else "unmeasurable"
            print(f"⚠ retiming ({cur}) did not beat baseline "
                  f"({self.protected_best_wns:.3f} ns); reverting.")
            if self.protected_best_dcp.exists():
                await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(self.protected_best_dcp.resolve())})
                print("✓ Restored protected baseline into Vivado.\n")
            return None

        # WNS improved -> write candidate and GATE on functional equivalence before adopting.
        cand = Path(self.temp_dir) / "retime_candidate.dcp"
        await self.call_tool("vivado_write_checkpoint", {
            "dcp_path": str(cand.resolve()), "force": True})
        print(f"Retiming improved WNS to {wns_after:.3f} ns (was {self.protected_best_wns:.3f} ns); "
              f"running equivalence check before adopting (retiming edits the netlist)...")
        equiv = await self._validate_equivalence(Path(self._golden_input), cand, vectors=100)
        if not equiv:
            print("⚠ retiming FAILED the equivalence check; reverting to protected baseline "
                  "(safe: nothing broken is ever kept).\n")
            await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(self.protected_best_dcp.resolve())})
            return None

        fmax = self.calculate_fmax(wns_after, self.clock_period)
        fmax_str = f", fmax: {fmax:.2f} MHz" if fmax is not None else ""
        print(f"✓ retiming improved WNS to {wns_after:.3f} ns AND passed equivalence "
              f"(was {self.protected_best_wns:.3f} ns{fmax_str}). Saving as new baseline.\n")
        await self.call_tool("vivado_write_checkpoint", {
            "dcp_path": str(output_dcp.resolve()), "force": True})
        try:
            shutil.copy2(output_dcp, self.protected_best_dcp)
        except Exception as e:
            logger.warning(f"Could not update protected baseline copy: {e}")
        self.protected_best_wns = wns_after
        if wns_after > self.best_wns:
            self.best_wns = wns_after
        return wns_after

    async def run_reimpl_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """Fresh aggressive re-implementation -- the FINAL FALLBACK stage.

        Instead of incrementally polishing the contest's given placement (what every other
        stage does), this RE-SOLVES the design from scratch: reload the golden netlist,
        unplace everything, then place/phys_opt/route with strong timing directives. On
        route-dominated designs this recovers far more than incremental relocation because
        the given placement is often mediocre. (Proven 2026-07-19 on rosetta: WNS
        -1.078 -> -0.887, +21.8 MHz vs +10.6 for the incremental flow, VALIDATED PASS.)

        It runs LAST and is adopted ONLY if it beats the protected best from the upstream
        steps AND passes the equivalence gate (phys_opt edits the netlist, so we validate
        like retime). If the incremental steps already did better, re-impl self-reverts.
        This makes it a strict, safe "final hit": it can only help, never hurt."""
        print("\n=== Deterministic Pre-LLM Optimization: fresh re-implementation (final fallback) ===\n")
        if getattr(self, "reimpl_mode", "always") == "never":
            print("re-impl disabled (--reimpl-mode never). Skipping.\n")
            return None
        if self._golden_input is None or not Path(self._golden_input).exists():
            print("No golden input available to re-solve; skipping re-impl.\n")
            return None

        # GAIN GATE: re-impl is the most expensive stage in the pipeline (it re-places and
        # re-routes the whole design from scratch, ~4 min on amd and far more on large
        # designs). Empirically it only WINS when the incremental stages barely moved --
        # it is an escape from a bad local basin, not a polisher:
        #   rosetta   recovered 10% of its slack deficit incrementally -> re-impl WON (-0.967 -> -0.887)
        #   amd       recovered 49% -> re-impl LOST (-0.858 vs -0.904), 250s wasted (53% of runtime)
        #   logicnets recovered 57% -> re-impl LOST badly (-0.417 vs -1.161)
        # Runtime is scored (gamma = runtime/3600 costs 0.1*alpha, i.e. ~0.17 MHz per minute
        # at alpha=100), so burning minutes on a stage that will revert is a direct score
        # loss. Skip re-impl once the incremental stages have already closed a large share
        # of the initial negative slack. Tunable via --reimpl-skip-if-recovered.
        skip_frac = getattr(self, "reimpl_skip_if_recovered", 0.30)
        if (skip_frac > 0 and self.initial_wns is not None and self.initial_wns < 0
                and self.protected_best_wns > float('-inf')):
            recovered = (self.protected_best_wns - self.initial_wns) / (-self.initial_wns)
            if recovered >= skip_frac:
                print(f"re-impl skipped: incremental stages already recovered "
                      f"{recovered*100:.1f}% of the initial slack deficit "
                      f"(WNS {self.initial_wns:.3f} -> {self.protected_best_wns:.3f} ns, "
                      f"threshold {skip_frac*100:.0f}%). A fresh re-implementation only wins "
                      f"from a weak incremental result, and its runtime is scored.\n")
                logger.info(f"re-impl gain-gated: recovered {recovered:.3f} >= {skip_frac}")
                return None
            print(f"re-impl proceeding: incremental stages recovered only "
                  f"{recovered*100:.1f}% of the slack deficit (< {skip_frac*100:.0f}% threshold).")

        budget = int(timeout) if timeout else 2400
        place_dir = getattr(self, "reimpl_place_directive", "ExtraTimingOpt") or "ExtraTimingOpt"
        route_dir = getattr(self, "reimpl_route_directive", "AggressiveExplore") or "AggressiveExplore"
        # Split the stage budget across the sub-steps (place-heavy, route-heavy).
        t_place = max(300, int(budget * 0.40))
        t_physa = max(120, int(budget * 0.15))
        t_route = max(300, int(budget * 0.35))
        t_physb = max(120, int(budget * 0.10))

        print(f"Re-solving from golden netlist: {Path(self._golden_input).name}")
        print(f"  place='{place_dir}' route='{route_dir}'  budget {budget}s "
              f"(place {t_place}s / physopt {t_physa}s / route {t_route}s / physopt {t_physb}s)")
        try:
            await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(Path(self._golden_input).resolve())})
            print("  unplace + place_design ...")
            await self.call_tool("vivado_run_tcl", {"command": "place_design -unplace", "timeout": 300})
            await self.call_tool("vivado_run_tcl", {
                "command": f"place_design -directive {place_dir}", "timeout": t_place})
            print("  phys_opt_design (pre-route) ...")
            await self.call_tool("vivado_run_tcl", {
                "command": "phys_opt_design -directive AggressiveExplore", "timeout": t_physa})
            print("  route_design ...")
            await self.call_tool("vivado_run_tcl", {
                "command": f"route_design -directive {route_dir}", "timeout": t_route})
            print("  phys_opt_design (post-route) ...")
            await self.call_tool("vivado_run_tcl", {
                "command": "phys_opt_design", "timeout": t_physb})
        except Exception as e:
            logger.exception(f"re-impl flow errored: {e}")
            print(f"⚠ re-impl flow errored ({e}); reverting to protected baseline.\n")
            if self.protected_best_dcp and self.protected_best_dcp.exists():
                await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(self.protected_best_dcp.resolve())})
            return None

        # Route-status gate: a not-fully-routed design reports optimistic WNS (missing
        # route delays). Never adopt/compare on such a reading -- treat it as a failure.
        route_status = await self.call_tool("vivado_report_route_status", {})
        fully_routed = ("fully routed" in route_status.lower()) or ("100.00%" in route_status)
        if not fully_routed:
            print("⚠ re-impl did not route cleanly; reverting to protected baseline.\n")
            if self.protected_best_dcp and self.protected_best_dcp.exists():
                await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(self.protected_best_dcp.resolve())})
            return None

        await self.call_tool("vivado_report_timing_summary", {})
        try:
            wns_after = await super().get_wns_for_target_clock(self._call_vivado_tool)
        except Exception:
            wns_after = None

        baseline = self.protected_best_wns if self.protected_best_wns > float('-inf') else self.best_wns
        if wns_after is None or (baseline > float('-inf') and wns_after <= baseline):
            cur = f"{wns_after:.3f} ns" if wns_after is not None else "unmeasurable"
            base_s = f"{baseline:.3f} ns" if baseline > float('-inf') else "n/a"
            print(f"⚠ re-impl ({cur}) did not beat the upstream best ({base_s}); reverting.")
            if self.protected_best_dcp and self.protected_best_dcp.exists():
                await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(self.protected_best_dcp.resolve())})
                print("✓ Restored protected baseline into Vivado.\n")
            return None

        # Beat the upstream best -> gate on equivalence before adopting (phys_opt edits netlist).
        cand = Path(self.temp_dir) / "reimpl_candidate.dcp"
        await self.call_tool("vivado_write_checkpoint", {
            "dcp_path": str(cand.resolve()), "force": True})
        base_s = f"{baseline:.3f} ns" if baseline > float('-inf') else "n/a"
        print(f"Re-impl improved WNS to {wns_after:.3f} ns (upstream best {base_s}); "
              f"running equivalence check before adopting...")
        equiv = await self._validate_equivalence(Path(self._golden_input), cand, vectors=100)
        if not equiv:
            print("⚠ re-impl FAILED the equivalence check; reverting to protected baseline.\n")
            if self.protected_best_dcp and self.protected_best_dcp.exists():
                await self.call_tool("vivado_open_checkpoint", {
                    "dcp_path": str(self.protected_best_dcp.resolve())})
            return None

        fmax = self.calculate_fmax(wns_after, self.clock_period)
        fmax_str = f", fmax: {fmax:.2f} MHz" if fmax is not None else ""
        print(f"✓ re-impl improved WNS to {wns_after:.3f} ns AND passed equivalence "
              f"(upstream best {base_s}{fmax_str}). Saving as new baseline.\n")
        await self.call_tool("vivado_write_checkpoint", {
            "dcp_path": str(output_dcp.resolve()), "force": True})
        if self.protected_best_dcp is None:
            self.protected_best_dcp = self.run_dir / "best_protected.dcp"
        try:
            shutil.copy2(output_dcp, self.protected_best_dcp)
        except Exception as e:
            logger.warning(f"Could not update protected baseline copy: {e}")
        self.protected_best_wns = wns_after
        if wns_after > self.best_wns:
            self.best_wns = wns_after
        return wns_after

    async def run_cell_replacement_baseline(self, output_dcp: Path, timeout: Optional[int] = None) -> Optional[float]:
        """Deterministic targeted cell re-placement (detour fix): move the highest-detour
        critical cells to the centroid of their connections (RapidWright
        optimize_cell_placement), then re-route in Vivado.

        Placement-only: no netlist edits, no added registers, no clock changes -> stays
        functionally equivalent (validate-safe). Runs AFTER pblock, on the CURRENT best
        design (the adopted pblock result, or the reverted phys_opt result if pblock
        raised congestion / failed). Adopted only if it routes cleanly AND beats the
        protected best; otherwise the protected baseline is reloaded and kept."""
        print("\n=== Deterministic Pre-LLM Optimization: cell re-placement (detour fix) ===\n")
        if self.cell_replace_mode == "never":
            print("cell re-placement disabled (--cell-replace-mode never). Skipping.\n")
            return None
        if self.protected_best_dcp is None or not self.protected_best_dcp.exists():
            print("No protected baseline to operate on; skipping cell re-placement.\n")
            return None

        # 1. Critical-path pins from the CURRENT best design open in Vivado.
        pins_file = Path(self.temp_dir) / "cellrepl_critical_pins.json"
        await self.call_tool("vivado_extract_critical_path_pins", {
            "num_paths": 10, "output_file": str(pins_file)})
        if not pins_file.exists():
            print("⚠ Could not extract critical-path pins; skipping cell re-placement.\n")
            return None

        # 2. Load the current best into RapidWright and analyze routing detours.
        rw = await self.call_tool("rapidwright_read_checkpoint", {
            "dcp_path": str(self.protected_best_dcp.resolve())})
        if "error" in rw.lower() and "success" not in rw.lower():
            print(f"⚠ RapidWright could not read current best DCP (possibly encrypted netlist); "
                  f"skipping cell re-placement.\n  {rw[:200]}\n")
            return None
        analysis_txt = await self.call_tool("rapidwright_analyze_net_detour", {
            "input_file": str(pins_file), "detour_threshold": 2.0})
        try:
            analysis = json.loads(analysis_txt)
        except (ValueError, TypeError):
            analysis = {}
        candidates = analysis.get("candidates", []) if isinstance(analysis, dict) else []
        if not candidates:
            print("No high-detour candidate cells found; skipping cell re-placement.\n")
            return None
        # Target cells on the worst paths (index <= 2), else the single worst.
        cell_names = list({str(c["cell"]) for c in candidates if c.get("path", 99) <= 2})
        if not cell_names:
            cell_names = [str(candidates[0]["cell"])]
        print(f"Re-placing {len(cell_names)} high-detour cells "
              f"(of {len(candidates)} candidates): {', '.join(cell_names[:5])}"
              f"{'...' if len(cell_names) > 5 else ''}")

        # 3. Re-place cells in RapidWright and write an intermediate DCP.
        self._netlist_touched = True  # RapidWright round-trip can alter the netlist
        await self.call_tool("rapidwright_optimize_cell_placement", {
            "cell_names": cell_names, "max_candidates": 10})
        rw_dcp = Path(self.temp_dir) / "cellrepl_optimized.dcp"
        await self.call_tool("rapidwright_write_checkpoint", {"dcp_path": str(rw_dcp)})
        if not rw_dcp.exists():
            print("⚠ RapidWright produced no optimized DCP; reverting.\n")
            await self.call_tool("vivado_open_checkpoint", {
                "dcp_path": str(self.protected_best_dcp.resolve())})
            return None

        # 4. Open in Vivado, re-route the moved nets, measure.
        pr_timeout = int(timeout) if timeout else 3600
        await self.call_tool("vivado_open_checkpoint", {"dcp_path": str(rw_dcp.resolve())})
        print(f"Re-routing after cell moves (timeout {pr_timeout}s)...")
        await self.call_tool("vivado_route_design", {"directive": "Default", "timeout": pr_timeout})
        route_status = await self.call_tool("vivado_report_route_status", {})
        err_m = re.search(r"nets with routing errors.*?:\s*(\d+)", route_status)
        route_errors = int(err_m.group(1)) if err_m else None
        routed_ok = ("fully routed" in route_status.lower()) or (route_errors == 0)
        await self.call_tool("vivado_report_timing_summary", {})
        try:
            wns_after = await super().get_wns_for_target_clock(self._call_vivado_tool)
        except Exception:
            wns_after = None

        # 5. Adopt only if routed cleanly AND improved; else revert to protected best.
        if routed_ok and wns_after is not None and wns_after > self.protected_best_wns:
            fmax = self.calculate_fmax(wns_after, self.clock_period)
            fmax_str = f", fmax: {fmax:.2f} MHz" if fmax is not None else ""
            print(f"✓ cell re-placement improved WNS to {wns_after:.3f} ns "
                  f"(was {self.protected_best_wns:.3f} ns{fmax_str}). Saving as new baseline.\n")
            await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()), "force": True})
            try:
                shutil.copy2(output_dcp, self.protected_best_dcp)
            except Exception as e:
                logger.warning(f"Could not update protected baseline copy: {e}")
            self.protected_best_wns = wns_after
            if wns_after > self.best_wns:
                self.best_wns = wns_after
            return wns_after

        if not routed_ok:
            reason = f"did not route cleanly ({route_errors} routing errors)" if route_errors else "did not route cleanly"
        elif wns_after is not None:
            reason = f"WNS {wns_after:.3f} ns did not beat baseline {self.protected_best_wns:.3f} ns"
        else:
            reason = "WNS could not be measured"
        print(f"⚠ cell re-placement {reason}; reverting to protected baseline.")
        await self.call_tool("vivado_open_checkpoint", {
            "dcp_path": str(self.protected_best_dcp.resolve())})
        print("✓ Restored protected baseline into Vivado.\n")
        return None

    def _restore_baseline_as_newest(self, output_dcp: Path):
        """Copy the protected baseline over the output DCP and make it the NEWEST file.

        shutil.copy2 preserves the source mtime, so a plain copy2 would stamp the scored
        file with the time the baseline was originally written -- which can be OLDER than
        a leftover *_optimized*.dcp from a previous run. The contest picks the
        most-recently-modified match, so bump the mtime to now after copying."""
        shutil.copy2(self.protected_best_dcp, output_dcp)
        try:
            os.utime(output_dcp, None)  # mtime = now; this file must win the newest-match rule
        except Exception as e:
            logger.warning(f"Could not refresh output DCP mtime: {e}")

    async def finalize_output(self, output_dcp: Path):
        """Guarantee the scored output DCP is never worse than the protected best.

        The contest scores the most-recently-modified *_optimized*.dcp, so the LLM
        stage can clobber a good baseline with a worse design. Here we make the LAST
        write the best design: if the current in-memory design beats the protected
        best we save it; otherwise we restore the protected baseline as the newest
        file. Safe against invalid/unroutable in-memory states (falls back to baseline)."""
        if self.protected_best_dcp is None or not self.protected_best_dcp.exists():
            return  # no baseline to protect (pre_opt disabled or copy failed)

        # ROUTE GATE: a design that is not fully routed reports an OPTIMISTIC WNS
        # (missing route delays), so it can "beat" the protected best on paper and then
        # score ZERO -- the official validation requires a fully placed-and-routed design.
        # Every adopting stage checks this; the final write must too, since this function
        # decides which file actually gets scored. If the in-memory design is not clean
        # (e.g. the LLM left it mid-edit), fall straight back to the protected baseline.
        try:
            route_status = await self.call_tool("vivado_report_route_status", {})
            fully_routed = (("fully routed" in route_status.lower())
                            or ("100.00%" in route_status))
        except Exception as e:
            logger.warning(f"finalize: could not read route status ({e}); restoring baseline")
            fully_routed = False

        if not fully_routed:
            logger.info("finalize: in-memory design is NOT fully routed; "
                        "restoring protected baseline as final output")
            print("⚠ Final in-memory design is not fully routed; restoring the protected "
                  "(routed) baseline as the scored output.")
            current_wns = None
        else:
            try:
                current_wns = await super().get_wns_for_target_clock(self._call_vivado_tool)
            except Exception as e:
                logger.warning(f"finalize: could not measure current WNS ({e}); restoring baseline")
                current_wns = None

        if current_wns is not None and current_wns > self.protected_best_wns:
            logger.info(f"finalize: current design ({current_wns:.3f} ns) beats protected best "
                        f"({self.protected_best_wns:.3f} ns); saving it as final output")
            try:
                await self.call_tool("vivado_write_checkpoint", {
                    "dcp_path": str(output_dcp.resolve()), "force": True
                })
                shutil.copy2(output_dcp, self.protected_best_dcp)
                self.protected_best_wns = current_wns
            except Exception as e:
                logger.warning(f"finalize: writing improved output failed ({e}); restoring baseline")
                self._restore_baseline_as_newest(output_dcp)
        else:
            cur_str = f"{current_wns:.3f}" if current_wns is not None else "unknown"
            logger.info(f"finalize: current design ({cur_str} ns) does not beat protected best "
                        f"({self.protected_best_wns:.3f} ns); restoring protected baseline as final output")
            self._restore_baseline_as_newest(output_dcp)
        print(f"✓ Final output DCP guaranteed at WNS {self.protected_best_wns:.3f} ns "
              f"(never worse than the deterministic baseline): {output_dcp}")

        # EQUIVALENCE GATE (last line of defense): route+WNS checks do NOT catch a
        # non-equivalent netlist. A stage can adopt a result that is routed and has a
        # good WNS yet is functionally BROKEN -- observed on corescore, where the
        # RapidWright cell_replace round-trip (optimize_cell_placement -> write_checkpoint
        # -> re-route) produced a design that passed route+WNS but FAILED Phase-2
        # simulation. Such a file scores ZERO on the official validator (disqualified).
        # Validate the actual scored file; if it fails, fall back to the ORIGINAL input
        # (self._golden_input), which is trivially equivalent -- a valid 0-improvement
        # submission beats a high-WNS design that gets disqualified.
        # (Future: fall back to the last construction-safe checkpoint instead of the
        # original, to preserve the pure-Vivado placement gains.)
        if not self._netlist_touched:
            # Every stage that ran was placement/routing-only (netlist untouched) -> the
            # output is provably equivalent to the input. Skip the expensive ~15-min
            # xelab/xsim check entirely (it is pure waste here and is what kept getting
            # reaped on huge designs). Netlist-editing stages (cell_replace/retime/LLM)
            # flip self._netlist_touched and DO get validated below.
            print("✓ Run was placement/routing-only (netlist untouched) -- skipping the "
                  "equivalence simulation; the output is equivalent by construction.")
        elif self._golden_input is not None and Path(self._golden_input).exists():
            print("Validating the final scored DCP for functional equivalence...")
            equiv = await self._validate_equivalence(
                Path(self._golden_input), output_dcp, vectors=100)
            if not equiv:
                logger.warning("finalize: FINAL OUTPUT FAILED EQUIVALENCE -- a stage produced a "
                               "non-equivalent design (route+WNS passed but simulation did not). "
                               "Restoring the ORIGINAL input as the scored output so the submission "
                               "is VALID (0 improvement) rather than disqualified.")
                print("⚠ FINAL OUTPUT FAILED FUNCTIONAL EQUIVALENCE -- a stage (e.g. RapidWright "
                      "cell_replace or the LLM) broke the netlist. Restoring the ORIGINAL input as "
                      "the scored output: VALID with 0 improvement, NOT disqualified.")
                try:
                    shutil.copy2(Path(self._golden_input), output_dcp)
                    os.utime(output_dcp, None)  # must be the newest matching file
                except Exception as e:
                    logger.warning(f"finalize: could not restore original input as output ({e})")
            else:
                print("✓ Final scored DCP PASSED functional equivalence.")

        # Normal, in-budget completion: cancel the hard-deadline watchdog.
        self._disarm_watchdog()

    async def optimize(self, input_dcp: Path, output_dcp: Path) -> bool:
        """Run the optimization workflow."""
        # Start timing the optimization process
        self.start_time = time.time()
        self.output_dcp = output_dcp
        self._golden_input = input_dcp  # needed by the retiming stage's equivalence gate

        # Perform initial analysis without LLM
        try:
            initial_analysis = await self.perform_initial_analysis(input_dcp)
        except Exception as e:
            logger.exception(f"Initial analysis failed: {e}")
            print(f"\n✗ Initial analysis failed: {e}\n")
            self.end_time = time.time()
            return False
        
        # Check if timing is already met
        if self.initial_wns is not None and self.initial_wns >= 0:
            print("✓ Design already meets timing! No optimization needed.\n")
            logger.info("Design already meets timing")
            # Save the design as-is
            result = await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            })
            print(f"Saved design to: {output_dcp}\n")
            
            # End timing
            self.end_time = time.time()
            total_runtime = self.end_time - self.start_time
            
            # Print summary even for early exit
            print("\n=== No Optimization Required ===")
            initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            if initial_fmax is not None:
                print(f"Design already meets timing - Fmax: {initial_fmax:.2f} MHz (WNS: {self.initial_wns:.3f} ns)")
            else:
                print(f"Design already meets timing (WNS: {self.initial_wns:.3f} ns)")
            print(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
            print(f"LLM API calls: 0 (analysis performed without LLM)")
            print(f"Estimated cost: $0.00")
            print("="*70 + "\n")
            return True

        # === Guaranteed fallback ===
        # Immediately save the current (original, already-routed) design as the output
        # so a valid, functionally-identical submission ALWAYS exists, even if every
        # optimization step below fails or the 1-hour budget is exhausted early.
        try:
            print("Saving initial fallback DCP (original design)...")
            await self.call_tool("vivado_write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()), "force": True})
            self.protected_best_dcp = self.run_dir / "best_protected.dcp"
            shutil.copy2(output_dcp, self.protected_best_dcp)
            self.protected_best_wns = self.initial_wns if self.initial_wns is not None else float('-inf')
            logger.info(f"Initial fallback saved (WNS {self.protected_best_wns:.3f} ns)")
        except Exception as e:
            logger.warning(f"Could not save initial fallback: {e}")

        # Arm the HARD wall-clock kill now that a valid protected-best exists on disk.
        # From here on, we are GUARANTEED to stop under the 1-hour contest limit even
        # if a single Vivado place/route runs away (soft budgets can't interrupt it).
        self._arm_watchdog(output_dcp)

        # === FRONT LLM PLANNER (optional, advisory) ===
        # One up-front call reads the design diagnosis and picks the ordered recipe of
        # OPTIONAL stages. phys_opt is ALWAYS forced as the first stage (non-negotiable:
        # it banks the reliable first gain, which is what carried the big designs in beta).
        # On ANY planner failure we keep the full default recipe, so the floor is never
        # worse than the proven deterministic flow.
        if self.use_planner and not self.skip_llm:
            planner_recipe = await self._run_llm_planner(initial_analysis)
            if planner_recipe is not None:
                self.pre_opt_steps = ["phys_opt"] + [s for s in planner_recipe if s != "phys_opt"]
                self._planner_ran = True
                print(f"⏱ Planner-selected pipeline: {','.join(self.pre_opt_steps)}\n")
            else:
                print("⏱ Planner unavailable/failed -- using the full default recipe "
                      f"({','.join(self.pre_opt_steps)}).\n")

        # === Deterministic pre-LLM optimization pipeline (wall-clock bounded) ===
        # Runs each configured step in order (e.g. phys_opt -> pblock). Every step is
        # functionally equivalent and only keeps a result that beats the protected
        # baseline. Each step is capped by its own budget AND the overall time left,
        # so a single place/route on a huge design cannot blow the 1-hour limit.
        # Phased budgets: phys_opt gets its own phase; pblock + cell_replace SHARE the
        # "manual" phase budget (so both fit in the middle 20-min slot, per the plan).
        MANUAL_STEPS = {"relocate", "relocate_rw", "pblock", "cell_replace"}
        manual_deadline = None  # elapsed-time deadline for the shared manual phase
        for step in self.pre_opt_steps:
            remaining = self._remaining_total()
            if remaining <= 60:
                print(f"⏱ Time budget nearly exhausted ({remaining:.0f}s left); skipping '{step}'.\n")
                continue
            if step in MANUAL_STEPS:
                if manual_deadline is None:
                    manual_deadline = self._elapsed() + self.manual_timeout
                manual_left = manual_deadline - self._elapsed()
                if manual_left <= 30:
                    print(f"⏱ Manual-opt phase budget exhausted; skipping '{step}'.\n")
                    continue
                budget = int(min(manual_left, remaining - 30))
            elif step == "pinopt":  # cheap: targeted critical-pin swaps only (~2 min); small cap
                budget = int(min(300, remaining - 30))
            elif step == "routeopt":  # best-of-N explore-route: needs room for a few routes
                budget = int(min(1500, remaining - 30))
            elif step == "reimpl":  # final fallback: its own (larger) dedicated budget
                # WORST-CASE GATE: reimpl runs a full from-scratch place+route whose
                # internal Vivado ops CANNOT be interrupted mid-run (soft timeouts).
                # If we can't finish it before the HARD deadline, do NOT start it --
                # otherwise the watchdog kills it mid-route, wasting the whole budget
                # (this is exactly what blew boom to 2.5h). Need at least half the
                # reimpl budget of real wall-clock left before the hard kill.
                hard_left = (self.hard_deadline - self._elapsed()
                             if self.hard_deadline else remaining)
                need = max(int(self.reimpl_timeout * 0.5), 600)
                if hard_left < need:
                    print(f"⏱ Skipping 'reimpl': only {hard_left:.0f}s until hard "
                          f"deadline, need ≥{need}s to finish a from-scratch place+route "
                          f"(reimpl can't be interrupted; not starting it).\n")
                    continue
                budget = int(min(self.reimpl_timeout, hard_left - 60, remaining - 30))
            elif step == "llm":  # LLM stage self-limits via llm_timeout; budget here is informational
                budget = int(min(self.llm_timeout, remaining - 30))
            else:  # phys_opt (and any future dedicated-phase step)
                budget = int(min(self.phys_opt_timeout, remaining - 30))
            print(f"⏱ Stage '{step}': budget {budget}s (overall {remaining:.0f}s left of {self.total_timeout}s)")
            try:
                if step == "phys_opt":
                    await self.run_physopt_baseline(output_dcp, timeout=budget)
                elif step == "pinopt":
                    await self.run_pinopt_baseline(output_dcp, timeout=budget)
                elif step == "routeopt":
                    await self.run_routeopt_baseline(output_dcp, timeout=budget)
                elif step == "relocate":
                    await self.run_relocate_baseline(output_dcp, timeout=budget)
                elif step == "relocate_rw":
                    await self.run_relocate_rw_baseline(output_dcp, timeout=budget)
                elif step == "retime":
                    await self.run_retime_baseline(output_dcp, timeout=budget)
                elif step == "pblock":
                    await self.run_pblock_baseline(output_dcp, timeout=budget)
                elif step == "cell_replace":
                    await self.run_cell_replacement_baseline(output_dcp, timeout=budget)
                elif step == "reimpl":
                    await self.run_reimpl_baseline(output_dcp, timeout=budget)
                elif step == "llm":
                    if self.skip_llm:
                        print("'llm' step present in --pre-opt but --skip-llm is set; skipping it.\n")
                    else:
                        await self._run_llm_stage(input_dcp, output_dcp, initial_analysis)
                else:
                    logger.warning(f"Unknown pre-opt step '{step}', skipping")
            except Exception as e:
                logger.exception(f"Pre-opt step '{step}' failed: {e}")
                print(f"⚠ Pre-opt step '{step}' failed, continuing: {e}\n")
            if self.protected_best_wns >= 0:
                break  # timing met on the SAVED (routed) design; no need for further pre-opt

        # WNS of the best deterministic result so far (for the LLM status note).
        baseline_wns = (self.protected_best_wns
                        if (self.protected_best_dcp and self.protected_best_wns > float('-inf'))
                        else None)

        # If timing is now met after the deterministic pipeline, finish here.
        if self.protected_best_wns >= 0:
            print("✓ Timing met after deterministic optimization!\n")
            await self.finalize_output(output_dcp)
            self.end_time = time.time()
            self._print_optimization_summary()
            return True

        # Trailing LLM phase: run ONLY if 'llm' was not already a positioned pipeline
        # step (in --pre-opt), the LLM is not disabled, AND the front planner did not run.
        # In front-planner mode the LLM's role is planning the recipe, not a finishing
        # optimization pass (per the front-planner-only design), so we do NOT also run a
        # trailing finisher.
        if "llm" not in self.pre_opt_steps and not self.skip_llm and not self._planner_ran:
            await self._run_llm_stage(input_dcp, output_dcp, initial_analysis)
        elif self._planner_ran:
            print("Front-planner mode: recipe was chosen up front; no trailing LLM finisher.\n")
        elif self.skip_llm and "llm" not in self.pre_opt_steps:
            print("Skipping LLM stage (--skip-llm). Deterministic baseline is the final result.\n")

        await self.finalize_output(output_dcp)
        self.end_time = time.time()
        self._print_optimization_summary()
        return self.protected_best_dcp is not None

    async def _planner_diagnosis(self, initial_analysis: str) -> str:
        """Build a compact design profile for the planner: worst-path logic/route split,
        logic levels + CARRY8 count, design size, DSP/BRAM density, plus the existing
        initial analysis. Best-effort -- any missing field just isn't reported."""
        prof = []
        try:
            rpt = await self.call_tool("vivado_run_tcl", {
                "command": "report_timing -max_paths 1 -setup -return_string", "timeout": 120})
            m = re.search(r"logic\s+[\d.]+ns\s+\(([\d.]+)%\)\s+route\s+[\d.]+ns\s+\(([\d.]+)%\)", rpt)
            if m:
                prof.append(f"worst_path_logic_pct: {float(m.group(1)):.0f}")
                prof.append(f"worst_path_route_pct: {float(m.group(2)):.0f}")
            lm = re.search(r"Logic Levels:\s+(\d+)\s*\(([^)]*)\)", rpt)
            if lm:
                cm = re.search(r"CARRY8=(\d+)", lm.group(2))
                prof.append(f"worst_path_logic_levels: {int(lm.group(1))}")
                prof.append(f"worst_path_carry8_count: {int(cm.group(1)) if cm else 0}")
        except Exception as e:
            logger.warning(f"planner diagnosis (worst-path) failed: {e}")
        try:
            stats_tcl = (
                r'set occ [llength [get_sites -quiet -filter {SITE_TYPE =~ SLICE* && IS_USED} SLICE_*]]; '
                r'set ndsp [llength [get_sites -quiet -filter {IS_USED} DSP*]]; '
                r'set nbram [llength [get_sites -quiet -filter {IS_USED} RAMB*]]; '
                r'puts "PDIAG occ $occ dsp $ndsp bram $nbram"')
            stats = await self.call_tool("vivado_run_tcl", {"command": stats_tcl, "timeout": 120})
            sm = re.search(r"PDIAG occ (\d+) dsp (\d+) bram (\d+)", stats)
            if sm:
                occ, ndsp, nbram = (int(g) for g in sm.groups())
                prof.append(f"occupied_slices: {occ}")
                prof.append(f"dsp_bram_density: {((ndsp + nbram) / occ):.3f}" if occ else "dsp_bram_density: 0")
                prof.append(f"design_size: {'LARGE' if occ > 20000 else ('MEDIUM' if occ > 5000 else 'SMALL')}")
        except Exception as e:
            logger.warning(f"planner diagnosis (size/density) failed: {e}")
        prof.append(f"pblock_recommended(spread heuristic): {bool(getattr(self, 'pblock_recommended', False))}")
        if getattr(self, "high_fanout_nets", None):
            prof.append(f"max_fanout: {self.high_fanout_nets[0][1]} (top high-fanout net)")
        return "WORST-PATH / DESIGN PROFILE:\n" + "\n".join(prof) + "\n\n" + (initial_analysis or "")

    async def _run_llm_planner(self, initial_analysis: str) -> Optional[list]:
        """FRONT PLANNER: one LLM call that reads the diagnosis and returns an ordered
        recipe of OPTIONAL post-phys_opt stages. phys_opt is mandatory and NOT the
        planner's to choose. Returns a validated list, or None on ANY failure (missing
        prompt/key, API error, bad JSON, empty/invalid recipe) so the caller uses the
        full default recipe. This is the safety contract: the planner can only add value,
        never drop below the proven deterministic flow."""
        # NOTE: "llm" is deliberately NOT in the menu. Letting the planner append the LLM
        # finisher triggered a full 50+ call agentic loop (e.g. on amd) that added large
        # runtime+$ for no measured Fmax gain -- a net SCORE LOSS (gamma+beta penalties).
        # The OG fanout strategy it would invoke was disproven on boom and rarely helps.
        # It stays documented in SYSTEM_PROMPT.TXT for manual --no-planner use; the front
        # planner remains fast + cheap (deterministic recipe only).
        MENU = {"pblock", "relocate", "pinopt", "reimpl", "routeopt", "cell_replace", "retime"}
        planner_prompt = load_planner_prompt()
        if planner_prompt is None or not self.api_key:
            return None
        try:
            diagnosis = await self._planner_diagnosis(initial_analysis)
        except Exception as e:
            logger.warning(f"planner: diagnosis build failed ({e}); using default recipe.")
            return None
        try:
            self.llm_call_count += 1
            logger.info("LLM planner call (recipe selection)")
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": planner_prompt},
                          {"role": "user", "content": f"{diagnosis}\n\nReturn the JSON recipe now."}],
                max_tokens=512,
                timeout=60,  # fail fast to the default recipe if OpenRouter hangs
                extra_body={"usage": {"include": True}})
            if hasattr(response, "usage") and response.usage and getattr(response.usage, "cost", None) is not None:
                self.total_cost += float(response.usage.cost)
            content = (response.choices[0].message.content or "").strip()
            jm = re.search(r"\{.*\}", content, re.DOTALL)  # tolerate stray prose
            data = json.loads(jm.group(0) if jm else content)
            raw = data.get("recipe", [])
            recipe = [s for s in raw if isinstance(s, str) and s in MENU and s != "phys_opt"]
            if not recipe:
                logger.warning("planner: empty/invalid recipe; using default.")
                return None
            print(f"🧠 Planner recipe (after mandatory phys_opt): {recipe}")
            print(f"   rationale: {data.get('rationale', '(none)')}\n")
            return recipe
        except Exception as e:
            logger.warning(f"planner: LLM call/parse failed ({e}); using default recipe.")
            return None

    async def _run_llm_stage(self, input_dcp: Path, output_dcp: Path, initial_analysis: str) -> None:
        """LLM-driven optimization loop. Runs either as a positioned pipeline step ('llm'
        in --pre-opt, e.g. phys_opt,llm,relocate,...) or as the trailing phase. Updates the
        protected best + output DCP but does NOT print the final summary or set end_time --
        optimize() does that exactly once after all stages complete."""
        self._netlist_touched = True  # the LLM may issue netlist-editing tool calls
        baseline_wns = self.protected_best_wns if self.protected_best_wns > float('-inf') else None
        system_prompt_template = load_system_prompt()
        system_prompt = system_prompt_template.format(
            temp_dir=self.temp_dir, input_dcp=input_dcp.resolve())
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Optimize this FPGA design for timing.

PATHS:
- Input DCP: {input_dcp.resolve()}
- Output DCP (save final result here): {output_dcp.resolve()}
- Run directory (for intermediate files): {self.temp_dir}

CURRENT STATE:
- Vivado has the input design ALREADY OPEN and analyzed
- RapidWright has the input design ALREADY LOADED (from initial analysis)

INITIAL ANALYSIS RESULTS:
{initial_analysis}

Proceed with optimization strategy based on the analysis above. Do NOT reload the design in either Vivado or RapidWright - both already have it loaded."""}
        ]
        # Tell the LLM the current best (already saved) so it only pursues improvements.
        if baseline_wns is not None:
            baseline_fmax = self.calculate_fmax(baseline_wns, self.clock_period)
            fmax_str = f" (fmax: {baseline_fmax:.2f} MHz)" if baseline_fmax is not None else ""
            self.messages.append({
                "role": "user",
                "content": (
                    f"IMPORTANT UPDATE: Deterministic optimization has ALREADY been run on the open "
                    f"design. The CURRENT in-memory WNS is {baseline_wns:.3f} ns{fmax_str}, and this "
                    f"baseline is ALREADY SAVED to the output DCP. Only commit further changes if they "
                    f"IMPROVE on {baseline_wns:.3f} ns; if you cannot beat it, keep the current result "
                    f"and finish."
                )
            })

        max_iterations = self.iteration + 50  # safety limit (relative, in case called again)
        llm_stage_start = self._elapsed()
        print(f"=== Starting LLM-Driven Optimization (budget: {self.llm_timeout}s, "
              f"${self.cost_cap:.2f}, {self._remaining_total():.0f}s of 1hr left) ===\n")

        stop_reason = None
        while self.iteration < max_iterations:
            # Enforce wall-clock and cost budgets before each LLM iteration.
            if self._remaining_total() <= 30:
                stop_reason = f"overall 1-hour budget reached ({self._remaining_total():.0f}s left)"
                break
            if (self._elapsed() - llm_stage_start) >= self.llm_timeout:
                stop_reason = f"LLM stage time budget ({self.llm_timeout}s) reached"
                break
            if self.total_cost >= self.cost_cap:
                stop_reason = f"cost cap (${self.cost_cap:.2f}) reached at ${self.total_cost:.4f}"
                break

            self.iteration += 1
            logger.info(f"=== Iteration {self.iteration} ===")
            try:
                response_text, is_done = await self.get_completion()
                print(f"\n{response_text}\n")
                if is_done:
                    logger.info("LLM stage signaled completion")
                    await self.finalize_output(output_dcp)
                    return
            except Exception as e:
                logger.exception(f"Error during LLM optimization: {e}")
                self.messages.append({
                    "role": "user",
                    "content": f"An error occurred: {e}. Please verify your approach and continue or report if unrecoverable."
                })

        logger.warning(f"LLM stage stopped: {stop_reason or 'reached maximum iterations'}")
        print(f"\n⏱ LLM stage stopped: {stop_reason or 'reached maximum iterations'}\n")
        # Guarantee the output DCP is the best design (never worse than the baseline).
        await self.finalize_output(output_dcp)
        return
    
    def save_token_usage_report(self, output_path: Path):
        """Save detailed token usage report to JSON file."""
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        # Calculate tool call statistics
        total_tool_time = sum(detail['elapsed_time'] for detail in self.tool_call_details)
        tool_counts = {}
        for detail in self.tool_call_details:
            tool_name = detail['tool_name']
            if tool_name not in tool_counts:
                tool_counts[tool_name] = 0
            tool_counts[tool_name] += 1
        
        # Calculate total runtime
        total_runtime = None
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
        
        # Calculate fmax values
        initial_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
        best_fmax = self.calculate_fmax(self.best_wns, self.clock_period) if self.best_wns > float('-inf') else None
        fmax_improvement = (best_fmax - initial_fmax) if (initial_fmax is not None and best_fmax is not None) else None
        
        report = {
            "model": self.model,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_runtime_seconds": total_runtime,
                "total_llm_calls": self.llm_call_count,
                "total_iterations": self.iteration,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
                "total_cached_tokens": total_cached,
                "total_reasoning_tokens": total_reasoning,
                "total_cost": self.total_cost,
                "clock_period_ns": self.clock_period,
                "initial_wns": self.initial_wns,
                "best_wns": self.best_wns,
                "wns_improvement": self.best_wns - self.initial_wns if self.initial_wns is not None else None,
                "initial_fmax_mhz": initial_fmax,
                "best_fmax_mhz": best_fmax,
                "fmax_improvement_mhz": fmax_improvement,
                "total_tool_calls": len(self.tool_call_details),
                "total_tool_time_seconds": total_tool_time,
                "tool_call_counts": tool_counts
            },
            "per_llm_call_details": self.api_call_details,
            "per_tool_call_details": self.tool_call_details
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Token usage report saved to {output_path}")
    
    def _print_optimization_summary(self, max_iterations_reached: bool = False):
        """Print detailed optimization summary including token usage and costs."""
        title = "Optimization Summary (Max Iterations Reached)" if max_iterations_reached else "Optimization Summary"
        print(f"\n{'='*70}")
        print(f"{title}")
        print(f"{'='*70}")
        
        # Calculate total runtime
        if self.start_time is not None:
            total_runtime = (self.end_time or time.time()) - self.start_time
            print(f"\nTOTAL RUNTIME: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
        
        # Report the WNS of the design ACTUALLY SAVED (protected best = fully-routed,
        # adopted result), NOT the optimistic self.best_wns which the generic call_tool
        # wrapper updates on every timing read including not-fully-routed intermediates.
        if self.protected_best_wns > float('-inf'):
            best_wns = self.protected_best_wns
        elif self.best_wns > float('-inf'):
            best_wns = self.best_wns
        else:
            best_wns = None
        result_lines = self._format_fmax_results(
            self.clock_period, self.initial_wns, best_wns, result_label="Best"
        )
        if result_lines:
            print(f"\nFMAX RESULTS:")
            print("\n".join(result_lines))
        
        # Iteration stats
        print(f"\nITERATION STATS:")
        print(f"  Total iterations:    {self.iteration}")
        print(f"  LLM API calls:       {self.llm_call_count}")
        
        # Token usage
        print(f"\nTOKEN USAGE:")
        print(f"  Prompt tokens:       {self.total_prompt_tokens:,}")
        print(f"  Completion tokens:   {self.total_completion_tokens:,}")
        print(f"  Total tokens:        {self.total_tokens:,}")
        
        # Calculate total cached and reasoning tokens
        total_cached = sum(detail.get('cached_tokens', 0) for detail in self.api_call_details)
        total_reasoning = sum(detail.get('reasoning_tokens', 0) for detail in self.api_call_details)
        
        if total_cached > 0:
            print(f"  Cached tokens:       {total_cached:,} (saved cost)")
        if total_reasoning > 0:
            print(f"  Reasoning tokens:    {total_reasoning:,}")
        
        # Cost
        print(f"\nCOST:")
        print(f"  Model:               {self.model}")
        if self.total_cost > 0:
            print(f"  Total cost:          ${self.total_cost:.4f}")
        else:
            print(f"  Total cost:          Not available")
        
        # Tool call summary
        if self.tool_call_details:
            print(f"\nTOOL CALLS SUMMARY:")
            print(f"  Total tool calls:    {len(self.tool_call_details)}")
            
            # Calculate total time spent in tool calls
            total_tool_time = sum(detail['elapsed_time'] for detail in self.tool_call_details)
            print(f"  Total tool time:     {total_tool_time:.2f}s")
            
            # Count by tool type
            tool_counts = {}
            for detail in self.tool_call_details:
                tool_name = detail['tool_name']
                if tool_name not in tool_counts:
                    tool_counts[tool_name] = 0
                tool_counts[tool_name] += 1
            
            print(f"\n  Tool call breakdown:")
            for tool_name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
                print(f"    {tool_name}: {count}")
            
            # Detailed tool call list
            print(f"\n  Detailed tool call log:")
            print(f"  {'#':<5} {'Iter':<6} {'Tool':<40} {'Time (s)':<12} {'WNS (ns)':<12} {'Status':<10}")
            print(f"  {'-'*5} {'-'*6} {'-'*40} {'-'*12} {'-'*12} {'-'*10}")
            
            for i, detail in enumerate(self.tool_call_details, 1):
                tool_name = detail['tool_name']
                iteration = detail.get('iteration', 0)
                elapsed = detail['elapsed_time']
                wns = detail.get('wns')
                error = detail.get('error', False)
                
                # Format WNS column
                wns_str = f"{wns:.3f}" if wns is not None else "-"
                
                # Format status
                status_str = "ERROR" if error else "OK"
                
                print(f"  {i:<5} {iteration:<6} {tool_name:<40} {elapsed:<12.2f} {wns_str:<12} {status_str:<10}")
                
                # If error, show error message on next line
                if error and 'error_message' in detail:
                    print(f"        Error: {detail['error_message'][:80]}")
        
        # Per-call breakdown if debug mode
        if self.debug and self.api_call_details:
            print(f"\nPER-CALL BREAKDOWN:")
            
            # Check if we have cached or reasoning tokens to display
            has_cached = any(detail.get('cached_tokens', 0) > 0 for detail in self.api_call_details)
            has_reasoning = any(detail.get('reasoning_tokens', 0) > 0 for detail in self.api_call_details)
            has_cost = any(detail.get('cost', 0) > 0 for detail in self.api_call_details)
            
            # Build header
            header = f"  {'Call':<6} {'Iter':<6} {'Prompt':<10} {'Completion':<12}"
            if has_cached:
                header += f" {'Cached':<10}"
            if has_reasoning:
                header += f" {'Reasoning':<10}"
            header += f" {'Total':<10}"
            if has_cost:
                header += f" {'Cost':<12}"
            print(header)
            
            # Build separator
            separator = f"  {'-'*6} {'-'*6} {'-'*10} {'-'*12}"
            if has_cached:
                separator += f" {'-'*10}"
            if has_reasoning:
                separator += f" {'-'*10}"
            separator += f" {'-'*10}"
            if has_cost:
                separator += f" {'-'*12}"
            print(separator)
            
            # Print details
            for detail in self.api_call_details:
                line = (f"  {detail['call_number']:<6} {detail['iteration']:<6} "
                       f"{detail['prompt_tokens']:<10,} {detail['completion_tokens']:<12,}")
                if has_cached:
                    line += f" {detail.get('cached_tokens', 0):<10,}"
                if has_reasoning:
                    line += f" {detail.get('reasoning_tokens', 0):<10,}"
                line += f" {detail['total_tokens']:<10,}"
                if has_cost:
                    cost = detail.get('cost', 0)
                    line += f" ${cost:<11.4f}" if cost > 0 else f" {'N/A':<12}"
                print(line)
        
        print(f"\n{'='*70}\n")
        
        # Save detailed report to JSON in run directory
        try:
            report_path = self.run_dir / "token_usage.json"
            self.save_token_usage_report(report_path)
            print(f"Detailed token usage report saved to: {report_path}\n")
        except Exception as e:
            logger.warning(f"Failed to save token usage report: {e}")
    


class FPGAOptimizerTest(DCPOptimizerBase):
    """
    Test mode for FPGA Design Optimization - hardcodes all tool calls to diagnose issues.
    
    This class runs a deterministic optimization flow without using any LLM, 
    making it easier to identify where MCP servers or Vivado might hang.
    """
    
    def __init__(self, debug: bool = False, run_dir: Optional[Path] = None):
        super().__init__(debug=debug, run_dir=run_dir)
        self.final_wns = None
    
    async def start_servers(self):
        """Start and connect to both MCP servers."""
        await super().start_servers(log_prefix="[TEST]")
    
    async def call_vivado_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a Vivado tool call with timing and logging."""
        logger.info(f"[VIVADO] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling vivado_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.vivado_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[VIVADO] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] vivado_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: vivado_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[VIVADO] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: vivado_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    async def call_rapidwright_tool(self, tool_name: str, arguments: dict, timeout: float = 300.0) -> str:
        """Execute a RapidWright tool call with timing and logging."""
        logger.info(f"[RAPIDWRIGHT] Calling {tool_name} with args: {json.dumps(arguments)[:200]}...")
        print(f"[TEST] Calling rapidwright_{tool_name}...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                self.rapidwright_session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            logger.info(f"[RAPIDWRIGHT] {tool_name} completed in {elapsed:.2f}s")
            print(f"[TEST] rapidwright_{tool_name} completed in {elapsed:.2f}s")
            
            # Extract text content from result
            if result.content:
                text_parts = [c.text for c in result.content if hasattr(c, 'text')]
                return "\n".join(text_parts)
            return "(no output)"
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} TIMED OUT after {elapsed:.2f}s")
            print(f"[TEST] ERROR: rapidwright_{tool_name} TIMED OUT after {elapsed:.2f}s")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[RAPIDWRIGHT] {tool_name} FAILED after {elapsed:.2f}s: {e}")
            print(f"[TEST] ERROR: rapidwright_{tool_name} failed after {elapsed:.2f}s: {e}")
            raise
    
    def parse_wns_from_timing_report(self, timing_report: str) -> Optional[float]:
        """Extract WNS from timing report using shared parsing logic."""
        return parse_timing_summary_static(timing_report)["wns"]
    
    async def _call_vivado_for_clock(self, tool_name: str, arguments: dict) -> str:
        """Helper to call Vivado tools for clock period query."""
        return await self.call_vivado_tool(tool_name, arguments, timeout=60.0)
    
    async def fetch_clock_period(self) -> Optional[float]:
        """Query clock period with test-mode logging."""
        period = await super().get_clock_period(self._call_vivado_for_clock)
        if period is not None:
            clock_info = f" (target clock: {self.target_clock})" if self.target_clock else ""
            print(f"[TEST] Clock period: {period:.3f} ns{clock_info}")
        else:
            print("[TEST] WARNING: Could not parse clock period from Vivado")
        return period
    
    async def run_test(self, input_dcp: Path, output_dcp: Path, max_nets_to_optimize: int = 5) -> bool:
        """
        Run the deterministic test optimization flow.
        
        Steps:
        1. Open the input DCP in Vivado
        2. Report timing in Vivado
        3. Get the critical high fan out nets from Vivado
        4. Open the DCP in RapidWright
        5. Apply the fanout optimization for each high fanout net
        6. Write a DCP out from RapidWright
        7. Read the RapidWright generated DCP into Vivado
        8. Route the design in Vivado
        9. Report timing and compare WNS
        """
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print(f"Max nets to optimize: {max_nets_to_optimize}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # ================================================================
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            # Initialize RapidWright (Vivado will auto-start when first used)
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # ================================================================
            # Step 1: Open the input DCP in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # ================================================================
            # Step 2: Report timing in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 2: Report timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # ================================================================
            # Step 3: Get critical high fanout nets
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 3: Get critical high fanout nets")
            print("-"*60)
            
            result = await self.call_vivado_tool("get_critical_high_fanout_nets", {
                "num_paths": 50,
                "min_fanout": 100,
                "exclude_clocks": True
            }, timeout=600.0)
            print(f"High fanout nets report:\n{result}")
            logger.info(f"High fanout nets: {result}")
            
            # Parse the nets
            self.high_fanout_nets = self.parse_high_fanout_nets(result)
            print(f"\nParsed {len(self.high_fanout_nets)} high fanout nets")
            
            if not self.high_fanout_nets:
                print("WARNING: No high fanout nets found to optimize!")
                logger.warning("No high fanout nets found to optimize")
            
            # Select top nets to optimize
            nets_to_optimize = self.high_fanout_nets[:max_nets_to_optimize]
            print(f"Will optimize {len(nets_to_optimize)} nets:")
            for net_name, fanout, path_count in nets_to_optimize:
                print(f"  - {net_name} (fanout={fanout}, paths={path_count})")
            
            # ================================================================
            # Step 4: Open the DCP in RapidWright
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 4: Open DCP in RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # ================================================================
            # Step 5: Apply fanout optimization for each high fanout net
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 5: Apply fanout optimizations in RapidWright")
            print("-"*60)
            
            successful_optimizations = 0
            for i, (net_name, fanout, path_count) in enumerate(nets_to_optimize):
                print(f"\n[{i+1}/{len(nets_to_optimize)}] Optimizing net: {net_name}")
                print(f"    Fanout: {fanout}, Critical paths: {path_count}")
                
                # Calculate split factor: fanout/100, min 2, max 8
                split_factor = max(2, min(8, fanout // 100))
                print(f"    Split factor: {split_factor}")
                
                try:
                    result = await self.call_rapidwright_tool("optimize_fanout", {
                        "net_name": net_name,
                        "split_factor": split_factor
                    }, timeout=300.0)
                    print(f"    Result: {result[:500]}...")
                    logger.info(f"Optimize fanout {net_name}: {result}")
                    
                    # Check if successful
                    if "error" not in result.lower() or "success" in result.lower():
                        successful_optimizations += 1
                except Exception as e:
                    print(f"    FAILED: {e}")
                    logger.error(f"Failed to optimize {net_name}: {e}")
            
            print(f"\nSuccessfully optimized {successful_optimizations}/{len(nets_to_optimize)} nets")
            
            # ================================================================
            # Step 6: Write DCP from RapidWright
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 6: Write DCP from RapidWright")
            print("-"*60)
            
            rapidwright_dcp = Path(self.temp_dir) / "rapidwright_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rapidwright_dcp),
                "overwrite": True
            }, timeout=600.0)
            print(f"Write checkpoint result:\n{result}")
            logger.info(f"RapidWright write checkpoint: {result}")
            
            # Check if the file was created
            if rapidwright_dcp.exists():
                print(f"DCP file created: {rapidwright_dcp} ({rapidwright_dcp.stat().st_size} bytes)")
            else:
                print("WARNING: DCP file was not created!")
                logger.warning("RapidWright DCP file not created")
            
            # ================================================================
            # Step 7: Read RapidWright DCP into Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 7: Read RapidWright DCP into Vivado")
            print("-"*60)
            
            # Note: Opening a RapidWright-generated DCP takes MUCH longer than
            # opening the original DCP because:
            # 1. Vivado must reload encrypted IP blocks from disk
            # 2. Vivado must reconstruct internal data structures
            # For large designs, this can take 10-30 minutes
            RAPIDWRIGHT_DCP_TIMEOUT = 300.0  # 5 minutes
            
            # Check if there's a Tcl script we need to source first (for encrypted IP)
            tcl_script = rapidwright_dcp.with_suffix('.tcl')
            if tcl_script.exists():
                print(f"Found Tcl script for encrypted IP: {tcl_script}")
                print(f"Note: This may take 10-30 minutes for large designs...")
                # Source the Tcl script instead of directly opening the DCP
                result = await self.call_vivado_tool("run_tcl", {
                    "command": f"source {{{tcl_script}}}"
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Source Tcl script result:\n{result}")
            else:
                # Opening a RapidWright-generated DCP can take longer than original
                # because Vivado needs to reconstruct some internal data structures
                result = await self.call_vivado_tool("open_checkpoint", {
                    "dcp_path": str(rapidwright_dcp)
                }, timeout=RAPIDWRIGHT_DCP_TIMEOUT)
                print(f"Open RapidWright DCP result:\n{result}")
            logger.info(f"Open RapidWright DCP: {result}")
            
            # ================================================================
            # Step 8: Route the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 8: Route design in Vivado")
            print("-"*60)
            
            # First check route status
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status before routing:\n{result[:1500]}...")
            logger.info(f"Route status before routing: {result}")
            
            # Route the design
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default",
            }, timeout=600.0)  # 2 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status again
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # ================================================================
            # Step 9: Report final timing
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 9: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # ================================================================
            # Write final DCP and report results
            # ================================================================
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint (regardless of improvement)
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Nets optimized: {successful_optimizations}/{len(nets_to_optimize)}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"Test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False
############# Adding my own test flow: kitta 
    async def run_test_kitta(self, input_dcp: Path, output_dcp: Path, max_nets_to_optimize: int = 5) -> bool:
        """
        Run a deterministic phys_opt + route flow with before/after analysis.

        Flow:
        1. Open the input DCP

        Initial Analysis:
        2. Report timing summary
        3. Report worst critical path
        4. Report QoR suggestions
        5. Identify critical high-fanout nets
        6. Analyze critical path spread using RapidWright

        Optimization:
        7. Run phys_opt_design
        8. Run route_design

        Post-Optimization Analysis:
        9. Report timing summary
        10. Report worst critical path
        11. Report QoR suggestions
        12. Analyze critical path spread using RapidWright

        Output:
        13. Write optimized DCP
        """
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE - KITTA PHYS_OPT FLOW")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            async def report_worst_critical_path(label: str) -> str:
                print("\n" + "-"*60)
                print(label)
                print("-"*60)
                result = await self.call_vivado_tool("run_tcl", {
                    "command": "report_timing -max_paths 1"
                }, timeout=600.0)
                print(f"Worst critical path report:\n{result}")
                logger.info(f"{label}: {result}")
                return result

            async def report_qor_suggestions(label: str) -> str:
                print("\n" + "-"*60)
                print(label)
                print("-"*60)
                result = await self.call_vivado_tool("run_tcl", {
                    "command": "report_qor_suggestions"
                }, timeout=600.0)
                print(f"QoR suggestions:\n{result}")
                logger.info(f"{label}: {result}")
                return result

            async def analyze_critical_path_spread(label: str, dcp_path: Path, output_name: str) -> str:
                print("\n" + "-"*60)
                print(label)
                print("-"*60)

                critical_paths_file = Path(self.temp_dir) / output_name
                result = await self.call_vivado_tool("extract_critical_path_cells", {
                    "num_paths": 50,
                    "output_file": str(critical_paths_file)
                }, timeout=600.0)
                print(f"Extract critical paths result:\n{result[:2000]}...")
                logger.info(f"{label} extract critical paths: {result}")

                result = await self.call_rapidwright_tool("read_checkpoint", {
                    "dcp_path": str(dcp_path.resolve())
                }, timeout=600.0)
                print(f"RapidWright read checkpoint result:\n{result[:1000]}...")
                logger.info(f"{label} RapidWright read checkpoint: {result}")

                result = await self.call_rapidwright_tool("analyze_critical_path_spread", {
                    "input_file": str(critical_paths_file)
                }, timeout=300.0)
                result_text = result if isinstance(result, str) else str(result)
                print(f"Critical path spread analysis:\n{result_text[:3000]}...")
                logger.info(f"{label} critical path spread: {result}")
                return result_text

            # ================================================================
            # Setup: Initialize RapidWright (Vivado starts automatically)
            # ================================================================
            print("\n" + "-"*60)
            print("SETUP: Initialize RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # ================================================================
            # Step 1: Open the input DCP in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")

            print("\n" + "="*70)
            print("INITIAL ANALYSIS")
            print("="*70)
            
            # ================================================================
            # Step 2: Report timing summary
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 2: Report timing summary")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # ================================================================
            # Step 3: Report worst critical path
            # ================================================================
            await report_worst_critical_path("STEP 3: Report worst critical path")
                        
            # ================================================================
            # Step 4: Report QoR suggestions
            # ================================================================
            await report_qor_suggestions("STEP 4: Report QoR suggestions")

            # ================================================================
            # Step 5: Identify critical high-fanout nets
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 5: Identify critical high-fanout nets")
            print("-"*60)

            result = await self.call_vivado_tool("get_critical_high_fanout_nets", {
                "num_paths": 50,
                "min_fanout": 100,
                "exclude_clocks": True
            }, timeout=600.0)
            print(f"High fanout nets report:\n{result}")
            logger.info(f"Initial high fanout nets: {result}")

            self.high_fanout_nets = self.parse_high_fanout_nets(result)
            print(f"\nParsed {len(self.high_fanout_nets)} critical high-fanout nets")
            for net_name, fanout, path_count in self.high_fanout_nets[:max_nets_to_optimize]:
                print(f"  - {net_name} (fanout={fanout}, paths={path_count})")
            if len(self.high_fanout_nets) > max_nets_to_optimize:
                print(f"  ... and {len(self.high_fanout_nets) - max_nets_to_optimize} more")

            # ================================================================
            # Step 6: Analyze critical path spread using RapidWright
            # ================================================================
            await analyze_critical_path_spread(
                "STEP 6: Analyze critical path spread using RapidWright",
                input_dcp,
                "kitta_initial_critical_paths.json"
            )

            print("\n" + "="*70)
            print("OPTIMIZATION")
            print("="*70)

            # ================================================================
            # Step 7: Run phys_opt_design
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 7: Run phys_opt_design")
            print("-"*60)
            result = await self.call_vivado_tool("run_tcl", {
                "command": "phys_opt_design -directive Default"
            }, timeout=1200.0)
            print(f"phys_opt_design result:\n{result}")
            logger.info(f"phys_opt_design result: {result}")
            
            # ================================================================
            # Step 8: Run route_design
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 8: Run route_design")
            print("-"*60)

            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=3000.0)
            print(f"Route status before routing:\n{result[:1500]}...")
            logger.info(f"Route status before routing: {result}")
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default",
            }, timeout=3600.0)
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            result = await self.call_vivado_tool("report_route_status", {
                "show_unrouted": True,
                "show_errors": True,
                "max_nets": 20
            }, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")

            print("\n" + "="*70)
            print("POST-OPTIMIZATION ANALYSIS")
            print("="*70)
            
            # ================================================================
            # Step 9: Report timing summary
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 9: Report timing summary")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()

            # ================================================================
            # Step 10: Report worst critical path
            # ================================================================
            await report_worst_critical_path("STEP 10: Report worst critical path")

            # ================================================================
            # Step 11: Report QoR suggestions
            # ================================================================
            await report_qor_suggestions("STEP 11: Report QoR suggestions")

            # ================================================================
            # Step 12: Analyze critical path spread using RapidWright
            # ================================================================
            post_opt_analysis_dcp = Path(self.temp_dir) / "kitta_post_optimization_for_spread.dcp"
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(post_opt_analysis_dcp),
                "force": True
            }, timeout=600.0)
            print(f"\nWrote temporary DCP for post-optimization spread analysis:\n{result}")
            logger.info(f"Temporary post-optimization DCP for spread analysis: {result}")

            await analyze_critical_path_spread(
                "STEP 12: Analyze critical path spread using RapidWright",
                post_opt_analysis_dcp,
                "kitta_post_optimization_critical_paths.json"
            )

            # Disabled for directive experiments: this second QoR-guided pass can be
            # re-enabled after testing the first phys_opt_design directive above.
            # print("\n" + "="*70)
            # print("QOR-GUIDED FINAL OPTIMIZATION")
            # print("="*70)
            #
            # # ================================================================
            # # Step 13: Run phys_opt_design with QoR-suggested directive
            # # ================================================================
            # print("\n" + "-"*60)
            # print("STEP 13: Run phys_opt_design -directive AggressiveExplore")
            # print("-"*60)
            # result = await self.call_vivado_tool("run_tcl", {
            #     "command": "phys_opt_design -directive AggressiveExplore"
            # }, timeout=1800.0)
            # print(f"QoR-guided phys_opt_design result:\n{result}")
            # logger.info(f"QoR-guided phys_opt_design result: {result}")
            #
            # # ================================================================
            # # Step 14: Run route_design with QoR-suggested timing directives
            # # ================================================================
            # print("\n" + "-"*60)
            # print("STEP 14: Run route_design -directive NoTimingRelaxation -tns_cleanup")
            # print("-"*60)
            # result = await self.call_vivado_tool("run_tcl", {
            #     "command": "route_design -directive NoTimingRelaxation -tns_cleanup"
            # }, timeout=3600.0)
            # print(f"QoR-guided route_design result:\n{result}")
            # logger.info(f"QoR-guided route_design result: {result}")
            #
            # result = await self.call_vivado_tool("report_route_status", {
            #     "show_unrouted": True,
            #     "show_errors": True,
            #     "max_nets": 20
            # }, timeout=300.0)
            # print(f"Route status after QoR-guided routing:\n{result[:1500]}...")
            # logger.info(f"Route status after QoR-guided routing: {result}")
            #
            # # ================================================================
            # # Step 15: Report final timing summary after QoR-guided pass
            # # ================================================================
            # print("\n" + "-"*60)
            # print("STEP 15: Report final timing summary after QoR-guided pass")
            # print("-"*60)
            # result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            # print(f"QoR-guided final timing summary (first 2000 chars):\n{result[:2000]}...")
            # logger.info(f"QoR-guided final timing summary: {result}")
            #
            # target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            # if target_wns is not None:
            #     self.final_wns = target_wns
            # else:
            #     self.final_wns = self.parse_wns_from_timing_report(result)
            #
            # self.print_fmax_status("QoR-guided Final", self.final_wns)
            # logger.info(f"QoR-guided final WNS: {self.final_wns} ns")
            # print()

            print("\n" + "="*70)
            print("OUTPUT")
            print("="*70)

            # ================================================================
            # Step 13: Write final DCP and report results
            # ================================================================
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Critical high-fanout nets identified: {len(self.high_fanout_nets)}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"Test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False



    async def run_test_logicnets(self, input_dcp: Path, output_dcp: Path) -> bool:
        """
        Run the pblock-based optimization flow for LogicNets designs.
        
        Steps:
        1. Open the input DCP in Vivado
        2. Report timing in Vivado (Initialize WNS)
        3. Run the Vivado tool extract_critical_path_cells
        4. Run the RapidWright tool analyze_critical_path_spread
        5. Use known-optimal pblock range for LogicNets (SLICE_X55Y60:SLICE_X111Y254)
        6. Unplace the design in Vivado
        7. Create and apply pblock to entire design
        8. Place the design in Vivado
        9. Route the design in Vivado
        10. Report timing in Vivado (compare against initial WNS)
        """
        pblock_ranges = "SLICE_X55Y60:SLICE_X111Y254"
        
        print("\n" + "="*70)
        print("FPGA OPTIMIZER TEST MODE - LOGICNETS PBLOCK FLOW")
        print("="*70)
        print(f"Input DCP:  {input_dcp}")
        print(f"Output DCP: {output_dcp}")
        print(f"Temp dir:   {self.temp_dir}")
        print("="*70 + "\n")
        
        overall_start = time.time()
        
        try:
            # ================================================================
            # Step 0: Initialize RapidWright (Vivado starts automatically)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 0: Initialize RapidWright")
            print("-"*60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            print(f"RapidWright init result:\n{result[:500]}...")
            logger.info(f"RapidWright init result: {result}")
            
            # ================================================================
            # Step 1: Open the input DCP in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 1: Open input DCP in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"Open checkpoint result:\n{result}")
            logger.info(f"Open checkpoint result: {result}")
            
            # ================================================================
            # Step 2: Report timing in Vivado (Initialize WNS)
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 2: Report initial timing in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Initial timing summary: {result}")
            
            # Get clock period for fmax calculation (also detects target clock)
            self.clock_period = await self.fetch_clock_period()
            
            # Get WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                self.initial_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Initial", self.initial_wns)
            logger.info(f"Initial WNS: {self.initial_wns} ns")
            print()
            
            # ================================================================
            # Step 3: Extract critical path cells from Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 3: Extract critical path cells")
            print("-"*60)
            
            # Write to a file for efficient data transfer
            critical_paths_file = Path(self.temp_dir) / "critical_paths.json"
            result = await self.call_vivado_tool("extract_critical_path_cells", {
                "num_paths": 50,
                "output_file": str(critical_paths_file)
            }, timeout=600.0)
            print(f"Extract critical paths result:\n{result[:2000]}...")
            logger.info(f"Extract critical paths: {result}")
            
            # ================================================================
            # Step 4: Open DCP in RapidWright and analyze critical path spread
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 4: Analyze critical path spread in RapidWright")
            print("-"*60)
            
            # First, open the DCP in RapidWright
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            print(f"RapidWright read checkpoint result:\n{result}")
            logger.info(f"RapidWright read checkpoint: {result}")
            
            # Analyze critical path spread
            result = await self.call_rapidwright_tool("analyze_critical_path_spread", {
                "input_file": str(critical_paths_file)
            }, timeout=300.0)
            print(f"Critical path spread analysis:\n{result[:3000] if isinstance(result, str) else str(result)[:3000]}...")
            logger.info(f"Critical path spread: {result}")
            
            # Parse the spread analysis result to check if pblock is recommended
            spread_result = result if isinstance(result, str) else str(result)
            pblock_recommended = "spread-out" in spread_result.lower() or "pblock" in spread_result.lower()
            print(f"\n*** Pblock optimization {'RECOMMENDED' if pblock_recommended else 'may not be needed'} ***")
            
            # ================================================================
            # Step 5: Apply pblock constraint for LogicNets
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 5: Apply pblock for LogicNets")
            print("-"*60)
            
            print(f"Using pblock range: {pblock_ranges}")
            
            # ================================================================
            # Step 6: Unplace the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 6: Unplace the design in Vivado")
            print("-"*60)
            
            # Use place_design -unplace to remove all placement
            result = await self.call_vivado_tool("run_tcl", {
                "command": "place_design -unplace"
            }, timeout=300.0)
            print(f"Unplace result:\n{result}")
            logger.info(f"Unplace result: {result}")
            
            # ================================================================
            # Step 7: Create and apply pblock to entire design
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 7: Create and apply pblock to entire design")
            print("-"*60)
            
            result = await self.call_vivado_tool("create_and_apply_pblock", {
                "pblock_name": "pblock_opt",
                "ranges": pblock_ranges,
                "apply_to": "current_design",  # Apply to entire design
                "is_soft": False  # Hard constraint
            }, timeout=300.0)
            print(f"Create and apply pblock result:\n{result}")
            logger.info(f"Create pblock result: {result}")
            
            # ================================================================
            # Step 8: Place the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 8: Place the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("place_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for placement
            print(f"Place design result:\n{result}")
            logger.info(f"Place design: {result}")
            
            # ================================================================
            # Step 9: Route the design in Vivado
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 9: Route the design in Vivado")
            print("-"*60)
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)  # 1 hour timeout for routing
            print(f"Route design result:\n{result}")
            logger.info(f"Route design: {result}")
            
            # Check route status
            result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            print(f"Route status after routing:\n{result[:1500]}...")
            logger.info(f"Route status after routing: {result}")
            
            # ================================================================
            # Step 10: Report timing and compare WNS
            # ================================================================
            print("\n" + "-"*60)
            print("STEP 10: Report final timing")
            print("-"*60)
            
            result = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
            print(f"Final timing summary (first 2000 chars):\n{result[:2000]}...")
            logger.info(f"Final timing summary: {result}")
            
            # Get final WNS for the target clock domain
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                self.final_wns = self.parse_wns_from_timing_report(result)
            
            self.print_fmax_status("Final", self.final_wns)
            logger.info(f"Final WNS: {self.final_wns} ns")
            print()
            
            # ================================================================
            # Write final DCP and report results
            # ================================================================
            self.print_wns_change(self.initial_wns, self.final_wns, self.clock_period)
            
            # Always write the final checkpoint
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            print(f"Write final DCP result:\n{result}")
            
            # ================================================================
            # Summary
            # ================================================================
            elapsed = time.time() - overall_start
            self.print_test_summary(
                title="TEST SUMMARY - LOGICNETS PBLOCK OPTIMIZATION",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Pblock applied: {pblock_ranges}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"LogicNets test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def run_test_vexriscv(self, input_dcp: Path, output_dcp: Path) -> bool:
        """
        Cell re-placement optimization flow for VexRiscv.
        
        Mirrors the script in docs/optimization_example.md:
          Step 1 — Vivado baseline (open, get Fmax, extract critical path pins)
          Step 2 — RapidWright analysis (analyze_net_detour, filter candidates)
          Step 3 — RapidWright optimization (optimize_cell_placement, write DCP)
          Step 4 — Vivado verification (open optimized DCP, route, measure Fmax)
        """
        overall_start = time.time()
        
        try:
            # ==============================================================
            # Step 1: Vivado baseline
            # ==============================================================
            print("=" * 60)
            print("Step 1  Vivado baseline")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"Open checkpoint result: {result}")
            
            self.clock_period = await self.fetch_clock_period()
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.initial_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.initial_wns = self.parse_wns_from_timing_report(ts)
            
            baseline_fmax = self.calculate_fmax(self.initial_wns, self.clock_period)
            print(f"  Clock period:   {self.clock_period} ns")
            print(f"  Baseline WNS:   {self.initial_wns} ns")
            if baseline_fmax is not None:
                print(f"  Baseline Fmax:  {baseline_fmax:.2f} MHz")
            
            pins_file = Path(self.temp_dir) / "critical_path_pins.json"
            result = await self.call_vivado_tool("extract_critical_path_pins", {
                "num_paths": 10,
                "output_file": str(pins_file)
            }, timeout=600.0)
            
            critical_paths = json.loads(Path(pins_file).read_text()) if pins_file.exists() else json.loads(result)
            print(f"  Extracted {len(critical_paths)} critical path pin lists")
            
            # ==============================================================
            # Step 2: RapidWright analysis
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 2  RapidWright analysis")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("initialize_rapidwright", {
                "jvm_max_memory": "8G"
            }, timeout=120.0)
            logger.info(f"RapidWright init: {result}")
            
            result = await self.call_rapidwright_tool("read_checkpoint", {
                "dcp_path": str(input_dcp.resolve())
            }, timeout=600.0)
            logger.info(f"RapidWright read checkpoint: {result}")
            
            result = await self.call_rapidwright_tool("analyze_net_detour", {
                "input_file": str(pins_file),
                "detour_threshold": 2.0
            }, timeout=300.0)
            logger.info(f"analyze_net_detour: {result}")
            
            analysis = json.loads(result) if isinstance(result, str) else result
            if "error" in analysis:
                raise RuntimeError(f"analyze_net_detour failed: {analysis['error']}")
            candidates = analysis.get("candidates", [])
            print(f"  Cells analyzed: {analysis.get('cells_analyzed', '?')}")
            print(f"  Candidates (detour > 2.0): {len(candidates)}")
            for c in candidates[:5]:
                print(f"    {str(c['cell']):55s}  ratio={c['max_detour_ratio']}")
            
            if not candidates:
                print("\n  No candidates found — nothing to optimize")
                self.final_wns = self.initial_wns
                return True
            
            worst_path_cells = list(set(
                str(c["cell"]) for c in candidates if c.get("path", 0) <= 2
            ))
            if not worst_path_cells:
                worst_path_cells = [str(candidates[0]["cell"])]
            
            print(f"\n  Targeting {len(worst_path_cells)} cells on paths 1-2:")
            for name in worst_path_cells:
                print(f"    {name}")
            
            # ==============================================================
            # Step 3: RapidWright optimization
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 3  RapidWright optimization")
            print("=" * 60)
            
            result = await self.call_rapidwright_tool("optimize_cell_placement", {
                "cell_names": worst_path_cells
            }, timeout=300.0)
            logger.info(f"optimize_cell_placement: {result}")
            
            opt_result = json.loads(result) if isinstance(result, str) else result
            for r in opt_result.get("results", []):
                print(f"  {r['cell']}: {r['status']} — {r['message']}")
            
            rw_output = Path(self.temp_dir) / "vexriscv_rw_optimized.dcp"
            result = await self.call_rapidwright_tool("write_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            print(f"  Wrote {rw_output.name}")
            
            # ==============================================================
            # Step 4: Vivado verification
            # ==============================================================
            print("\n" + "=" * 60)
            print("Step 4  Vivado verification")
            print("=" * 60)
            
            result = await self.call_vivado_tool("open_checkpoint", {
                "dcp_path": str(rw_output)
            }, timeout=600.0)
            logger.info(f"Open optimized checkpoint: {result}")
            
            result = await self.call_vivado_tool("route_design", {
                "directive": "Default"
            }, timeout=3600.0)
            logger.info(f"Route design: {result}")
            
            route_result = await self.call_vivado_tool("report_route_status", {}, timeout=300.0)
            error_match = re.search(r"# of nets with routing errors.*?:\s+(\d+)", route_result)
            error_count = int(error_match.group(1)) if error_match else -1
            
            target_wns = await self.get_wns_for_target_clock(self._call_vivado_for_clock)
            if target_wns is not None:
                self.final_wns = target_wns
            else:
                ts = await self.call_vivado_tool("report_timing_summary", {}, timeout=300.0)
                self.final_wns = self.parse_wns_from_timing_report(ts)
            
            new_fmax = self.calculate_fmax(self.final_wns, self.clock_period)
            
            print(f"  Routing errors:  {error_count}")
            if baseline_fmax is not None and new_fmax is not None:
                print(f"  Baseline WNS:    {self.initial_wns} ns  →  Fmax {baseline_fmax:.2f} MHz")
                print(f"  Optimized WNS:   {self.final_wns} ns  →  Fmax {new_fmax:.2f} MHz")
                delta = new_fmax - baseline_fmax
                print(f"  Fmax improvement: {delta:+.2f} MHz")
            else:
                print(f"  Baseline WNS:  {self.initial_wns} ns")
                print(f"  Optimized WNS: {self.final_wns} ns")
            
            # Write final DCP
            print(f"\nWriting final DCP to: {output_dcp}")
            result = await self.call_vivado_tool("write_checkpoint", {
                "dcp_path": str(output_dcp.resolve()),
                "force": True
            }, timeout=600.0)
            
            # Summary
            elapsed = time.time() - overall_start
            cells_info = ", ".join(worst_path_cells)
            self.print_test_summary(
                title="TEST SUMMARY - VEXRISCV CELL RE-PLACEMENT",
                elapsed_seconds=elapsed,
                initial_wns=self.initial_wns,
                final_wns=self.final_wns,
                clock_period=self.clock_period,
                extra_info=f"Cells re-placed: {cells_info}"
            )
            
            return True
            
        except Exception as e:
            logger.exception(f"VexRiscv test failed with exception: {e}")
            print(f"\n*** TEST FAILED ***")
            print(f"Exception: {type(e).__name__}: {e}")
            return False

    async def cleanup(self):
        """Clean up resources."""
        print("\n[TEST] Cleaning up...")
        await super().cleanup()
        print(f"[TEST] Run directory preserved at: {self.run_dir}")


async def run_test_mode(input_dcp: Path, output_dcp: Path, debug: bool = False, max_nets: int = 5, run_dir: Optional[Path] = None):
    """Run the test mode optimization.
    
    Detects which example DCP is being used and applies the appropriate optimization flow:
    - logicnets_jscl: Pblock-based placement optimization flow
    - vexriscv_re-place: Cell re-placement flow (same recipe as docs/optimization_example.md)
    """
    # Detect which DCP is being used based on filename (substring match:
    # filenames carry version suffixes like "logicnets_jscl_2025.1.dcp", so an
    # exact "== 'logicnets'" check never fires — use `in`).
    dcp_name = input_dcp.name.lower()
    if "logicnets" in dcp_name:
        design_type = "logicnets"
        print(f"[TEST] Detected LogicNets design - using pblock optimization flow")
    elif "vexriscv" in dcp_name:
        design_type = "vexriscv"
        print(f"[TEST] Detected VexRiscv design - using cell re-placement flow")
    else:
        # Unknown DCP: fall back to the cell re-placement recipe (previous behavior).
        design_type = "vexriscv"
        print(f"[TEST] Unknown design '{input_dcp.name}' - defaulting to cell re-placement flow")
    tester = FPGAOptimizerTest(debug=debug, run_dir=run_dir)
    
    try:
        await tester.start_servers()
        
        if design_type == "logicnets":
            success = await tester.run_test_logicnets(input_dcp, output_dcp)
        else:
            success = await tester.run_test_vexriscv(input_dcp, output_dcp)
        
        if success:
            print("\n[TEST] Test completed successfully")
            print(f"\n[TEST] Output files:")
            print(f"[TEST]   Optimized DCP: {output_dcp}")
            print(f"[TEST]   Run directory: {tester.run_dir}")
            return 0
        else:
            print("\n[TEST] Test failed")
            print(f"[TEST] Run directory: {tester.run_dir}")
            return 1
            
    except KeyboardInterrupt:
        print("\n[TEST] Interrupted by user")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 130
    except Exception as e:
        logger.exception(f"Test mode fatal error: {e}")
        print(f"\n[TEST] Fatal error: {e}")
        print(f"[TEST] Run directory: {tester.run_dir}")
        return 1
    finally:
        await tester.cleanup()


async def main():
    parser = argparse.ArgumentParser(
        description="FPGA Design Optimization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dcp_optimizer.py input.dcp
  python dcp_optimizer.py input.dcp --output output.dcp
  python dcp_optimizer.py input.dcp --model anthropic/claude-sonnet-4
  python dcp_optimizer.py input.dcp --debug
  python dcp_optimizer.py fpl26_contest_benchmarks/logicnets_jscl_2025.1.dcp --test
  python dcp_optimizer.py fpl26_contest_benchmarks/vexriscv_re-place_2025.1.dcp --test
        """
    )
    parser.add_argument("input_dcp", type=Path, help="Input design checkpoint (.dcp)")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        dest="output_dcp",
        help="Output optimized checkpoint (.dcp). Default: <input_name>_optimized-<timestamp>.dcp in same directory as input"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENROUTER_API_KEY"),
        help="OpenRouter API key (default: OPENROUTER_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (verbose logging, save intermediate checkpoints)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: run without LLM. Pblock for LogicNets, cell re-placement for VexRiscv (see docs/optimization_example.md)."
    )
    parser.add_argument(
        "--max-nets",
        type=int,
        default=5,
        help="Maximum number of high fanout nets to optimize in test mode (default: 5)"
    )
    parser.add_argument(
        "--pre-opt",
        type=str,
        default="phys_opt,pblock,relocate,pinopt,reimpl,routeopt",
        help="Comma-separated pipeline steps to run (in order). Supported: phys_opt, pinopt, llm, "
             "relocate, pblock, cell_replace, reimpl, retime, relocate_rw, none. cell_replace and "
             "relocate_rw (RapidWright netlist round-trips) are NO LONGER in the default -- they "
             "broke equivalence (corescore) and never clearly won; select them explicitly if needed. "
             "'llm' is now a "
             "POSITIONABLE step: place it early (e.g. 'phys_opt,llm,relocate,pblock,cell_replace,reimpl') "
             "so the LLM works right after phys_opt and is never squeezed by the slow deterministic stages; "
             "if 'llm' is omitted (and --skip-llm is not set) the LLM still runs as the trailing phase. "
             "'pinopt' (phys_opt_design -critical_pin_opt) is a CHEAP (~2 min) function-preserving pin-swap "
             "pass -- good for large designs where a full phys_opt is too expensive. "
             "'reimpl' (fresh from-scratch place/route) is kept only if it beats the incremental steps. "
             "'retime' is OFF by default (no measured gain, competes for budget). "
             "Default: 'phys_opt,pblock,relocate,cell_replace,pinopt,reimpl'."
    )
    parser.add_argument(
        "--physopt-directive",
        type=str,
        default="",
        help="Optional phys_opt_design directive (e.g. Explore, AggressiveExplore, RuntimeOptimized). Empty = default optimizations."
    )
    parser.add_argument(
        "--pblock-mode",
        choices=["auto", "always", "never"],
        default="always",
        help="When to run the pblock re-placement step: always = attempt on every design "
             "(default; self-reverts if it doesn't help, time-bounded), auto = only if spread "
             "analysis recommends it, never = skip."
    )
    parser.add_argument("--cell-replace-mode", choices=["auto", "always", "never"], default="auto",
                        help="Cell re-placement (detour fix) step: auto = run if high-detour cells exist "
                             "(default), always = force, never = skip.")
    parser.add_argument("--relocate-mode", choices=["auto", "always", "never"], default="always",
                        help="Targeted relocation step: pull a route-bound critical path's movable "
                             "register bank next to its fixed hard-macro anchor (DSP/BRAM/URAM). "
                             "always = attempt every design (default; self-reverts if it doesn't help), "
                             "never = skip.")
    parser.add_argument("--retime-mode", choices=["auto", "always", "never"], default="always",
                        help="Register retiming step (phys_opt_design -retime): rebalances registers "
                             "across logic to shorten logic-depth-bound paths, latency-preserving. "
                             "Gated by an in-stage equivalence check; adopts only if timing improves "
                             "AND equivalence passes. always = attempt (default), never = skip.")
    parser.add_argument("--reimpl-mode", choices=["auto", "always", "never"], default="always",
                        help="Fresh re-implementation fallback stage: re-solve the design from scratch "
                             "(unplace + strong-directive place/phys_opt/route). Runs LAST; adopts only if it "
                             "beats the incremental steps AND passes the equivalence gate. always = attempt "
                             "(default), never = skip.")
    parser.add_argument("--reimpl-place-directive", type=str, default="ExtraTimingOpt",
                        help="place_design directive for the re-impl stage (default: ExtraTimingOpt)")
    parser.add_argument("--reimpl-route-directive", type=str, default="AggressiveExplore",
                        help="route_design directive for the re-impl stage (default: AggressiveExplore)")
    parser.add_argument("--reimpl-skip-if-recovered", type=float, default=0.30,
                        help="Skip the re-impl stage if the incremental stages already recovered "
                             "this FRACTION of the initial negative slack (default: 0.30). Re-impl "
                             "only wins when the incremental result is weak (rosetta: 10%% recovered "
                             "-> re-impl won; amd: 49%% -> lost, 250s wasted), and its runtime is "
                             "scored (gamma). Set 0 to always run it.")
    parser.add_argument("--reimpl-timeout", type=int, default=2400,
                        help="Wall-clock budget (s) for the re-impl fallback stage (default: 2400 = 40 min)")
    parser.add_argument("--phys-opt-timeout", type=int, default=1200,
                        help="Wall-clock budget (s) for the phys_opt stage (default: 1200 = 20 min)")
    parser.add_argument("--manual-timeout", type=int, default=1200,
                        help="Wall-clock budget (s) SHARED by the manual stages (pblock + cell_replace) (default: 1200 = 20 min)")
    parser.add_argument("--llm-timeout", type=int, default=1200,
                        help="Wall-clock budget (s) for the LLM stage (default: 1200 = 20 min)")
    parser.add_argument("--total-timeout", type=int, default=3600,
                        help="Soft cap (s): gates stage STARTS only (default: 3600 = 1 hr)")
    parser.add_argument("--hard-deadline", type=int, default=3420,
                        help="HARD wall-clock kill (s): force-writes the protected-best DCP and "
                             "exits, guaranteeing we finish under the 1-hr contest limit even if a "
                             "Vivado place/route runs away (default: 3420 = 57 min, ~3 min AWS "
                             "safety margin under the 3600s cutoff; 0 disables)")
    parser.add_argument("--cost-cap", type=float, default=1.0,
                        help="Stop the LLM stage before spending more than this many USD (default: 1.0)")
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run only the deterministic pre-LLM optimization and skip the LLM stage (fast, ~$0 cost, safe baseline)."
    )
    parser.add_argument(
        "--no-planner",
        dest="use_planner",
        action="store_false",
        help="Disable the front LLM planner (recipe selection). phys_opt-first + full default "
             "recipe still runs. The planner is ON by default; --skip-llm also disables it."
    )
    parser.set_defaults(use_planner=True)

    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dcp.exists():
        print(f"Error: Input file not found: {args.input_dcp}", file=sys.stderr)
        sys.exit(1)
    
    # Generate default output DCP name if not provided
    if args.output_dcp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        input_stem = args.input_dcp.stem  # Filename without extension
        input_dir = args.input_dcp.parent  # Directory of input file
        args.output_dcp = input_dir / f"{input_stem}_optimized-{timestamp}.dcp"
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory if needed
    args.output_dcp.parent.mkdir(parents=True, exist_ok=True)
    
    # Test mode - run without LLM
    if args.test:
        # Create run directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
        
        print(f"FPGA Design Optimization - TEST MODE")
        print(f"=====================================")
        print(f"Input:       {args.input_dcp.resolve()}")
        print(f"Output:      {args.output_dcp.resolve()}")
        print(f"Run dir:     {run_dir}")
        print(f"Max nets to optimize: {args.max_nets}")
        print()
        
        exit_code = await run_test_mode(
            args.input_dcp, 
            args.output_dcp, 
            debug=args.debug,
            max_nets=args.max_nets,
            run_dir=run_dir
        )
        sys.exit(exit_code)
    
    # Normal mode - requires API key, unless the LLM stage is skipped (baseline-only)
    if not args.api_key and not args.skip_llm:
        print("Error: OpenRouter API key required. Set OPENROUTER_API_KEY or use --api-key", file=sys.stderr)
        print("       Use --skip-llm to run the deterministic baseline without an LLM,", file=sys.stderr)
        print("       or --test for the standalone test mode.", file=sys.stderr)
        sys.exit(1)

    if OpenAI is None:
        print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    
    # Create run directory with timestamp (before creating optimizer so we can show it)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path.cwd() / f"dcp_optimizer_run-{timestamp}"
    
    print(f"FPGA Design Optimization Agent")
    print(f"================================")
    print(f"Input:       {args.input_dcp.resolve()}")
    print(f"Output:      {args.output_dcp.resolve()}")
    print(f"Run dir:     {run_dir}")
    print(f"Model:       {args.model}")
    print(f"Pre-opt:     {args.pre_opt}" + (f" (physopt directive={args.physopt_directive})" if args.physopt_directive else "") + f" [relocate={args.relocate_mode}, retime={args.retime_mode}, pblock={args.pblock_mode}, cell_replace={args.cell_replace_mode}, reimpl={args.reimpl_mode}]")
    print(f"LLM stage:   {'DISABLED (--skip-llm)' if args.skip_llm else 'enabled'}")
    print(f"Budgets:     phys_opt {args.phys_opt_timeout}s | manual(pblock+cell) {args.manual_timeout}s | LLM {args.llm_timeout}s | total(soft) {args.total_timeout}s | HARD-kill {args.hard_deadline}s | cost ${args.cost_cap:.2f}")
    print()

    optimizer = DCPOptimizer(
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
        run_dir=run_dir,
        pre_opt=args.pre_opt,
        physopt_directive=args.physopt_directive,
        pblock_mode=args.pblock_mode,
        cell_replace_mode=args.cell_replace_mode,
        relocate_mode=args.relocate_mode,
        retime_mode=args.retime_mode,
        reimpl_mode=args.reimpl_mode,
        reimpl_place_directive=args.reimpl_place_directive,
        reimpl_route_directive=args.reimpl_route_directive,
        reimpl_skip_if_recovered=args.reimpl_skip_if_recovered,
        skip_llm=args.skip_llm,
        use_planner=args.use_planner,
        phys_opt_timeout=args.phys_opt_timeout,
        manual_timeout=args.manual_timeout,
        reimpl_timeout=args.reimpl_timeout,
        llm_timeout=args.llm_timeout,
        total_timeout=args.total_timeout,
        hard_deadline=args.hard_deadline,
        cost_cap=args.cost_cap
    )
    
    try:
        await optimizer.start_servers()
        success = await optimizer.optimize(args.input_dcp, args.output_dcp)
        
        if success:
            print("\n✓ Optimization completed successfully")
            print(f"\nOutput files:")
            print(f"  Optimized DCP: {args.output_dcp}")
            print(f"  Run directory: {run_dir}")
            sys.exit(0)
        else:
            print("\n✗ Optimization did not complete successfully")
            print(f"\nRun directory: {run_dir}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        print(f"Run directory: {run_dir}")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"Run directory: {run_dir}")
        sys.exit(1)
    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
