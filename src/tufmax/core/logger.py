"""
TUFMax Provenance Logger Module.

This module implements a thread-safe, fail-safe singleton logger designed for 
scientific auditing. It captures the full execution lineage, including model 
selections, fallback triggers, and validation outcomes, ensuring reproducibility.

Key Features:
    - Singleton Pattern: Ensures a single log file per execution context.
    - Hierarchical Indentation: Visualizes the call stack depth.
    - Fallback Tracking: Explicitly logs model degradation events.
    - Fail-Safe I/O: Prevents logging errors from crashing the main simulation.
    - Context Manager Support: Guarantees file closure even on unexpected crashes.
"""

import os
import sys
import threading
import logging
from datetime import datetime, timezone
from typing import Optional, Any, List, Dict
from pathlib import Path

# Import custom exceptions if needed for specific handling
from ..exceptions import TUFMaxDegradationWarning


class TUFMaxLogger:
    """
    Centralized Provenance Logger for TUFMax.
    
    Designed to create a human-readable, structured audit trail of the entire 
    simulation pipeline. It uses a singleton pattern to ensure all modules 
    write to the same unique file per run.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Thread-safe Singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TUFMaxLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize internal state. Actual file setup happens in initialize_run."""
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._file_handler = None
        self._start_time = None
        self._indent_level = 0
        self._indent_stack = []
        self._log_path = None
        self._models_used = {}
        self._fallback_count = 0
        self._initialized = False
        self._lock = threading.Lock() # Lock for writing

    def initialize_run(self, ground_pos: Dict[str, Any], sat_pos: Dict[str, Any], 
                       config: Dict[str, Any], log_dir: str = "./logs"):
        """
        Initialize the log file with unique naming based on geometry and time.
        
        Args:
            ground_pos: Dict with 'lat', 'lon', 'alt' of ground station.
            sat_pos: Dict with 'lat', 'lon', 'alt' of satellite.
            config: User configuration dictionary.
            log_dir: Directory to store logs.
        """
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        g_lat = f"{abs(ground_pos['lat']):.1f}{'N' if ground_pos['lat']>=0 else 'S'}"
        g_lon = f"{abs(ground_pos['lon']):.1f}{'E' if ground_pos['lon']>=0 else 'W'}"
        s_alt = f"{sat_pos['alt']/1000:.1f}km" # Convert m to km for filename
        
        filename = f"TUFMax_{timestamp}_Est{g_lat}{g_lon}_Sat{s_alt}.log"
        self._log_path = Path(log_dir) / filename
        
        try:
            self._file_handler = open(self._log_path, 'w', encoding='utf-8')
            self._start_time = datetime.now(timezone.utc)
            self._write_header(ground_pos, sat_pos, config)
            self._initialized = True
        except Exception as e:
            print(f"CRITICAL: Failed to initialize log file at {self._log_path}: {e}", file=sys.stderr)
            self._file_handler = None # Disable logging to file, fallback to stderr if needed

    def _write_header(self, ground_pos, sat_pos, config):
        """Write the standardized header box."""
        header_lines = [
            "=" * 80,
            "TUFMax Execution Log | Atmospheric & Plasma Profile Generator",
            f"Start Time: {self._start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Ground Station: Lat={ground_pos['lat']:.2f}, Lon={ground_pos['lon']:.2f}, Alt={ground_pos['alt']}m",
            f"Satellite Pos:  Lat={sat_pos['lat']:.2f}, Lon={sat_pos['lon']:.2f}, Alt={sat_pos['alt']}m",
            f"Configuration:  {config.get('summary', 'Standard Mode')}",
            "=" * 80,
            ""
        ]
        self._safe_write_lines(header_lines)

    def _safe_write_lines(self, lines: List[str]):
        """Thread-safe and fail-safe write operation."""
        if not self._file_handler:
            return
            
        with self._lock:
            try:
                for line in lines:
                    self._file_handler.write(line + "\n")
                self._file_handler.flush() # Ensure immediate write to disk
            except Exception as e:
                print(f"Logging Error: {e}", file=sys.stderr)

    def _get_relative_time(self) -> str:
        """Calculate elapsed time since start."""
        if not self._start_time:
            return "+0.00s"
        delta = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return f"+{delta:.2f}s"

    def _sanitize_msg(self, msg: Any) -> str:
        """Convert objects (like astropy Quantity) to safe strings."""
        if hasattr(msg, 'value'): # Handle astropy.units.Quantity
            return f"{msg.value} {msg.unit}"
        if hasattr(msg, '__len__') and len(msg) > 50 and not isinstance(msg, str):
            return f"{type(msg).__name__}[shape={getattr(msg, 'shape', 'unknown')}]"
        return str(msg)

    def write(self, message: str, level: int = 0, status: str = "INFO"):
        """
        Write a formatted log entry.
        
        Args:
            message: The message content.
            level: Indentation level (0 = root).
            status: Status tag (INFO, OK, WARN, FAIL, FBACK, CRIT).
        """
        if not self._initialized:
            return

        rel_time = self._get_relative_time()
        prefix = "  " * level + ("+-- " if level > 0 else "")
        clean_msg = self._sanitize_msg(message)
        
        # Format: [+1.23s] [STATUS] +-- Message
        line = f"[{rel_time}] [{status:<6}] {prefix}{clean_msg}"
        
        self._safe_write_lines([line])
        
        # Optional: Echo to console if verbose (could be controlled by config)
        # print(line)

    def log_module_start(self, module_name: str):
        """Log the start of a major block and increase indentation."""
        current_level = len(self._indent_stack)
        self.write(f"Starting Module: {module_name}", level=current_level, status="INFO")
        self._indent_stack.append(module_name)

    def log_module_end(self, module_name: str, success: bool = True):
        """Log the end of a major block and decrease indentation."""
        if self._indent_stack:
            self._indent_stack.pop()
        
        current_level = len(self._indent_stack)
        status = "OK" if success else "FAIL"
        self.write(f"Completed Module: {module_name}", level=current_level, status=status)

    def log_fallback(self, from_model: str, to_model: str, reason: str):
        """Explicitly log a model degradation event."""
        self._fallback_count += 1
        current_level = len(self._indent_stack)
        
        self.write("FALLBACK TRIGGERED", level=current_level, status="FBACK")
        self.write(f"Source: {from_model} -> Target: {to_model}", level=current_level + 1, status="INFO")
        self.write(f"Reason: {reason}", level=current_level + 1, status="INFO")

    def log_validation(self, report: Any):
        """Log results from the Sanity Check Validator."""
        current_level = len(self._indent_stack)
        status_map = {
            "VALID": "OK",
            "WARNING": "WARN",
            "CRITICAL": "CRIT"
        }
        
        status_tag = status_map.get(getattr(report, 'status', 'UNKNOWN'), "INFO")
        msg_prefix = "Validation Passed" if report.status == "VALID" else \
                     ("Validation Warning" if report.status == "WARNING" else "Validation Failed")
        
        self.write(f"{msg_prefix}: {getattr(report, 'check_name', 'Check')}", 
                   level=current_level, status=status_tag)
        
        if hasattr(report, 'message') and report.message:
            self.write(f"Details: {report.message}", level=current_level + 1, status="INFO")
            
    def log_warning(self, message: str):
        """Log a warning message."""
        current_level = len(self._indent_stack)
        self.write(message, level=current_level, status="WARN")

    def log_error(self, message: str):
        """Log a critical error message."""
        current_level = len(self._indent_stack)
        self.write(message, level=current_level, status="CRIT")

    def register_model(self, block_id: int, model_name: str):
        """Track which model was finally used for a specific block."""
        self._models_used[block_id] = model_name

    def finalize(self, total_time: Optional[float] = None, final_models_used: Optional[Dict[int, str]] = None):
        """Write the summary footer and close the file."""
        if not self._file_handler:
            return

        end_time = datetime.now(timezone.utc)
        # Handle case where total_time is already a string (e.g., "FAILED")
        if isinstance(total_time, str):
            duration_str = total_time
        else:
            duration = total_time or (end_time - self._start_time).total_seconds()
            duration_str = f"{duration:.2f}s"

        footer_lines = [
            "",
            "=" * 80,
            "EXECUTION SUMMARY",
            f"Total Duration: {duration_str}",
            f"Fallback Events: {self._fallback_count}",
            "Final Models Chain:"
        ]
        
        # Use provided models or fallback to internal state
        models_to_log = final_models_used if final_models_used is not None else self._models_used
        
        for blk_id in sorted(models_to_log.keys()):
            footer_lines.append(f"  Block {blk_id}: {models_to_log[blk_id]}")
            
        footer_lines.extend([
            "Status: COMPLETED" if "FAILED" not in duration_str else "Status: FAILED",
            "=" * 80
        ])
        
        self._safe_write_lines(footer_lines)
        
        try:
            self._file_handler.close()
            self._initialized = False
            # Only print if not in silent mode or if failed
            if "FAILED" in duration_str:
                print(f"Log file closed (FAILED): {self._log_path}")
        except Exception as e:
            print(f"Error closing log file: {e}", file=sys.stderr)

    def __enter__(self):
        """Context Manager Entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Exit: Ensures cleanup even on exception."""
        if exc_type is not None:
            self._safe_write_lines([
                "",
                "=" * 80,
                f"FATAL ERROR: {exc_type.__name__}",
                f"Message: {exc_val}",
                "Status: ABORTED",
                "=" * 80
            ])
        self.finalize()
        return False # Do not suppress the exception
