"""
Clinical War Room - Structured Logging

Provides consistent, structured logging across all modules.
All logs include context for traceability and debugging.
"""

import logging
import sys
import json
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class LogContext:
    """Structured context for log entries."""
    case_id: Optional[str] = None
    agent_name: Optional[str] = None
    phase: Optional[str] = None
    action: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter for production."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        
        # Add context if present
        if hasattr(record, "context") and record.context:
            log_entry["context"] = record.context
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class ReadableFormatter(logging.Formatter):
    """Human-readable formatter for development."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        base = f"{color}[{timestamp}] {record.levelname:8}{self.RESET} {record.module}: {record.getMessage()}"
        
        if hasattr(record, "context") and record.context:
            base += f" | {record.context}"
        
        return base


class WarRoomLogger:
    """
    Context-aware logger for Clinical War Room.
    
    Provides structured logging with support for:
    - Case tracking
    - Agent identification
    - Phase tracking
    - Decision audit trails
    """
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.context = LogContext()
        
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Configure console handler."""
        handler = logging.StreamHandler(sys.stdout)
        
        # Use readable format for now (switch to structured in production)
        handler.setFormatter(ReadableFormatter())
        
        self.logger.addHandler(handler)
    
    def with_context(self, **kwargs) -> "WarRoomLogger":
        """Return logger with updated context."""
        new_logger = WarRoomLogger.__new__(WarRoomLogger)
        new_logger.logger = self.logger
        new_logger.context = LogContext(**{**asdict(self.context), **kwargs})
        return new_logger
    
    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Internal log method with context injection."""
        extra = {"context": self.context.to_dict()}
        self.logger.log(level, msg, *args, extra=extra, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    # Domain-specific logging methods
    
    def case_submitted(self, case_id: str, patient_id: str) -> None:
        """Log case submission."""
        self.with_context(case_id=case_id, action="case_submitted").info(
            f"New case submitted for patient {patient_id}"
        )
    
    def agent_analysis(self, agent_name: str, case_id: str, confidence: float) -> None:
        """Log agent analysis completion."""
        self.with_context(
            case_id=case_id, 
            agent_name=agent_name, 
            action="analysis_complete"
        ).info(f"Analysis complete with confidence {confidence:.2f}")
    
    def decision_made(self, case_id: str, decision: str, confidence: float) -> None:
        """Log final decision."""
        self.with_context(
            case_id=case_id, 
            action="decision_made"
        ).info(f"Decision: {decision} (confidence: {confidence:.2f})")
    
    def safety_override(self, case_id: str, rule: str, reason: str) -> None:
        """Log safety rule override."""
        self.with_context(
            case_id=case_id, 
            action="safety_override"
        ).warning(f"Safety override triggered by {rule}: {reason}")


# Default logger instance
logger = WarRoomLogger("clinical_warroom")
