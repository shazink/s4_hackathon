"""
Clinical War Room - Audit Logger

Immutable, append-only audit log for human decisions.
"""

import os
import json
from datetime import datetime
from typing import Optional, List
from pathlib import Path
import threading
import fcntl

from hitl.models import HumanDecision
from core.logging import logger


class AuditLogger:
    """
    Append-only audit log for human decisions.
    
    CRITICAL REQUIREMENTS:
    - All writes are append-only
    - No modification of existing records
    - Thread-safe writes
    - Each entry is timestamp-ordered
    """
    
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log = logger.with_context(phase="audit")
        self._lock = threading.Lock()
    
    def _get_log_path(self, date: Optional[datetime] = None) -> Path:
        """Get log file path for a date."""
        if date is None:
            date = datetime.now()
        filename = f"audit_{date.strftime('%Y-%m-%d')}.jsonl"
        return self.log_dir / filename
    
    def log_decision(self, decision: HumanDecision):
        """
        Log a human decision (append-only).
        
        Uses file locking for thread safety.
        """
        log_path = self._get_log_path(decision.timestamp)
        
        entry = {
            "type": "human_decision",
            "logged_at": datetime.now().isoformat(),
            "data": decision.to_dict(),
        }
        
        with self._lock:
            with open(log_path, "a") as f:
                # Use file locking for multi-process safety
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(entry) + "\n")
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        
        self.log.info(
            f"Audit log: {decision.decision_id} - "
            f"{decision.human_action.value} on case {decision.case_id}"
        )
    
    def log_event(self, event_type: str, data: dict):
        """Log a generic audit event."""
        log_path = self._get_log_path()
        
        entry = {
            "type": event_type,
            "logged_at": datetime.now().isoformat(),
            "data": data,
        }
        
        with self._lock:
            with open(log_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(entry) + "\n")
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    
    def get_decisions_for_case(self, case_id: str) -> List[dict]:
        """Retrieve all decisions for a case (read-only)."""
        decisions = []
        
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl")):
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if (
                            entry.get("type") == "human_decision"
                            and entry.get("data", {}).get("case_id") == case_id
                        ):
                            decisions.append(entry["data"])
                    except json.JSONDecodeError:
                        continue
        
        return decisions
    
    def get_recent_decisions(self, limit: int = 100) -> List[dict]:
        """Get most recent decisions."""
        decisions = []
        
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True):
            with open(log_file, "r") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    try:
                        entry = json.loads(line)
                        if entry.get("type") == "human_decision":
                            decisions.append(entry["data"])
                            if len(decisions) >= limit:
                                return decisions
                    except json.JSONDecodeError:
                        continue
        
        return decisions


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_dir: str = "audit_logs") -> AuditLogger:
    """Get or create the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_dir)
    return _audit_logger
