"""
Clinical War Room - Review Store

In-memory store for pending reviews with persistence.
"""

import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import threading

from hitl.models import (
    ReviewRequest,
    ReviewStatus,
    CaseSummary,
    HumanDecision,
    HumanAction,
    SystemAction,
    DecisionSubmission,
)
from hitl.audit import get_audit_logger
from core.logging import logger


class ReviewStore:
    """
    Store for pending and completed reviews.
    
    Provides thread-safe access to review data.
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        self.log = logger.with_context(phase="hitl")
        self._lock = threading.RLock()
        
        # In-memory storage
        self._pending: Dict[str, ReviewRequest] = {}
        self._completed: Dict[str, HumanDecision] = {}
        
        # Persistence
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._load_pending()
    
    def create_review(
        self,
        case_id: str,
        case_summary: CaseSummary,
        review_reason: str,
        is_mandatory: bool = False,
        expires_in: Optional[timedelta] = None,
    ) -> ReviewRequest:
        """Create a new review request."""
        request_id = f"REV-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now()
        
        request = ReviewRequest(
            request_id=request_id,
            case_id=case_id,
            case_summary=case_summary,
            review_reason=review_reason,
            is_mandatory=is_mandatory,
            status=ReviewStatus.PENDING,
            created_at=now,
            expires_at=now + expires_in if expires_in else None,
        )
        
        with self._lock:
            self._pending[request_id] = request
            self._persist_pending()
        
        self.log.info(f"Created review request {request_id} for case {case_id}")
        return request
    
    def get_review(self, request_id: str) -> Optional[ReviewRequest]:
        """Get a review by ID."""
        with self._lock:
            return self._pending.get(request_id)
    
    def get_review_by_case(self, case_id: str) -> Optional[ReviewRequest]:
        """Get pending review for a case."""
        with self._lock:
            for request in self._pending.values():
                if request.case_id == case_id and request.status == ReviewStatus.PENDING:
                    return request
        return None
    
    def get_pending_reviews(self) -> List[ReviewRequest]:
        """Get all pending reviews."""
        with self._lock:
            return [
                r for r in self._pending.values()
                if r.status == ReviewStatus.PENDING
            ]
    
    def submit_decision(
        self,
        submission: DecisionSubmission,
    ) -> HumanDecision:
        """
        Submit a human decision.
        
        This always overrides any system recommendation.
        """
        with self._lock:
            request = self._pending.get(submission.request_id)
            if not request:
                raise ValueError(f"Review request {submission.request_id} not found")
            
            if request.status == ReviewStatus.COMPLETED:
                raise ValueError(f"Review {submission.request_id} already completed")
            
            decision_id = f"DEC-{uuid.uuid4().hex[:8].upper()}"
            
            decision = HumanDecision(
                decision_id=decision_id,
                case_id=submission.case_id,
                request_id=submission.request_id,
                system_action=request.case_summary.system_action,
                human_action=submission.human_action,
                override_action=submission.override_action,
                notes=submission.notes,
                reviewer_id=submission.reviewer_id,
                reviewer_name=submission.reviewer_name,
                timestamp=datetime.now(),
            )
            
            # Update request status
            request.status = ReviewStatus.COMPLETED
            
            # Store decision
            self._completed[decision_id] = decision
            
            # Audit log (immutable)
            get_audit_logger().log_decision(decision)
            
            self._persist_pending()
        
        self.log.info(
            f"Decision {decision_id}: {decision.human_action.value} "
            f"(final: {decision.final_action}) on case {decision.case_id}"
        )
        
        return decision
    
    def get_decision(self, decision_id: str) -> Optional[HumanDecision]:
        """Get a decision by ID."""
        with self._lock:
            return self._completed.get(decision_id)
    
    def get_decision_for_case(self, case_id: str) -> Optional[HumanDecision]:
        """Get the most recent decision for a case."""
        with self._lock:
            for decision in sorted(
                self._completed.values(),
                key=lambda d: d.timestamp,
                reverse=True,
            ):
                if decision.case_id == case_id:
                    return decision
        return None
    
    def _persist_pending(self):
        """Persist pending reviews to disk."""
        if not self.persist_dir:
            return
        
        path = self.persist_dir / "pending_reviews.json"
        data = {
            request_id: request.model_dump(mode="json")
            for request_id, request in self._pending.items()
        }
        
        with open(path, "w") as f:
            json.dump(data, f, default=str)
    
    def _load_pending(self):
        """Load pending reviews from disk."""
        if not self.persist_dir:
            return
        
        path = self.persist_dir / "pending_reviews.json"
        if not path.exists():
            return
        
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            for request_id, request_data in data.items():
                # Convert datetime strings
                request_data["created_at"] = datetime.fromisoformat(request_data["created_at"])
                if request_data.get("expires_at"):
                    request_data["expires_at"] = datetime.fromisoformat(request_data["expires_at"])
                
                # Reconstruct nested objects
                request_data["case_summary"] = CaseSummary(**request_data["case_summary"])
                
                self._pending[request_id] = ReviewRequest(**request_data)
        except Exception as e:
            self.log.warning(f"Failed to load pending reviews: {e}")


# Global store instance
_review_store: Optional[ReviewStore] = None


def get_review_store(persist_dir: Optional[str] = None) -> ReviewStore:
    """Get or create the global review store."""
    global _review_store
    if _review_store is None:
        _review_store = ReviewStore(persist_dir)
    return _review_store
