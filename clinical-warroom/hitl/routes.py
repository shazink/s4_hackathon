"""
Clinical War Room - HITL API Routes

FastAPI endpoints for human-in-the-loop review.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends

from hitl.models import (
    ReviewRequest,
    HumanDecision,
    DecisionSubmission,
    DecisionResponse,
    HumanAction,
)
from hitl.store import ReviewStore, get_review_store
from hitl.audit import AuditLogger, get_audit_logger


router = APIRouter(prefix="/human-review", tags=["Human Review"])


def get_store() -> ReviewStore:
    """Dependency to get review store."""
    return get_review_store()


def get_audit() -> AuditLogger:
    """Dependency to get audit logger."""
    return get_audit_logger()


@router.post("/submit", response_model=DecisionResponse)
async def submit_decision(
    submission: DecisionSubmission,
    store: ReviewStore = Depends(get_store),
):
    """
    Submit a human decision for a review request.
    
    Human decisions ALWAYS override system recommendations.
    """
    try:
        # Validate override action if overriding
        if submission.human_action == HumanAction.OVERRIDE:
            if not submission.override_action:
                raise HTTPException(
                    status_code=400,
                    detail="override_action required when human_action is OVERRIDE"
                )
        
        decision = store.submit_decision(submission)
        
        return DecisionResponse(
            success=True,
            decision_id=decision.decision_id,
            final_action=decision.final_action,
            message=f"Decision recorded. Final action: {decision.final_action}",
        )
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit decision: {e}")


@router.get("/pending", response_model=List[dict])
async def get_pending_reviews(
    store: ReviewStore = Depends(get_store),
):
    """
    Get all pending review requests.
    
    Returns list of reviews awaiting human decision.
    """
    pending = store.get_pending_reviews()
    return [r.to_dict() for r in pending]


@router.get("/{case_id}")
async def get_review_for_case(
    case_id: str,
    store: ReviewStore = Depends(get_store),
):
    """
    Get review information for a specific case.
    
    Returns both pending review and any past decisions.
    """
    pending = store.get_review_by_case(case_id)
    decision = store.get_decision_for_case(case_id)
    
    return {
        "case_id": case_id,
        "pending_review": pending.to_dict() if pending else None,
        "latest_decision": decision.to_dict() if decision else None,
        "has_pending": pending is not None,
        "has_decision": decision is not None,
    }


@router.get("/{case_id}/history")
async def get_case_history(
    case_id: str,
    audit: AuditLogger = Depends(get_audit),
):
    """
    Get full audit history for a case.
    
    Returns all human decisions in chronological order.
    """
    decisions = audit.get_decisions_for_case(case_id)
    
    return {
        "case_id": case_id,
        "decision_count": len(decisions),
        "decisions": decisions,
    }


@router.get("/request/{request_id}")
async def get_review_request(
    request_id: str,
    store: ReviewStore = Depends(get_store),
):
    """
    Get a specific review request by ID.
    """
    request = store.get_review(request_id)
    
    if not request:
        raise HTTPException(status_code=404, detail=f"Review {request_id} not found")
    
    return {
        "request": request.to_dict(),
        "case_summary": request.case_summary.model_dump(),
    }
