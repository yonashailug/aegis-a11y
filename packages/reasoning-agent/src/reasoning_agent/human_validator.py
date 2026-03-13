"""
Human-in-the-Loop Validation Interface

Based on technical paper Section 4.6: Implements validation interfaces
for educator review and correction of AI-generated accessibility metadata.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .quality_assessor import QualityMetrics
from .schemas import ReasoningOutput, SubjectArea
from .verifier import VerificationResult


class ReviewAction(str, Enum):
    """Actions available during human review."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    REQUEST_REGENERATION = "request_regeneration"
    ESCALATE = "escalate"


class ReviewerRole(str, Enum):
    """Types of human reviewers."""

    EDUCATOR = "educator"  # Subject matter expert
    ACCESSIBILITY_SPECIALIST = "accessibility_specialist"  # A11y expert
    ADMINISTRATOR = "administrator"  # Oversight role
    STUDENT = "student"  # End-user feedback


@dataclass
class ReviewFeedback:
    """Feedback provided by human reviewer."""

    reviewer_id: str
    reviewer_role: ReviewerRole
    action: ReviewAction
    confidence_rating: float  # 1-5 scale
    quality_rating: float  # 1-5 scale

    # Detailed feedback
    pedagogical_accuracy: float | None = None
    accessibility_compliance: float | None = None
    subject_appropriateness: float | None = None

    # Text feedback
    comments: str = ""
    suggested_improvements: list[str] = None
    corrected_alt_text: str | None = None

    # Metadata
    review_duration: float = 0.0
    timestamp: datetime = None

    def __post_init__(self):
        if self.suggested_improvements is None:
            self.suggested_improvements = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ReviewSession:
    """Complete review session for an element."""

    element_id: str
    original_output: ReasoningOutput
    quality_metrics: QualityMetrics
    verification_result: VerificationResult

    # Review data
    reviews: list[ReviewFeedback]
    final_action: ReviewAction | None = None
    consensus_reached: bool = False
    requires_expert_review: bool = False

    # Session metadata
    session_id: str = ""
    created_at: datetime = None
    completed_at: datetime | None = None
    priority_level: str = "normal"  # low, normal, high, critical

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if not self.session_id:
            self.session_id = (
                f"review_{self.element_id}_{int(self.created_at.timestamp())}"
            )


class HumanValidator:
    """
    Human-in-the-Loop validation system for accessibility remediation.

    Provides interfaces for educators and accessibility specialists to review,
    validate, and correct AI-generated pedagogical alt-text and metadata.
    """

    def __init__(self, callback_handler: Callable | None = None):
        """
        Initialize human validation system.

        Args:
            callback_handler: Optional callback for external system integration
        """
        self.callback_handler = callback_handler
        self.active_sessions: dict[str, ReviewSession] = {}
        self.completed_sessions: list[ReviewSession] = []

        # Review routing rules (simplified for now)
        self.routing_rules = {
            "high_priority": lambda session: session.priority_level
            in ["high", "critical"],
            "accessibility_expert": lambda session: session.verification_result.wcag_pass_rate
            < 0.8,
            "subject_expert": lambda session: session.quality_metrics.subject_relevance
            < 0.7,
            "escalation": lambda session: len(session.reviews) > 2
            and not session.consensus_reached,
        }

        # Quality thresholds for automatic routing
        self.quality_thresholds = {
            "auto_approve": 4.5,  # High quality, minimal review
            "standard_review": 3.0,  # Standard educator review
            "expert_review": 2.0,  # Requires specialist attention
            "reject_threshold": 1.5,  # Likely needs regeneration
        }

        self.initialized = True

    def create_review_session(
        self,
        reasoning_output: ReasoningOutput,
        quality_metrics: QualityMetrics,
        verification_result: VerificationResult,
        priority_override: str | None = None,
    ) -> ReviewSession:
        """
        Create a new human review session.

        Args:
            reasoning_output: AI-generated output to review
            quality_metrics: Quality assessment results
            verification_result: WCAG verification results
            priority_override: Manual priority setting

        Returns:
            ReviewSession configured for human validation
        """

        # Determine priority level
        priority = priority_override or self._calculate_priority(
            quality_metrics, verification_result
        )

        # Create review session
        session = ReviewSession(
            element_id=reasoning_output.element_id,
            original_output=reasoning_output,
            quality_metrics=quality_metrics,
            verification_result=verification_result,
            reviews=[],
            priority_level=priority,
        )

        # Determine if expert review is required
        session.requires_expert_review = self._check_expert_review_required(
            quality_metrics, verification_result
        )

        # Store active session
        self.active_sessions[session.session_id] = session

        return session

    def submit_review(
        self, session_id: str, feedback: ReviewFeedback
    ) -> dict[str, Any]:
        """
        Submit human review feedback for a session.

        Args:
            session_id: ID of the review session
            feedback: ReviewFeedback from human reviewer

        Returns:
            Dict with review status and next actions
        """

        if session_id not in self.active_sessions:
            return {"error": "Session not found", "session_id": session_id}

        session = self.active_sessions[session_id]
        session.reviews.append(feedback)

        # Check if consensus is reached or session can be finalized
        consensus_result = self._check_consensus(session)

        if consensus_result["consensus_reached"]:
            session.consensus_reached = True
            session.final_action = consensus_result["final_action"]
            session.completed_at = datetime.now()

            # Move to completed sessions
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]

            # Trigger callback if configured
            if self.callback_handler:
                self.callback_handler("review_completed", session)

        return {
            "status": "submitted",
            "session_id": session_id,
            "consensus_reached": session.consensus_reached,
            "requires_additional_review": not session.consensus_reached,
            "next_reviewer_type": consensus_result.get("next_reviewer_type"),
            "review_count": len(session.reviews),
        }

    def get_review_queue(
        self,
        reviewer_role: ReviewerRole,
        subject_filter: SubjectArea | None = None,
        priority_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get pending reviews for a specific reviewer type.

        Args:
            reviewer_role: Role of the requesting reviewer
            subject_filter: Optional subject area filter
            priority_filter: Optional priority level filter

        Returns:
            List of review sessions assigned to this reviewer type
        """

        queue = []

        for session in self.active_sessions.values():
            # Check if this session needs this type of reviewer
            if not self._session_needs_reviewer(session, reviewer_role):
                continue

            # Apply filters
            if (
                subject_filter
                and session.original_output.detected_subject_area != subject_filter
            ):
                continue

            if priority_filter and session.priority_level != priority_filter:
                continue

            # Create queue item
            queue_item = {
                "session_id": session.session_id,
                "element_id": session.element_id,
                "subject_area": session.original_output.detected_subject_area.value,
                "priority": session.priority_level,
                "created_at": session.created_at.isoformat(),
                "quality_score": session.quality_metrics.pedagogical_score,
                "wcag_pass_rate": session.verification_result.wcag_pass_rate,
                "requires_expert_review": session.requires_expert_review,
                "current_review_count": len(session.reviews),
                "estimated_review_time": self._estimate_review_time(session),
            }

            queue.append(queue_item)

        # Sort by priority and creation time
        priority_order = {"critical": 0, "high": 1, "normal": 2, "low": 3}
        queue.sort(
            key=lambda x: (priority_order.get(x["priority"], 2), x["created_at"])
        )

        return queue

    def get_review_interface_data(self, session_id: str) -> dict[str, Any]:
        """
        Get all data needed for the review interface.

        Args:
            session_id: ID of the review session

        Returns:
            Dict with complete review interface data
        """

        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        session = self.active_sessions[session_id]

        return {
            "session": {
                "id": session.session_id,
                "element_id": session.element_id,
                "priority": session.priority_level,
                "created_at": session.created_at.isoformat(),
                "requires_expert_review": session.requires_expert_review,
            },
            "original_element": (
                session.original_output.extracted_element
                if hasattr(session.original_output, "extracted_element")
                else {}
            ),
            "ai_output": {
                "subject_area": session.original_output.detected_subject_area.value,
                "subject_confidence": session.original_output.subject_confidence,
                "alt_text": session.original_output.pedagogical_alt_text,
                "rationale": session.original_output.alt_text_rationale,
                "importance": session.original_output.contextual_importance,
                "quality_score": session.original_output.pedagogical_quality_score,
                "confidence_level": session.original_output.confidence_level.value,
            },
            "quality_assessment": {
                "pedagogical_score": session.quality_metrics.pedagogical_score,
                "structural_score": session.quality_metrics.structural_score,
                "subject_relevance": session.quality_metrics.subject_relevance,
                "udl_compliance": session.quality_metrics.udl_compliance,
                "accessibility_features": session.quality_metrics.accessibility_features,
                "improvement_suggestions": session.quality_metrics.improvement_suggestions,
                "overall_confidence": session.quality_metrics.overall_confidence,
            },
            "verification_results": {
                "wcag_pass_rate": session.verification_result.wcag_pass_rate,
                "issues": [
                    asdict(issue) for issue in session.verification_result.issues
                ],
                "corrections_applied": session.verification_result.corrections_applied,
                "overall_status": session.verification_result.overall_status.value,
            },
            "previous_reviews": [asdict(review) for review in session.reviews],
            "review_guidelines": self._get_review_guidelines(
                session.original_output.detected_subject_area
            ),
        }

    def generate_review_report(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> dict[str, Any]:
        """
        Generate comprehensive review activity report.

        Args:
            start_date: Report start date (optional)
            end_date: Report end date (optional)

        Returns:
            Dict with review activity statistics and insights
        """

        # Filter completed sessions by date range
        sessions = self.completed_sessions
        if start_date:
            sessions = [
                s for s in sessions if s.completed_at and s.completed_at >= start_date
            ]
        if end_date:
            sessions = [
                s for s in sessions if s.completed_at and s.completed_at <= end_date
            ]

        if not sessions:
            return {"message": "No completed reviews in specified period"}

        # Calculate statistics
        total_sessions = len(sessions)
        avg_quality_score = (
            sum(s.quality_metrics.pedagogical_score for s in sessions) / total_sessions
        )
        avg_wcag_pass_rate = (
            sum(s.verification_result.wcag_pass_rate for s in sessions) / total_sessions
        )

        # Review outcomes
        outcomes = {}
        for session in sessions:
            action = session.final_action.value if session.final_action else "unknown"
            outcomes[action] = outcomes.get(action, 0) + 1

        # Reviewer statistics
        reviewer_stats = {}
        for session in sessions:
            for review in session.reviews:
                role = review.reviewer_role.value
                if role not in reviewer_stats:
                    reviewer_stats[role] = {
                        "count": 0,
                        "avg_rating": 0,
                        "total_rating": 0,
                    }
                reviewer_stats[role]["count"] += 1
                reviewer_stats[role]["total_rating"] += review.quality_rating
                reviewer_stats[role]["avg_rating"] = (
                    reviewer_stats[role]["total_rating"] / reviewer_stats[role]["count"]
                )

        # Subject area breakdown
        subject_stats = {}
        for session in sessions:
            subject = session.original_output.detected_subject_area.value
            if subject not in subject_stats:
                subject_stats[subject] = {
                    "count": 0,
                    "avg_quality": 0,
                    "total_quality": 0,
                }
            subject_stats[subject]["count"] += 1
            subject_stats[subject][
                "total_quality"
            ] += session.quality_metrics.pedagogical_score
            subject_stats[subject]["avg_quality"] = (
                subject_stats[subject]["total_quality"]
                / subject_stats[subject]["count"]
            )

        return {
            "period": {
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "total_sessions": total_sessions,
            },
            "quality_metrics": {
                "average_ai_quality_score": round(avg_quality_score, 2),
                "average_wcag_pass_rate": round(avg_wcag_pass_rate, 2),
                "sessions_requiring_expert_review": len(
                    [s for s in sessions if s.requires_expert_review]
                ),
            },
            "review_outcomes": outcomes,
            "reviewer_statistics": reviewer_stats,
            "subject_area_breakdown": subject_stats,
            "top_improvement_areas": self._identify_improvement_areas(sessions),
        }

    def _calculate_priority(
        self, quality_metrics: QualityMetrics, verification_result: VerificationResult
    ) -> str:
        """Calculate review priority based on quality and verification results."""

        # Critical issues require immediate attention
        critical_issues = len(
            [i for i in verification_result.issues if i.severity.value == "fail"]
        )
        if critical_issues > 2 or quality_metrics.pedagogical_score < 2.0:
            return "critical"

        # High priority for significant quality issues
        if (
            quality_metrics.pedagogical_score < 3.0
            or verification_result.wcag_pass_rate < 0.7
            or quality_metrics.overall_confidence < 0.6
        ):
            return "high"

        # Low priority for high-quality outputs
        if (
            quality_metrics.pedagogical_score >= 4.0
            and verification_result.wcag_pass_rate >= 0.9
            and quality_metrics.overall_confidence >= 0.8
        ):
            return "low"

        return "normal"

    def _check_expert_review_required(
        self, quality_metrics: QualityMetrics, verification_result: VerificationResult
    ) -> bool:
        """Check if expert review is required."""

        return (
            quality_metrics.pedagogical_score < 3.0
            or verification_result.wcag_pass_rate < 0.8
            or verification_result.requires_human_review
            or len(
                [i for i in verification_result.issues if i.severity.value == "fail"]
            )
            > 1
        )

    def _check_consensus(self, session: ReviewSession) -> dict[str, Any]:
        """Check if consensus is reached among reviewers."""

        if not session.reviews:
            return {"consensus_reached": False}

        # Single reviewer - auto-consensus for now
        if len(session.reviews) == 1:
            review = session.reviews[0]
            return {
                "consensus_reached": True,
                "final_action": review.action,
                "confidence": review.confidence_rating,
            }

        # Multiple reviewers - check for agreement
        actions = [r.action for r in session.reviews]
        most_common_action = max(set(actions), key=actions.count)
        action_consensus = actions.count(most_common_action) / len(actions)

        # Require 2/3 consensus
        if action_consensus >= 0.67:
            return {
                "consensus_reached": True,
                "final_action": most_common_action,
                "consensus_strength": action_consensus,
            }

        # No consensus - may need additional review
        return {
            "consensus_reached": False,
            "next_reviewer_type": (
                "accessibility_specialist"
                if any(
                    r.reviewer_role == ReviewerRole.EDUCATOR for r in session.reviews
                )
                else "educator"
            ),
        }

    def _session_needs_reviewer(
        self, session: ReviewSession, reviewer_role: ReviewerRole
    ) -> bool:
        """Check if session needs review from this type of reviewer."""

        # Check if already reviewed by this role type
        existing_roles = [r.reviewer_role for r in session.reviews]

        # Always allow multiple educators to review
        if reviewer_role == ReviewerRole.EDUCATOR:
            return True

        # Accessibility specialist needed for WCAG issues
        if (
            reviewer_role == ReviewerRole.ACCESSIBILITY_SPECIALIST
            and reviewer_role not in existing_roles
            and session.verification_result.wcag_pass_rate < 0.9
        ):
            return True

        # Administrator for escalated cases
        if (
            reviewer_role == ReviewerRole.ADMINISTRATOR
            and len(session.reviews) > 2
            and not session.consensus_reached
        ):
            return True

        return False

    def _estimate_review_time(self, session: ReviewSession) -> int:
        """Estimate review time in minutes."""

        base_time = 5  # 5 minutes base

        # Add time based on complexity
        if session.quality_metrics.pedagogical_score < 3.0:
            base_time += 5  # More time for poor quality

        if len(session.verification_result.issues) > 3:
            base_time += 3  # More time for multiple issues

        if session.requires_expert_review:
            base_time += 7  # Expert reviews take longer

        return base_time

    def _get_review_guidelines(self, subject_area: SubjectArea) -> dict[str, list[str]]:
        """Get subject-specific review guidelines."""

        general_guidelines = [
            "Verify pedagogical accuracy and educational value",
            "Check for appropriate subject-specific vocabulary",
            "Ensure description supports learning objectives",
            "Validate accessibility compliance (WCAG 2.1 AA)",
            "Assess clarity and appropriateness for target audience",
        ]

        subject_specific = {
            SubjectArea.PHYSICS: [
                "Verify accuracy of physical concepts and relationships",
                "Check for proper use of physics terminology",
                "Ensure mathematical relationships are correctly described",
                "Validate units and measurements",
            ],
            SubjectArea.CHEMISTRY: [
                "Verify chemical formulas and reaction accuracy",
                "Check for proper chemical terminology usage",
                "Ensure molecular relationships are correctly described",
                "Validate safety and procedural information",
            ],
            SubjectArea.BIOLOGY: [
                "Verify biological processes and structures accuracy",
                "Check for proper biological terminology",
                "Ensure evolutionary and ecological relationships are correct",
                "Validate anatomical and physiological descriptions",
            ],
            SubjectArea.MATHEMATICS: [
                "Verify mathematical accuracy and relationships",
                "Check for proper mathematical notation and terminology",
                "Ensure problem-solving approaches are sound",
                "Validate geometric and algebraic descriptions",
            ],
        }

        return {
            "general": general_guidelines,
            "subject_specific": subject_specific.get(subject_area, []),
        }

    def _identify_improvement_areas(
        self, sessions: list[ReviewSession]
    ) -> list[dict[str, Any]]:
        """Identify common improvement areas from completed reviews."""

        improvement_areas = {}

        for session in sessions:
            for review in session.reviews:
                for suggestion in review.suggested_improvements:
                    if suggestion not in improvement_areas:
                        improvement_areas[suggestion] = 0
                    improvement_areas[suggestion] += 1

        # Sort by frequency
        sorted_areas = sorted(
            improvement_areas.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {
                "area": area,
                "frequency": count,
                "percentage": (count / len(sessions)) * 100,
            }
            for area, count in sorted_areas[:10]  # Top 10
        ]
