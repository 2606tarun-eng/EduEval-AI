from pydantic import BaseModel, Field, model_validator
from typing import Literal
from enum import Enum


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvaluationCategory(str, Enum):
    """Strict enumeration of valid evaluation outcomes."""
    correct = "correct"
    contradictory = "contradictory"
    incorrect = "incorrect"


# ---------------------------------------------------------------------------
# Nested Models
# ---------------------------------------------------------------------------

class CategoryProbabilities(BaseModel):
    """
    Confidence scores for each evaluation category.
    All three values MUST sum to 1.0 (validated automatically).
    """
    correct: float = Field(..., ge=0.0, le=1.0, description="Confidence that the answer is correct.")
    contradictory: float = Field(..., ge=0.0, le=1.0, description="Confidence that the answer is contradictory.")
    incorrect: float = Field(..., ge=0.0, le=1.0, description="Confidence that the answer is incorrect.")

    @model_validator(mode="after")
    def probabilities_must_sum_to_one(self) -> "CategoryProbabilities":
        total = round(self.correct + self.contradictory + self.incorrect, 6)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Probabilities must sum to 1.0, but got {total:.6f}. "
                f"(correct={self.correct}, contradictory={self.contradictory}, incorrect={self.incorrect})"
            )
        return self


# ---------------------------------------------------------------------------
# Request Schema
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    """
    Payload for the /evaluate endpoint.
    `reference_answer` is optional — when omitted, the model evaluates
    the student's answer purely on semantic coherence and accuracy.
    """
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The exam question posed to the student.",
        examples=["What is Newton's Second Law of Motion?"],
    )
    reference_answer: str | None = Field(
        default=None,
        max_length=5000,
        description="The ideal/model answer provided by the instructor (optional).",
        examples=["Force equals mass times acceleration (F = ma)."],
    )
    student_answer: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The raw answer text submitted by the student.",
        examples=["F = ma, which means force is mass multiplied by acceleration."],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "What is Newton's Second Law of Motion?",
                "reference_answer": "Force equals mass times acceleration (F = ma).",
                "student_answer": "F = ma, which means force is mass multiplied by acceleration.",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response Schema
# ---------------------------------------------------------------------------

class EvaluationResponse(BaseModel):
    """
    Structured result returned by the /evaluate endpoint.
    """
    category: EvaluationCategory = Field(
        ...,
        description="Final classification: 'correct', 'contradictory', or 'incorrect'.",
    )
    probabilities: CategoryProbabilities = Field(
        ...,
        description="Per-category confidence scores. Must sum to 1.0.",
    )
    reasoning: str = Field(
        ...,
        min_length=10,
        max_length=3000,
        description="Semantic feedback explaining the evaluation decision.",
        examples=[
            "The student correctly identified F = ma and provided an accurate verbal explanation "
            "that aligns with the reference answer."
        ],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "category": "correct",
                "probabilities": {
                    "correct": 0.92,
                    "contradictory": 0.05,
                    "incorrect": 0.03,
                },
                "reasoning": (
                    "The student correctly identified F = ma and provided an accurate verbal "
                    "explanation that aligns with the reference answer."
                ),
            }
        }
    }
