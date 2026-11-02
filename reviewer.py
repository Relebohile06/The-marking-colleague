# reviewer.py

from typing import Literal
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


# ==========================================
# 1. DATA STRUCTURES
# ==========================================

class CriterionEvaluation(BaseModel):
    criterion_name: str
    assigned_score: float = Field(ge=0)
    max_score: float = Field(gt=0)
    reasoning: str


class MarkerOutput(BaseModel):
    essay_name: str
    criteria_scores: list[CriterionEvaluation]
    total_score: float = Field(ge=0)


class ReviewerOutput(BaseModel):
    verdict: Literal["approved", "send_back"]

    flagged_errors: list[str] = Field(default_factory=list)

    correction_instructions: str = ""


# ==========================================
# 2. REVIEWER AGENT INSTRUCTIONS
# ==========================================

REVIEWER_SYSTEM_INSTRUCTION = """
You are an independent academic essay marking auditor.

Your responsibility is to review the work produced by another AI
agent called the Marker.

You must independently evaluate the Marker’s decisions.

You will receive:
1. The original marking rubric.
2. The original learner essay.
3. The Marker’s scores and explanations.

YOUR RESPONSIBILITIES:

1. Examine every marking criterion against the rubric.
2. Check whether the awarded score is justified by evidence
   in the learner's actual essay.
3. Check whether the Marker has misunderstood the rubric.
4. Identify incorrect, inconsistent or unsupported scores.
5. Check whether the reasoning accurately represents the essay.
6. Check whether the total score is mathematically correct.
7. Identify missing criteria or incomplete evaluations.

DECISION RULES:

Return "approved" ONLY when the marking is sufficiently
consistent with the rubric and supported by essay evidence.

Return "send_back" when:
- A score is incorrect.
- A score is not supported by the essay.
- The reasoning contradicts the essay.
- A criterion has been overlooked.
- The total score is incorrect.
- The marking does not follow the rubric.

When sending work back:
- List the specific errors.
- Identify the affected criteria.
- Explain what needs to be corrected.
- Give actionable instructions to the Marker.

IMPORTANT:

Do not approve work simply because the Marker sounds confident.

Do not invent evidence that is not present in the essay.

Evaluate the original essay, not the Marker’s confidence.

Be fair, objective and consistent.

Return your response using the required structured output.
"""


# ==========================================
# 3. REVIEWER AGENT FUNCTION
# ==========================================

def run_reviewer_agent(
    client: genai.Client,
    model_name: str,
    rubric_text: str,
    essay_text: str,
    marker_output: MarkerOutput
) -> ReviewerOutput:

    """
    Runs the Reviewer AI agent.

    Returns:
        ReviewerOutput containing:
        - verdict
        - flagged_errors
        - correction_instructions
    """

    # Convert Marker output into readable structured data.
    marker_data = marker_output.model_dump_json(indent=2)

    prompt = f"""
    ORIGINAL MARKING RUBRIC:
    {rubric_text}

    ORIGINAL LEARNER ESSAY:
    {essay_text}

    MARKER'S EVALUATION:
    {marker_data}

    Review the marking independently.

    Check every criterion against the original rubric
    and the actual essay.

    Return your verdict and detailed correction instructions.
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=REVIEWER_SYSTEM_INSTRUCTION,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=ReviewerOutput
        )
    )

    # Extract structured response.
    if response.parsed is not None:
        review = ReviewerOutput.model_validate(response.parsed)
    elif response.text:
        review = ReviewerOutput.model_validate_json(response.text)
    else:
        raise RuntimeError(
            "Reviewer returned an empty response."
        )

    # Prevent an empty rejection from reaching the loop.
    if review.verdict == "send_back":
        if not review.flagged_errors and not review.correction_instructions:
            review.flagged_errors.append(
                "Reviewer rejected the marking without specifying errors."
            )

    return review


# ==========================================
# 4. BASIC STRUCTURAL VALIDATION
# ==========================================

def validate_marker_output(
    marker_output: MarkerOutput
) -> list[str]:

    """
    Checks objective structural and mathematical errors
    before the Reviewer makes its independent assessment.
    """

    errors = []

    if not marker_output.criteria_scores:
        errors.append("No marking criteria were provided.")

    criterion_names = [
        item.criterion_name.strip().lower()
        for item in marker_output.criteria_scores
    ]

    if len(criterion_names) != len(set(criterion_names)):
        errors.append("Duplicate marking criteria were found.")

    calculated_total = 0.0

    for criterion in marker_output.criteria_scores:

        if criterion.assigned_score > criterion.max_score:
            errors.append(
                f"{criterion.criterion_name}: awarded score "
                f"exceeds the maximum possible score."
            )

        calculated_total += criterion.assigned_score

        if not criterion.reasoning.strip():
            errors.append(
                f"{criterion.criterion_name}: missing reasoning."
            )

    if abs(calculated_total - marker_output.total_score) > 0.01:
        errors.append(
            f"Total score mismatch. Expected {calculated_total}, "
            f"but Marker returned {marker_output.total_score}."
        )

    return errors
