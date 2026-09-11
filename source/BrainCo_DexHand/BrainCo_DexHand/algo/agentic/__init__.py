"""Small, Isaac-Lab-independent modules for the visual/agentic student.

The modules in this package intentionally only depend on PyTorch.  This makes
it possible to pre-train them from exported RevoLab rollouts and unit-test the
data contract without starting Isaac Sim.
"""

from .visual_student import VisualStudentBatch, VisualLanguageStudent, evidence_loss
from .consequence_predictor import (
    ConsequenceBatch,
    ConsequencePredictor,
    consequence_loss,
    score_action_chunks,
)

__all__ = [
    "VisualStudentBatch",
    "VisualLanguageStudent",
    "evidence_loss",
    "ConsequenceBatch",
    "ConsequencePredictor",
    "consequence_loss",
    "score_action_chunks",
]
