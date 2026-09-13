"""Smoke tests for the Isaac-Lab-independent agentic modules."""

import pytest

torch = pytest.importorskip("torch")

from BrainCo_DexHand.algo.agentic import (
    ConsequenceBatch,
    ConsequencePredictor,
    VisualLanguageStudent,
    VisualStudentBatch,
    consequence_loss,
    evidence_loss,
    score_action_chunks,
    batch_instructions,
    encode_face_goal,
)


def test_language_goal_contract_is_compositional_and_stable():
    ids = torch.tensor([0, 5, 2])
    goal = encode_face_goal(ids)
    assert goal.shape == (3, 6)
    assert torch.allclose(goal.sum(dim=-1), torch.ones(3))
    assert batch_instructions(ids) == [
        "show the red marker", "show the cyan marker", "show the blue marker"
    ]


def test_visual_student_static_language_and_padding():
    torch.manual_seed(0)
    b, t = 3, 5
    model = VisualLanguageStudent(32, 16, 10, touch_dim=4, action_history_dim=21, hidden_dim=64)
    batch = VisualStudentBatch(
        torch.randn(b, t, 32), torch.randn(b, 16), torch.randn(b, t, 10),
        touch=torch.randn(b, t, 4), action_history=torch.randn(b, t, 21),
        valid_mask=torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool),
    )
    out = model(batch)
    assert out["action"].shape == (b, 21)
    assert out["evidence_logits"].shape == (b, 3)
    assert torch.all((out["action"] >= -1) & (out["action"] <= 1))
    aux = evidence_loss(out, torch.rand(b, 3), torch.rand(b))
    aux.backward()
    assert torch.isfinite(aux)


def test_consequence_predictor_loss_and_scoring():
    torch.manual_seed(1)
    b, k, h, a = 2, 4, 3, 21
    predictor = ConsequencePredictor(state_dim=48, action_dim=a, hidden_dim=64)
    batch = ConsequenceBatch(torch.randn(b, 48), torch.randn(b, k, h, a), torch.rand(b, k, 4))
    pred = predictor(batch)
    assert pred.shape == (b, k, 4)
    loss = consequence_loss(pred, batch.targets, mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss)
    scores = score_action_chunks(pred)
    assert scores.shape == (b, k)
