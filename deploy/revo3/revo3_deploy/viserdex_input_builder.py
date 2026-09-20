"""Explicit visual-language input contract for the Revo3 VisERDex adapter.

The existing :class:`Stage2InputBuilder` remains the compatibility path for
the old proprioceptive policies.  This builder is deliberately separate: a
visual policy must not silently receive the old 126-D tensor when its pose
estimator or language encoder is unavailable.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from revo3_deploy.input_builder import ACTION_SCALE, JOINT_DIM


POSE_DIM = 7  # translation (3) + quaternion wxyz (4)
LANGUAGE_DIM = 16
EXTERO_DIM = POSE_DIM + 1  # pose plus estimator confidence
HIST_LEN = 30
OBS_FRAMES = 3
PROPRIO_FRAME_DIM = 2 * JOINT_DIM
FRAME_DIM = PROPRIO_FRAME_DIM + EXTERO_DIM + LANGUAGE_DIM


class VisERDexInputBuilder:
    """Build a versioned Revo3 visual-language policy input.

    Inputs are a fixed-length history of normalized proprioception, the
    estimated Cube pose and confidence, and a precomputed language feature.
    The language encoder is intentionally outside this module so the same
    contract can be fed by a local text model or a recorded feature bank.
    """

    def __init__(
        self,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        joint_names: tuple[str, ...] | list[str],
        *,
        language_dim: int = LANGUAGE_DIM,
        history_len: int = HIST_LEN,
        action_scale: float = ACTION_SCALE,
    ) -> None:
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != JOINT_DIM or len(set(self.joint_names)) != JOINT_DIM:
            raise ValueError(f"joint_names must contain {JOINT_DIM} unique joints.")
        self.joint_lower = self._vector(joint_lower, "joint_lower")
        self.joint_upper = self._vector(joint_upper, "joint_upper")
        if np.any(self.joint_upper <= self.joint_lower):
            raise ValueError("Every joint upper limit must be greater than lower.")
        if language_dim <= 0 or history_len < OBS_FRAMES:
            raise ValueError("language_dim must be positive and history_len must be >= 3.")
        self.language_dim = int(language_dim)
        self.history_len = int(history_len)
        self.frame_dim = FRAME_DIM - LANGUAGE_DIM + self.language_dim
        self.action_scale = float(action_scale)
        self.current_target: np.ndarray | None = None
        self._language: np.ndarray | None = None
        self._frames: deque[np.ndarray] = deque(maxlen=self.history_len)

    def reset(
        self,
        joint_pos: np.ndarray,
        estimated_pose: np.ndarray,
        confidence: float,
        language_embedding: np.ndarray,
        target: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        q = self._vector(joint_pos, "joint_pos")
        target = q if target is None else self._vector(target, "target")
        self.current_target = self._clip_target(target)
        self._language = self._language_vector(language_embedding)
        frame = self._build_frame(q, self.current_target, estimated_pose, confidence)
        self._frames.clear()
        for _ in range(self.history_len):
            self._frames.append(frame.copy())
        return self._inputs()

    def observe(
        self,
        joint_pos: np.ndarray,
        estimated_pose: np.ndarray,
        confidence: float,
        language_embedding: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if self.current_target is None or self._language is None:
            return self.reset(joint_pos, estimated_pose, confidence, language_embedding)
        self._language = self._language_vector(language_embedding)
        q = self._vector(joint_pos, "joint_pos")
        self._frames.append(self._build_frame(q, self.current_target, estimated_pose, confidence))
        return self._inputs()

    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        if self.current_target is None:
            raise RuntimeError("Call reset() before action_to_target().")
        action = np.clip(self._vector(action, "action"), -1.0, 1.0)
        self.current_target = self._clip_target(self.current_target + self.action_scale * action)
        return self.current_target.copy()

    def _inputs(self) -> dict[str, np.ndarray]:
        if len(self._frames) != self.history_len or self._language is None:
            raise RuntimeError("Input history is not initialized.")
        history = np.stack(tuple(self._frames), axis=0).astype(np.float32)
        return {
            "obs": history[-OBS_FRAMES:].reshape(1, OBS_FRAMES * self.frame_dim),
            "proprio_hist": history[:, :PROPRIO_FRAME_DIM].reshape(1, self.history_len, PROPRIO_FRAME_DIM),
            "pose_hist": history[:, PROPRIO_FRAME_DIM : PROPRIO_FRAME_DIM + EXTERO_DIM].reshape(
                1, self.history_len, EXTERO_DIM
            ),
            "language": self._language.reshape(1, self.language_dim),
        }

    def _build_frame(self, q: np.ndarray, target: np.ndarray, pose: np.ndarray, confidence: float) -> np.ndarray:
        q_norm = (2.0 * q - self.joint_upper - self.joint_lower) / (self.joint_upper - self.joint_lower)
        pose = self._pose_vector(pose)
        confidence = float(np.clip(confidence, 0.0, 1.0))
        assert self._language is not None
        return np.concatenate((q_norm, target, pose, np.array([confidence], dtype=np.float32), self._language)).astype(
            np.float32
        )

    def _clip_target(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self.joint_lower, self.joint_upper).astype(np.float32)

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.shape != (JOINT_DIM,):
            raise ValueError(f"{name} must have shape ({JOINT_DIM},), got {vector.shape}.")
        return vector

    @staticmethod
    def _pose_vector(value: np.ndarray) -> np.ndarray:
        pose = np.asarray(value, dtype=np.float32).reshape(-1)
        if pose.shape != (POSE_DIM,):
            raise ValueError(f"estimated_pose must have shape ({POSE_DIM},), got {pose.shape}.")
        quat = pose[3:]
        norm = float(np.linalg.norm(quat))
        if norm < 1e-8:
            raise ValueError("estimated_pose quaternion has near-zero norm.")
        pose = pose.copy()
        pose[3:] = quat / norm
        return pose

    def _language_vector(self, value: np.ndarray) -> np.ndarray:
        language = np.asarray(value, dtype=np.float32).reshape(-1)
        if language.shape != (self.language_dim,):
            raise ValueError(f"language_embedding must have shape ({self.language_dim},), got {language.shape}.")
        return language
