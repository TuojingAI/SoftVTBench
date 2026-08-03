"""Policy gripper output -> environment gripper command. State advances per **executed step** (transcribed from the formal semantics).

Modes (gripper_execution.mode per checkpoint in policies.yaml):
  binary          training label is +-1; threshold at 0 -> env BinaryJointAction (+1 open / -1 close). 8D action.
  continuous_direct  soft-body continuous ablation migrated from the internal collection
                  stack's policy_abs: label = next frame's **measured single-finger joint
                  position** (not q_target), clamped to the mechanical range and applied
                  directly as the absolute joint target for both fingers. 9D.
  measured_aperture_direct  legacy name for continuous_direct, kept for compatibility with
                  existing results/configs. Both are value-for-value equivalent to the
                  internal policy_abs when closure_gain=1.
                  Treating the (closure_gain-scaled) value as the actuator target is a
                  decoder assumption -- **not strictly train/inference consistent**. 9D.
                  (True q_target_absolute would need retraining / inverse-dynamics
                  calibration; not implemented.)
  continuous_fixed_position  the continuous checkpoint's measured-aperture output is used
                  only for a Schmitt open/close decision; execution is an absolute joint
                  target: open=0.04, close=0.00. The actual closed position is clamped by
                  the gripper_width/2 joint limit calibrated at collection time for that
                  episode. No marker/FEM/task feedback. 9D.
  decoded_binary  rigid collection used a binary actuator; the continuous label is decoded
                  back to a binary command via a Schmitt trigger (close/open thresholds +
                  minimum hold steps). 8D. Same as the formal VT_BINARY_DECODER.
  relative_decoded_binary  relative decoding of the FastWAM measured-aperture label: close
                  after dropping close_delta below the rolling open peak, open after rising
                  open_delta above the closed trough. Thresholds adapt to the episode's
                  physically observable travel; with one_shot=True the gripper stays open
                  permanently after the first release; pre-close when the travel is too
                  small to observe. 8D.
  relative_fixed_position  same relative decoding, but executes absolute fixed open/close targets. 9D.

step() is called once per executed step: latch state only advances with real execution
(an episode terminated mid-way never commits ahead).
"""
from __future__ import annotations

import numpy as np


ABSOLUTE_GRIPPER_MODES = frozenset((
    "continuous_direct",
    "measured_aperture_direct",
    "continuous_fixed_position",
    "relative_fixed_position",
))
FIXED_POSITION_GRIPPER_MODES = frozenset((
    "continuous_fixed_position",
    "relative_fixed_position",
))
RELATIVE_GRIPPER_MODES = frozenset((
    "relative_decoded_binary",
    "relative_fixed_position",
))


class GripperExecutor:
    def __init__(self, mode: str, *, open_finger: float = 0.04, closure_gain: float = 1.0,
                 close_threshold: float | None = None, open_threshold: float | None = None,
                 min_hold_steps: int = 10, fixed_close_finger: float = 0.0,
                 close_delta: float | None = None, open_delta: float | None = None,
                 aperture_noise_tolerance: float = 0.0005,
                 one_shot: bool = False):
        if mode not in (
            "binary", *ABSOLUTE_GRIPPER_MODES, *RELATIVE_GRIPPER_MODES, "decoded_binary"
        ):
            raise ValueError(f"unknown gripper execution mode {mode!r}")
        if mode in ("decoded_binary", "continuous_fixed_position") and (
            close_threshold is None
            or open_threshold is None
            or not open_threshold > close_threshold
        ):
            raise ValueError(
                f"{mode} requires open_threshold > close_threshold"
            )
        if mode in RELATIVE_GRIPPER_MODES and (
            close_delta is None
            or open_delta is None
            or not np.isfinite(close_delta)
            or not np.isfinite(open_delta)
            or close_delta <= 0.0
            or open_delta <= 0.0
        ):
            raise ValueError(f"{mode} requires finite positive close_delta/open_delta")
        if min_hold_steps < 0:
            raise ValueError("min_hold_steps must be >= 0")
        if not np.isfinite(aperture_noise_tolerance) or aperture_noise_tolerance <= 0.0:
            raise ValueError("aperture_noise_tolerance must be finite and > 0")
        if not isinstance(one_shot, (bool, np.bool_)):
            raise ValueError("one_shot must be boolean")
        self.mode = mode
        self.open_finger = float(open_finger)
        self.closure_gain = float(closure_gain)
        self.close_threshold = close_threshold
        self.open_threshold = open_threshold
        self.min_hold_steps = int(min_hold_steps)
        self.fixed_close_finger = float(fixed_close_finger)
        self.close_delta = None if close_delta is None else float(close_delta)
        self.open_delta = None if open_delta is None else float(open_delta)
        self.aperture_noise_tolerance = float(aperture_noise_tolerance)
        self.one_shot = bool(one_shot)
        if not 0.0 <= self.fixed_close_finger < self.open_finger:
            raise ValueError(
                "fixed_close_finger must lie in [0, open_finger)"
            )
        self.reset()

    def reset(self, *, finger_lower_limit: float | None = None) -> None:
        self._latch_active = False
        self._hold_remaining = 0
        self._finger_lower_limit = (
            None if finger_lower_limit is None else float(finger_lower_limit)
        )
        self._relative_reference = None
        self._effective_close_delta = self.close_delta
        self._effective_open_delta = self.open_delta
        self._preclosed_unobservable = False
        self._ever_closed = False
        self._release_committed = False
        if self.mode in RELATIVE_GRIPPER_MODES and self._finger_lower_limit is not None:
            if not np.isfinite(self._finger_lower_limit):
                raise ValueError("finger_lower_limit must be finite")
            travel = self.open_finger - self._finger_lower_limit
            if travel < 0.0:
                raise ValueError(
                    "finger_lower_limit cannot exceed open_finger: "
                    f"{self._finger_lower_limit} > {self.open_finger}"
                )
            if travel <= self.aperture_noise_tolerance:
                # The measured-aperture label cannot encode binary intent when
                # open and physically closed positions coincide. Closing early
                # preserves contact force without changing the calibrated gap.
                self._latch_active = True
                self._ever_closed = True
                self._preclosed_unobservable = True
            else:
                adaptive_delta = max(self.aperture_noise_tolerance, 0.5 * travel)
                self._effective_close_delta = min(self.close_delta, adaptive_delta)
                self._effective_open_delta = min(self.open_delta, adaptive_delta)
        self.last_diag: dict[str, float | bool | str] = {
            "mode": self.mode,
            "action": "preclosed_unobservable" if self._preclosed_unobservable else "reset",
        }

    @property
    def action_dim(self) -> int:
        return 9 if self.mode in ABSOLUTE_GRIPPER_MODES else 8

    def step(self, raw_g: float) -> np.ndarray:
        """One step: policy gripper prediction -> env gripper columns ((1,) or (2,)). State advances with this step."""
        raw_g = float(raw_g)
        if not np.isfinite(raw_g):
            raise ValueError(f"gripper prediction must be finite, got {raw_g}")
        if self.mode == "binary":
            return np.asarray([1.0 if raw_g > 0.0 else -1.0], dtype=np.float32)
        if self.mode in RELATIVE_GRIPPER_MODES:
            self._step_relative_latch(raw_g)
            if self.mode == "relative_fixed_position":
                finger = self.fixed_close_finger if self._latch_active else self.open_finger
                command = np.asarray([finger, finger], dtype=np.float32)
                action = "fixed_close" if self._latch_active else "fixed_open"
                executed = {"executed_q_target": finger}
            else:
                command = np.asarray([-1.0 if self._latch_active else 1.0], dtype=np.float32)
                action = "binary_close" if self._latch_active else "binary_open"
                executed = {"executed_binary_command": float(command[0])}
            self.last_diag = {
                "mode": self.mode,
                "action": action,
                "raw_q_measured": raw_g,
                "close_intent": self._latch_active,
                "relative_reference": float(self._relative_reference),
                "effective_close_delta": float(self._effective_close_delta),
                "effective_open_delta": float(self._effective_open_delta),
                "physical_close_limit": float(self._finger_lower_limit),
                "preclosed_unobservable": self._preclosed_unobservable,
                "release_committed": self._release_committed,
                **executed,
            }
            return command
        if self.mode in ABSOLUTE_GRIPPER_MODES:
            if self.mode == "continuous_fixed_position":
                if self._finger_lower_limit is None:
                    raise RuntimeError(
                        "continuous_fixed_position requires the episode "
                        "finger lower limit (gripper_width / 2)"
                    )
                if self._latch_active:
                    if self._hold_remaining > 0:
                        self._hold_remaining -= 1
                    elif raw_g >= self.open_threshold:
                        self._latch_active = False
                elif raw_g <= self.close_threshold:
                    self._latch_active = True
                    self._hold_remaining = max(0, self.min_hold_steps - 1)
                finger = (
                    self.fixed_close_finger
                    if self._latch_active
                    else self.open_finger
                )
                finger = min(max(finger, 0.0), self.open_finger)
                self.last_diag = {
                    "mode": self.mode,
                    "action": "fixed_close" if self._latch_active else "fixed_open",
                    "raw_q_measured": raw_g,
                    "close_intent": self._latch_active,
                    "executed_q_target": finger,
                    "physical_close_limit": self._finger_lower_limit,
                }
                return np.asarray([finger, finger], dtype=np.float32)
            pred = min(max(raw_g, 0.0), self.open_finger)
            finger = self.open_finger - self.closure_gain * (self.open_finger - pred)
            finger = min(max(finger, 0.0), self.open_finger)
            return np.asarray([finger, finger], dtype=np.float32)
        # decoded_binary -- Schmitt trigger
        if self._latch_active:
            if self._hold_remaining > 0:
                self._hold_remaining -= 1
            elif raw_g >= self.open_threshold:
                self._latch_active = False
        elif raw_g <= self.close_threshold:
            self._latch_active = True
            self._hold_remaining = max(0, self.min_hold_steps - 1)
        return np.asarray([-1.0 if self._latch_active else 1.0], dtype=np.float32)

    def _step_relative_latch(self, raw_g: float) -> None:
        if self._finger_lower_limit is None:
            raise RuntimeError(
                f"{self.mode} requires the episode finger lower limit (gripper_width / 2)"
            )
        if self._preclosed_unobservable:
            self._relative_reference = raw_g
            return
        if self._release_committed:
            self._relative_reference = raw_g
            return
        if self._relative_reference is None:
            self._relative_reference = raw_g
            return
        if self._latch_active:
            self._relative_reference = min(self._relative_reference, raw_g)
            if self._hold_remaining > 0:
                self._hold_remaining -= 1
            elif raw_g >= self._relative_reference + self._effective_open_delta:
                self._latch_active = False
                self._release_committed = self.one_shot and self._ever_closed
                self._relative_reference = raw_g
        else:
            self._relative_reference = max(self._relative_reference, raw_g)
            if raw_g <= self._relative_reference - self._effective_close_delta:
                self._latch_active = True
                self._ever_closed = True
                self._hold_remaining = max(0, self.min_hold_steps - 1)
                self._relative_reference = raw_g
