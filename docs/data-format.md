# HDF5 Data Format Comparison (old_data / new_data x rigid / soft)

> Sampled from `demo_0` of the `libero_object` dataset under `{old_data,new_data}` in the project data root.
> The top-level structure is always `data/<demo_i>/{actions, initial_state/, obs/, states/}`; each code block below expands the internal layout of one trajectory under `data/demo_0/` (`T` = number of frames in that demo; object scenes have a fixed set of 7 rigid distractors).
> Conventions: `root_pose` = 7-D (xyz + quat wxyz), `root_velocity` = 6-D, Franka `joint_*` = 9-D (7 arm + 2 fingers).

---

## 1) old_data / rigid

```text
data/demo_0/                         attrs: {num_samples, success}
├─ actions                (T, 13)   float32   # primary actions (13D)
├─ initial_state/                              # single-frame (frame 0) initial values
│  ├─ articulation/robot/{joint_position(1,9), joint_velocity(1,9), root_pose(1,7), root_velocity(1,6)}
│  └─ rigid_object/<obj>/{root_pose(1,7), root_velocity(1,6)}      # 8 objects: target + basket + 6 distractors
├─ obs/                                         # observation streams
│  ├─ actions              (T, 8)   float32
│  ├─ arm_joint_pos        (T, 7)   float32
│  ├─ eef_pose             (T, 7)   float32
│  ├─ gripper_marker_motion(T,2,2,99,2) float32 # tactile marker displacement field
│  ├─ gripper_net_force    (T,1,2,3) float32     # net force proxy for the two fingers
│  └─ gripper_pos          (T, 2)   float32
└─ states/                                      # full-trajectory physical states (T frames)
   ├─ articulation/robot/{joint_position(T,9), joint_velocity(T,9), root_pose(T,7), root_velocity(T,6)}
   └─ rigid_object/<obj>/{root_pose(T,7), root_velocity(T,6)}
# no deformable_object; no torque / finger_force / soft_extras
```

---

## 2) old_data / soft

```text
data/demo_0/   attrs: {asset_name, language, task, task_id, num_frames, metadata_json, source_hdf5, success, ...}
├─ actions                (T, 13)   float32
├─ initial_state/
│  ├─ articulation/robot/{joint_position(1,9), joint_velocity(1,9), root_pose(1,7), root_velocity(1,6)}
│  ├─ deformable_object/<soft>/{bbox_max_w(1,3), bbox_min_w(1,3), nodal_pos_w(1,182,3), root_pos_w(1,3)}
│  └─ rigid_object/<obj>/{root_pose(1,7), root_velocity(1,6)}   # includes <soft>_tactile_proxy
├─ obs/
│  ├─ actions              (T, 8)    float32
│  ├─ arm_joint_pos        (T, 7)    float32
│  ├─ eef_pose             (T, 7)    float32
│  ├─ fem_bbox_dims        (T, 3)    float32   # <- new for soft
│  ├─ fem_deformation_max  (T,)      float32   # <- new for soft
│  ├─ fem_deformation_rms  (T,)      float32   # <- new for soft
│  ├─ gripper_close_norm   (T,)      float32   # <- new for soft
│  ├─ gripper_marker_motion(T,2,2,99,2) float32
│  ├─ gripper_net_force    (T,1,2,3) float32
│  ├─ gripper_pos          (T, 2)    float32
│  └─ gripper_width        (T,)      float32   # <- new for soft
├─ soft_extras/            attrs: {actions_13d_format, axis_angle_*, net_force_*, ...}
│  ├─ actions_13d          (T, 13)   float32
│  ├─ actions_raw          (T, 8)    float32
│  ├─ arm_joint_pos        (T, 7)    float32
│  ├─ fem_bbox_dims / fem_deformation_max / fem_deformation_rms / gripper_close_norm / gripper_width
│  ├─ gripper_marker_motion(T,2,2,99,2)
│  ├─ gripper_net_force    (T,1,2,3) / gripper_net_force_raw (T,1,2,3)
│  └─ gripper_net_force_source (T,)  int8
└─ states/
   ├─ articulation/robot/{joint_position(T,9), joint_velocity(T,9), root_pose(T,7), root_velocity(T,6)}
   ├─ deformable_object/<soft>/{bbox_max_w(T,3), bbox_min_w(T,3), nodal_pos_w(T,182,3), root_pos_w(T,3)}
   └─ rigid_object/<obj>/{root_pose(T,7), root_velocity(T,6)}
# vs. old rigid: + deformable_object (182-node mesh) + fem_* + gripper_close_norm/width + soft_extras
```

---

## 3) new_data / rigid

```text
data/demo_0/   attrs: {asset_name=rigid_pastryXXX, task='SoftVT-Soft-Collection-v0', num_frames, metadata_json, source_hdf5, ...}
├─ actions                (T, 13)   float32
├─ actions_binary         (T, 7)    float32   # <- new: binary-gripper action variant
├─ initial_state/
│  ├─ articulation/robot/{
│  │     applied_torque(1,9), computed_torque(1,9), joint_effort_target(1,9),   # <- new: torque chain
│  │     body_incoming_joint_wrench(1,17,6),                                    # <- new: joint wrench
│  │     joint_position(1,9), joint_velocity(1,9), root_pose(1,7), root_velocity(1,6)}
│  ├─ deformable_object/                       # empty group (rigid has no soft body)
│  └─ rigid_object/<obj>/{root_pose(1,7), root_velocity(1,6)}   # includes rigid_pastryXXX
├─ obs/
│  ├─ actions(T,8), arm_joint_pos(T,7), eef_pose(T,7)
│  ├─ applied_torque(T,9), computed_torque(T,9)      # <- new
│  ├─ eef_axis_angle       (T, 3)    float32          # <- new: pose as axis-angle
│  ├─ fem_bbox_dims        (T, 3)    float32
│  ├─ finger_force         (T, 2)    float32          # <- new: per-finger contact force
│  ├─ gripper_binary       (T,)      float32          # <- new
│  ├─ gripper_close_norm   (T,)      float32
│  ├─ gripper_marker_motion(T,2,2,99,2), gripper_net_force(T,1,2,3), gripper_pos(T,2), gripper_width(T,)
├─ soft_extras/  attrs: {actions_13d_format, actions_7d2_format, binary_*, axis_angle_*, net_force_*, ...}
│  ├─ actions_13d(T,13), actions_7d2(T,7), actions_raw(T,8)     # three action conventions
│  ├─ applied_torque(T,9), computed_torque(T,9), eef_axis_angle(T,3)
│  ├─ arm_joint_pos(T,7), fem_bbox_dims(T,3), finger_force(T,2)
│  ├─ gripper_binary(T,), gripper_close_norm(T,), gripper_width(T,)
│  ├─ gripper_marker_motion(T,2,2,99,2)
│  ├─ gripper_net_force(T,1,2,3), gripper_net_force_raw(T,1,2,3), gripper_net_force_source(T,) int8
└─ states/
   ├─ articulation/robot/{applied_torque(T,9), body_incoming_joint_wrench(T,17,6), computed_torque(T,9),
   │                      joint_effort_target(T,9), joint_position(T,9), joint_velocity(T,9),
   │                      root_pose(T,7), root_velocity(T,6)}
   └─ rigid_object/<obj>/{root_pose(T,7), root_velocity(T,6)}
# vs. old rigid: + actions_binary + full torque/wrench chain + eef_axis_angle + finger_force + gripper_binary + soft_extras
```

---

## 4) new_data / soft

```text
data/demo_0/   attrs: {asset_name=soft_pastryXXX, task='SoftVT-Soft-Collection-v0', num_frames, metadata_json, source_hdf5, ...}
├─ actions                (T, 13)   float32
├─ actions_binary         (T, 7)    float32
├─ initial_state/
│  ├─ articulation/robot/{applied_torque(1,9), body_incoming_joint_wrench(1,17,6), computed_torque(1,9),
│  │                      joint_effort_target(1,9), joint_position(1,9), joint_velocity(1,9),
│  │                      root_pose(1,7), root_velocity(1,6)}
│  ├─ deformable_object/<soft>/{bbox_max_w(1,3), bbox_min_w(1,3), nodal_pos_w(1,182,3), root_pos_w(1,3)}
│  └─ rigid_object/<obj>/{root_pose(1,7), root_velocity(1,6)}   # includes <soft>_tactile_proxy
├─ obs/
│  ├─ actions(T,8), arm_joint_pos(T,7), eef_pose(T,7)
│  ├─ applied_torque(T,9), computed_torque(T,9), eef_axis_angle(T,3)
│  ├─ fem_bbox_dims        (T, 3)    float32
│  ├─ fem_deformation_max  (T,)      float32
│  ├─ fem_deformation_rms  (T,)      float32
│  ├─ fem_kabsch_max_pct   (T,)      float32   # <- new for new soft: net deformation after removing rigid-body motion
│  ├─ fem_kabsch_rms_pct   (T,)      float32   # <- new for new soft
│  ├─ finger_force         (T, 2)    float32
│  ├─ gripper_binary(T,), gripper_close_norm(T,)
│  ├─ gripper_marker_motion(T,2,2,99,2), gripper_net_force(T,1,2,3), gripper_pos(T,2), gripper_width(T,)
├─ soft_extras/  attrs: {actions_13d_format, actions_7d2_format, binary_*, axis_angle_*, net_force_*, ...}
│  ├─ actions_13d(T,13), actions_7d2(T,7), actions_raw(T,8)
│  ├─ applied_torque(T,9), computed_torque(T,9), eef_axis_angle(T,3), arm_joint_pos(T,7)
│  ├─ fem_bbox_dims(T,3), fem_deformation_max(T,), fem_deformation_rms(T,), fem_kabsch_max_pct(T,), fem_kabsch_rms_pct(T,)
│  ├─ finger_force(T,2), gripper_binary(T,), gripper_close_norm(T,), gripper_width(T,)
│  ├─ gripper_marker_motion(T,2,2,99,2)
│  ├─ gripper_net_force(T,1,2,3), gripper_net_force_raw(T,1,2,3), gripper_net_force_source(T,) int8
└─ states/
   ├─ articulation/robot/{applied_torque(T,9), body_incoming_joint_wrench(T,17,6), computed_torque(T,9),
   │                      joint_effort_target(T,9), joint_position(T,9), joint_velocity(T,9),
   │                      root_pose(T,7), root_velocity(T,6)}
   ├─ deformable_object/<soft>/{bbox_max_w(T,3), bbox_min_w(T,3), nodal_pos_w(T,182,3), root_pos_w(T,3)}
   └─ rigid_object/<obj>/{root_pose(T,7), root_velocity(T,6)}
# = all fields of new rigid + deformable_object (182 nodes) + fem_deformation_* + fem_kabsch_* (net deformation)
```

---

## Difference quick reference

| Field/group | old rigid | old soft | new rigid | new soft |
|---|:--:|:--:|:--:|:--:|
| `actions` (T,13) | yes | yes | yes | yes |
| `actions_binary` (T,7) | no | no | yes | yes |
| `deformable_object` (182-node mesh) | no | yes | empty group | yes |
| `fem_deformation_max/rms` | no | yes | no | yes |
| `fem_kabsch_max/rms_pct` (net deformation) | no | no | no | yes |
| Torque chain `applied/computed_torque`, `joint_effort_target`, `body_incoming_joint_wrench` | no | no | yes | yes |
| `eef_axis_angle` (T,3) | no | no | yes | yes |
| `finger_force` (T,2) | no | no | yes | yes |
| `gripper_binary` | no | no | yes | yes |
| `soft_extras/` (raw / multi-convention actions) | no | yes | yes | yes |
| demo `attrs` (language/task/metadata_json...) | only num_samples, success | full | full | full |
| `task` field | - | `Isaac-Libero-Franka-IK-Camera-Tactile-v0` | `SoftVT-Soft-Collection-v0` | `SoftVT-Soft-Collection-v0` |

> Sampled files:
> - old rigid: `old_data/object_rigid/.../task0/..._alphabet_soup_...hdf5` (44 demos)
> - old soft:  `old_data/object_soft/.../task0/..._white_swirled_soft_pastry_...hdf5` (50 demos)
> - new rigid: `new_data/object_rigid/.../task0/..._white_swirled_pastry_...hdf5` (100 demos)
> - new soft:  `new_data/object_soft/task0/..._white_swirled_pastry_...hdf5` (100 demos)
