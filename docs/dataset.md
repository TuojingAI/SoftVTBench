# Tabero / SoftTacWorld-v0 Dataset

The public dataset is hosted on ModelScope:

```text
https://www.modelscope.cn/datasets/Arthur12137/SoftTacWorld-v0
```

Download it with:

```bash
scripts/download_tabero_dataset.py --local-dir data_tabero/raw/SoftTacWorld-v0
```

The script uses:

```bash
modelscope download --dataset Arthur12137/SoftTacWorld-v0 --local_dir data_tabero/raw/SoftTacWorld-v0
```

The training pipeline expects the downloaded data to expose this Tabero/Isaaclab-Libero style tree:

```text
raw_root/
  libero_object/
    replayed_demos/*.hdf5
    video_datasets/libero_object_task*/videos/*.mp4
    video_datasets/libero_object_task*/tactile_outputs/*.mp4
  libero_spatial/
    replayed_demos/*.hdf5
    video_datasets/libero_spatial_task*/videos/*.mp4
    video_datasets/libero_spatial_task*/tactile_outputs/*.mp4
```

The reference run used:

```text
628 episodes
88104 frames
20 FPS
20 tasks = 10 libero_object + 10 libero_spatial
```

## LeRobot Outputs

Vision dataset:

```text
image: [224, 224, 3]
wrist_image: [224, 224, 3]
state: [7]
actions: [7]
```

Vision+tactile dataset:

```text
image: [224, 224, 3]
wrist_image: [224, 224, 3]
tactile_image: [224, 224, 3]
tactile_gripper_force: [8, 6]
tactile_marker_motion: [9, 198, 2]
state: [7]
actions: [13]
```

## Conversion

Set environment variables from `configs/training/tabero_env.example`, then run:

```bash
scripts/prepare_tabero_dataset.sh vision
scripts/prepare_tabero_dataset.sh tactile
```

The wrapper calls:

```bash
examples/tabero/convert_tabero_vision_data_to_lerobot.py
examples/tabero/convert_tabero_tactile_data_to_lerobot.py
```

## Frame Alignment Check

Before training, verify HDF5/MP4 frame alignment:

```bash
$PYTHON examples/tabero/verify_tabero_frame_alignment.py \
  --root "${STAGE_ROOT}" \
  --decode
```

Expected reference result:

```text
checked demos: 628
summary: {'all_equal': 628}
missing count: 0
```
