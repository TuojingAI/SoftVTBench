"""简单脚本：把本地的 LeRobot 数据集目录上传到 Hugging Face Hub.

用法示例：

1）先登录（只需一次）：

    huggingface-cli login
    # 或者：export HF_TOKEN=xxx

2）只上传 data/ 和 meta/（不会上传 images/ 或 videos/，也不会创建空目录）：

    python scripts/tools/upload_lerobot_to_hf.py --softvtbench \
        --repo-id your_username/softvtbench_all_libero_suites

或：

    python scripts/tools/upload_lerobot_to_hf.py --softvtbench_force \
        --repo-id your_username/softvtbench_force_all_libero_suites

或（二值夹爪动作 / 7d2 录制 → OpenPI 转换得到的 pi0 数据）：

    python scripts/tools/upload_lerobot_to_hf.py --softvtbench_binary \
        --repo-id your_username/softvtbench_binary_all_libero_suites

3）如果你的本地目录不在默认位置，可直接覆盖：

    python scripts/tools/upload_lerobot_to_hf.py --local-path /abs/path/to/lerobot_dataset \
        --repo-id your_username/your_dataset_repo

需要本机已安装：

    pip install -U huggingface_hub
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi

# NOTE: huggingface_hub 的异常类导出路径在不同版本中有变化：
# - 有些版本: from huggingface_hub import HfHubHTTPError
# - 新版本:   from huggingface_hub.utils import HfHubHTTPError
try:
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore
except Exception:  # pragma: no cover
    try:
        from huggingface_hub import HfHubHTTPError  # type: ignore
    except Exception:  # pragma: no cover
        HfHubHTTPError = Exception  # type: ignore[misc,assignment]


@dataclass
class Config:
    """上传配置."""

    # 选择要上传的转换后数据集（唯一逻辑）
    dataset: str  # 'softvtbench' | 'softvtbench_force' | 'softvtbench_binary'

    # Hugging Face 上的数据集仓库名：形如 "用户名/仓库名"
    # 例如："qiweiw/lerobot_all_libero_suites"
    repo_id: str

    # 仓库类型，LeRobot 数据集统一用 "dataset"
    repo_type: str = 'dataset'

    # 是否创建（或确保存在）目标仓库
    create_repo: bool = True

    # 新建仓库时是否设为私有
    private: bool = False

    # 严格只上传这些子目录（images/videos 已拼到 data 字段里，不上传）
    include_subdirs: tuple[str, ...] = ('data', 'meta')

    # 可选：覆盖本地数据集目录（优先级最高）
    local_path: Path | None = None


def _iter_files(root: Path) -> Iterable[Path]:
    """递归遍历 root 下的文件（不包含目录本身）."""
    yield from (p for p in root.rglob('*') if p.is_file())


def _default_local_path(dataset: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    if dataset == 'softvtbench':
        return repo_root / 'benchmarks' / 'datasets' / 'softvtbench_pi0' / 'softvtbench_all_libero_suites'
    if dataset == 'softvtbench_force':
        return repo_root / 'benchmarks' / 'datasets' / 'softvtbench_force_pi0' / 'softvtbench_force_all_libero_suites'
    if dataset == 'softvtbench_binary':
        return repo_root / 'benchmarks' / 'datasets' / 'softvtbench_pi0_binary' / 'softvtbench_binary_all_libero_suites'
    raise ValueError(f'未知 dataset={dataset}，仅支持 softvtbench / softvtbench_force')


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(description='上传本地 LeRobot 数据集（仅 data/ + meta/）到 Hugging Face Hub')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--softvtbench', action='store_true', help='选择 softvtbench_pi0/softvtbench_all_libero_suites')
    group.add_argument('--softvtbench_force', action='store_true', help='选择 softvtbench_force_pi0/softvtbench_force_all_libero_suites')
    group.add_argument('--softvtbench_binary', action='store_true', help='选择 softvtbench_pi0_binary/softvtbench_binary_all_libero_suites')

    parser.add_argument('--repo-id', required=True, help='Hugging Face repo_id，例如：username/softvtbench_all_libero_suites')
    parser.add_argument('--repo-type', default='dataset', help='仓库类型（默认 dataset）')
    parser.add_argument('--create-repo', action='store_true', default=True, help='创建/确保仓库存在（默认 True）')
    parser.add_argument('--no-create-repo', action='store_false', dest='create_repo', help='不要创建仓库')
    parser.add_argument('--private', action='store_true', default=False, help='新建仓库时设为私有')
    parser.add_argument(
        '--local-path',
        type=str,
        default=None,
        help='覆盖本地数据集目录（优先级最高）。例如：/abs/path/to/softvtbench_pi0_binary/softvtbench_binary_all_libero_suites',
    )

    args = parser.parse_args()

    dataset = 'softvtbench'
    if args.softvtbench_force:
        dataset = 'softvtbench_force'
    elif args.softvtbench_binary:
        dataset = 'softvtbench_binary'
    elif args.softvtbench:
        dataset = 'softvtbench'

    return Config(
        dataset=dataset,
        repo_id=str(args.repo_id),
        repo_type=str(args.repo_type),
        create_repo=bool(args.create_repo),
        private=bool(args.private),
        local_path=Path(args.local_path).expanduser().resolve() if args.local_path else None,
    )


def main(cfg: Config) -> None:
    """脚本入口：仅上传指定子目录到 Hugging Face Hub."""
    local_path = (cfg.local_path or _default_local_path(cfg.dataset)).expanduser().resolve()

    if not local_path.exists():
        raise FileNotFoundError(f'本地路径不存在: {local_path}')

    print(f'\n本地数据集目录: {local_path}')
    print(f'Hugging Face repo_id: {cfg.repo_id} (type={cfg.repo_type})')

    api = HfApi()

    # 可选：先创建/确保仓库存在
    if cfg.create_repo:
        print('检查 / 创建 Hugging Face 仓库中...')
        api.create_repo(
            repo_id=cfg.repo_id,
            repo_type=cfg.repo_type,
            private=cfg.private,
            exist_ok=True,
        )

    print('开始上传（按子目录）到 Hugging Face Hub，这可能需要一段时间...')
    include_str = ', '.join(cfg.include_subdirs)
    print(f'只上传子目录: {include_str}')

    uploaded_any = False
    for sub in cfg.include_subdirs:
        sub_dir = local_path / sub
        if not sub_dir.exists():
            print(f'[SKIP] 子目录不存在：{sub_dir}')
            continue
        if not sub_dir.is_dir():
            print(f'[SKIP] 不是目录：{sub_dir}')
            continue
        if next(_iter_files(sub_dir), None) is None:
            # 明确要求：不要为了兼容创建空文件夹
            print(f'[SKIP] 子目录为空（不会创建空目录）：{sub_dir}')
            continue

        print(f'上传：{sub_dir}  ->  {cfg.repo_id}/{sub}')
        api.upload_folder(
            folder_path=str(sub_dir),
            repo_id=cfg.repo_id,
            repo_type=cfg.repo_type,
            path_in_repo=sub,
            commit_message=f'Upload {sub} from TacManip',
        )
        uploaded_any = True

    if not uploaded_any:
        raise RuntimeError(
            f'没有任何内容被上传：请检查 {local_path} 下是否存在且非空的 {cfg.include_subdirs}'
        )

    # 根据 meta/info.json 中的 codebase_version 给数据集打 tag（最小实现）
    info_path = local_path / 'meta' / 'info.json'
    if info_path.exists():
        with open(info_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        codebase_version = str(info.get('codebase_version', '')).strip()
        if codebase_version:
            api.create_tag(
                cfg.repo_id,
                tag=codebase_version,
                repo_type=cfg.repo_type,
            )

    print('\n上传完成！')
    print(f'数据集地址: https://huggingface.co/{cfg.repo_id}')


if __name__ == '__main__':
    main(_parse_args())
