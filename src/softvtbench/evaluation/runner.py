#!/usr/bin/env python
"""Unified evaluation entry point -- Isaac boots once, then loops over all task x demo in the suite.

Usage (inside the Isaac python environment):
  python runner.py --suite object_soft --policy pi05_vt_c \
      --episodes 1 --out OUTPUT_DIR [--tasks "0 1 2"] [--demo-offset 0] \
      [--server-port 9011]          # pi05 openpi server port
      [--worker-port 9101]          # act/dp serve_policy_worker port

All parameters (paths/physics/scene/language/checkpoint) are read from ../config/*.yaml;
nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from softvtbench.config import (
    load_benchmark_config,
    load_policy_manifest,
    repository_root,
)


def load_yaml(rel: str):
    """Compatibility name for benchmark-owned configuration."""
    return load_benchmark_config(rel)


def load_conditions(spec: str) -> list[tuple]:
    """Evaluation condition list -> [(label, ood_cfg, ood_hash), ...].

    One server + one env run ID and multiple OOD conditions back-to-back (saving the fixed
    cost of restarting the server/Isaac per condition).
    Condition file: one `<label> <authoritative ood json | ID> [level]` per line, `#`
    starts a comment; empty spec = ID only.
    Physical OOD modifies USD materials in place, so the runner restores the authored
    snapshot before every episode.
    """
    spec = os.path.expandvars(spec)
    if not spec.strip():
        return [("id", None, None)]
    from softvtbench.compat.ood_evaluation import load_ood_config_from_env
    out: list[tuple] = []
    for raw in Path(spec).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        label, src = parts[0], os.path.expandvars(parts[1])
        if src.upper() == "ID":
            out.append((label, None, None))
            continue
        os.environ["SOFTVTBENCH_OOD_CONFIG"] = src        # vendored module only reads env vars
        os.environ["SOFTVTBENCH_OOD_LEVEL"] = parts[2] if len(parts) > 2 else ""
        cfg, digest = load_ood_config_from_env()
        out.append((label, cfg, digest))
    os.environ.pop("SOFTVTBENCH_OOD_CONFIG", None)
    os.environ.pop("SOFTVTBENCH_OOD_LEVEL", None)
    labels = [c[0] for c in out]
    assert len(labels) == len(set(labels)), f"duplicate condition labels: {labels}"
    return out


def paired_episode_seed(run_seed: int, suite: str, task_id: int, episode: str,
                        condition: str) -> int:
    """Deterministic per-episode seed (used for DP's DDPM sampling).

    Same idea as the collection-time `sha1(f"W:{asset}:{demo_id}")`: depends only on
    content, not order, so the same episode gets identical sampling noise on any machine
    and in any position -> reproducible results.

    **condition is deliberately excluded from the key**: the same demo must receive the
    *same* policy sampling noise under ID and each OOD condition, otherwise OOD drops
    would mix in sampling-noise differences and ID/OOD would not be paired.
    (Fixed 7/26: the key originally included condition, making DP's OOD curves unusable
    for paired comparison.) The parameter is kept to avoid changing call-site signatures.
    """
    del condition                      # intentionally not part of the key, see docstring
    from softvtbench.evaluation.determinism import episode_seed
    return episode_seed(run_seed, suite, task_id, episode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, help="object_soft|spatial_soft|object_rigid|spatial_rigid")
    ap.add_argument("--policy", required=True, help="an id from policies.yaml")
    ap.add_argument("--tasks", default="", help='empty=all, or e.g. "0 1 2"')
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--demo-offset", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--server-host", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=0, help="pi05 openpi server")
    ap.add_argument("--worker-port", type=int, default=0, help="act/dp worker")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--num-envs", type=int, default=int(os.environ.get("SOFTVT_NUM_ENVS", "1")),
                    help="number of vectorized envs (openpi/replay only; ACT/DP workers are stateful, use process parallelism)")
    ap.add_argument("--conditions", default=os.environ.get("SOFTVT_CONDITIONS", ""),
                    help="condition file: one `<label> <ood json|ID> [level]` per line; empty=ID only")
    ap.add_argument("--no-headless", action="store_true")
    args = ap.parse_args()
    if args.num_envs != 1:
        raise SystemExit(
            "formal per-episode receipts currently require --num-envs 1; "
            "vector execution is disabled until every vector slot has independent reset/OOD evidence"
        )

    paths = load_yaml("paths.yaml")
    physics = load_yaml("physics.yaml")
    suite = load_yaml(f"suites/{args.suite}.yaml")
    pol_all = load_policy_manifest()["policies"]
    pol = next((p for p in pol_all if p["id"] == args.policy), None)
    if pol is None:
        raise SystemExit(f"unknown policy id {args.policy!r}; have {[p['id'] for p in pol_all]}")
    if pol["suite"] != args.suite:
        raise SystemExit(f"policy {args.policy} is for suite {pol['suite']}, not {args.suite}")
    evaluation_protocol = pol.get("evaluation_protocol")
    if evaluation_protocol not in {
        "diagnostic_replay", "native_env_steps", "chunked_30x10"
    }:
        raise SystemExit(
            f"policy {args.policy} has invalid/missing evaluation_protocol: "
            f"{evaluation_protocol!r}"
        )
    objects = {name: load_yaml(f"objects/{name}.yaml")
               for name in {t.get("object") for t in suite["tasks"] if t.get("object")}}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_args.json").write_text(json.dumps(
        {**vars(args), "policy_entry": pol}, indent=2, ensure_ascii=False))

    # ---- Isaac: boot once; add only the simulator extension and OpenPI client.
    # Model implementations stay in SoftVTBench-Models and never enter the Isaac
    # process. The client fallback supports sibling editable checkouts.
    benchmark_root = repository_root()
    models_root = Path(os.environ.get("SOFTVT_MODELS_ROOT", "")).expanduser()
    client_root = (
        models_root / "backends" / "openpi" / "packages" / "openpi-client" / "src"
    )
    runtime_paths = [benchmark_root / "source" / "tac_manip"]
    if client_root.is_dir():
        runtime_paths.append(client_root)
    for p in runtime_paths:
        p = str(p)
        if p not in sys.path:
            sys.path.append(p)
    from softvtbench.evaluation.envs.build import boot_app
    boot_app(headless=not args.no_headless)
    import tac_manip  # noqa: F401  registers gym environments (must come after AppLauncher)

    from softvtbench.evaluation.envs.build import (
        build_task_env,
        clamp_finger_lower_limit,
        episode_grip_width,
        load_episodes,
        load_initial_state,
        restore_physics,
        snapshot_dome_light,
        snapshot_target_physics,
    )
    conditions = load_conditions(args.conditions)   # [(label, ood_cfg, ood_hash), ...]; defaults to a single ID
    print(f"[runner] conditions: {[c[0] for c in conditions]}", flush=True)
    from softvtbench.evaluation.gripper_execution import (
        ABSOLUTE_GRIPPER_MODES,
        FIXED_POSITION_GRIPPER_MODES,
        RELATIVE_GRIPPER_MODES,
        GripperExecutor,
    )
    from softvtbench.evaluation.rollout import (
        apply_episode_ood,
        prepare_episode,
        run_episode,
    )
    from softvtbench.evaluation import metrics
    from softvtbench.evaluation import policies  # noqa: F401
    from softvtbench.evaluation.policies.base import make

    task_list = suite["tasks"]
    if args.tasks.strip():
        keep = {int(x) for x in args.tasks.split()}
        task_list = [t for t in task_list if t["id"] in keep]

    ge = pol.get("gripper_execution", {})
    total_width_tighten_m = float(ge.get("total_width_tighten_m", 0.0))
    if total_width_tighten_m and not (
        suite["name"].endswith("_soft")
        and pol["modality"] == "vo"
        and ge.get("mode") in FIXED_POSITION_GRIPPER_MODES
    ):
        raise ValueError(
            "total_width_tighten_m is restricted to soft VO fixed-position "
            "gripper execution"
        )
    results: list[dict] = []
    results_f = open(out_dir / "results.jsonl", "w")     # fail-closed: never mix in stale results
    seen_keys: set = set()

    def emit(row: dict) -> None:
        """Write one result row. (condition, task, episode) must be unique and complete --
        a missing field errors immediately instead of surfacing at stage aggregation
        (a missing condition field slipped through exactly that way on 7/26)."""
        key = (row["condition"], row["task_id"], row["episode"])
        assert None not in key, f"incomplete result row: {key}"
        assert key not in seen_keys, f"duplicate result row: {key}"
        seen_keys.add(key)
        results.append(row)
        results_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        results_f.flush()
    for task in task_list:
        handler, names = load_episodes(paths["data_root"] + "/" + suite["data_dir"], suite, task["id"])
        chosen = list(enumerate(names))[args.demo_offset:args.demo_offset + args.episodes]
        if len(chosen) < args.episodes:
            raise SystemExit(f"task{task['id']}: only {len(chosen)} demos at offset {args.demo_offset}, "
                             f"need {args.episodes} -- refusing to under-run silently")

        # The two spatial suites place (purely visual) distractors per demo -> build the
        # env once per task, then reposition visual assets via USD per episode (physics
        # bodies are restored by reset_to; avoids close() crashes).
        scene_params = None
        if suite.get("scene_params_file"):
            scene_text = Path(suite["scene_params_file"]).read_text()
            scene_params = json.loads(os.path.expandvars(scene_text))[str(task["id"])]
        obj = objects.get(task.get("object"))

        def ep_assets(ep_idx, ep_name):
            """Extra-assets list for this episode (incl. visual distractors); None if no per-demo assets."""
            if task.get("extra_assets_pattern"):
                return json.loads(Path(task["extra_assets_pattern"].format(episode=ep_idx)).read_text())
            if scene_params is not None:
                return json.loads(os.path.expandvars(scene_params[ep_name]["extra_assets_json"]))
            return None

        cur_task = dict(task)
        first_assets = ep_assets(*chosen[0])
        if first_assets is not None:                  # build the scene with the first episode's asset set
            cur_task["extra_assets_inline"] = json.dumps(first_assets)
            cur_task.pop("extra_assets_file", None)
        if scene_params is not None:                  # spatial_soft: task-level assets/scale come from scene_params
            sp0 = scene_params[chosen[0][1]]
            cur_task.update(asset_name=sp0["asset_name"], asset_spawn_pos=sp0["pos"],
                            asset_spawn_rot=sp0.get("rot", "1 0 0 0"),
                            asset_scale=sp0.get("scale", "1.0 1.0 1.0"),
                            usd_asset_override=sp0.get("target_usd_asset_id"))

        vec_n = args.num_envs if pol["backend"] in ("openpi", "replay") else 1
        if vec_n > 1 and pol["backend"] == "replay":
            vec_n = 1                      # replay uses a single env (self-check must be exact per episode)
        t0 = time.time()
        env, success_term = build_task_env(
            paths, suite, cur_task, obj, physics,
            gripper_action_mode="abs" if ge.get("mode") in ABSOLUTE_GRIPPER_MODES else "binary",
            seed=args.seed, num_envs=vec_n)
        print(f"[runner] task{task['id']} env built ({time.time()-t0:.0f}s) "
              f"episodes={[n for _, n in chosen]}", flush=True)
        # authored material snapshot: OOD apply_physical edits USD in place and irreversibly; restore per episode
        phys_snapshot = (
            snapshot_target_physics(env, cur_task["asset_name"])
            if cur_task.get("asset_name") else []
        )

        # Fail-closed scene preflight: after exactly restoring the first demo and before
        # constructing the policy or running any inference, read assets/lighting/cameras/
        # FEM back from the live USD stage. Missing fields and mismatches are equally hard failures.
        warm_name = chosen[0][1]
        hdf5_language = handler[f"data/{warm_name}"].attrs.get("language")
        if isinstance(hdf5_language, bytes):
            hdf5_language = hdf5_language.decode("utf-8")
        cur_task["_receipt_hdf5_language"] = (
            str(hdf5_language) if hdf5_language is not None else None
        )
        if first_assets is not None:
            from softvtbench.evaluation.envs.build import reposition_visual_assets
            reposition_visual_assets(env, first_assets)
        first_state = load_initial_state(
            handler, warm_name, task.get("removed_source_assets", ""), env.unwrapped.device)
        prepare_episode(env, first_state, suite=suite)
        from softvtbench.evaluation import receipt as receipt_mod
        rec = receipt_mod.scene_receipt(
            env, suite=suite, task=cur_task, paths=paths, policy_entry=pol,
            physics_config=physics)
        receipt_path = (out_dir / "receipt.json" if len(task_list) == 1
                        else out_dir / f"task{task['id']}" / "receipt.json")
        try:
            receipt_mod.assert_contract(rec, suite=suite, obj=obj)
        except Exception as exc:
            # Even on failure, keep the full evidence + contract.errors, not just logs.
            # Isaac teardown on exceptions can hang, so write to disk then hard-exit with a nonzero code.
            receipt_mod.write(receipt_path, rec)
            print(f"[runner] SCENE RECEIPT PREFLIGHT FAILED: {exc}", file=sys.stderr, flush=True)
            results_f.close()
            os._exit(42)
        receipt_mod.write(receipt_path, rec)
        print(f"[runner] task{task['id']} scene receipt preflight PASSED: "
              f"{receipt_path}", flush=True)
        light_snapshot = snapshot_dome_light()

        # Policy instance: pi05 depends on per-task language -> rebuild per task; act/dp have no language -> reuse
        if pol["backend"] == "openpi":
            policy = make("openpi", server_host=args.server_host, server_port=args.server_port,
                          modality=pol["modality"], language_instruction=task["language"],
                          replan_steps=int(suite["control"]["replan_steps"]),
                          mosaic_layout=suite.get("mosaic_layout", "rows"))
        elif pol["backend"] == "replay":     # open-loop replay: data/env pipeline self-check
            policy = make("replay", replan_steps=int(suite["control"]["replan_steps"]))
        else:
            policy = make("remote", port=args.worker_port, modality=pol["modality"])
        # fastwam (via the remote worker) takes language: sent once per task, loading the
        # matching t5 cache; openpi passes language via constructor args and has no such
        # method; act/dp workers return ok for compatibility.
        if hasattr(policy, "set_language"):
            policy.set_language(task["language"])

        gripper_exec = GripperExecutor(
            ge.get("mode", "binary"),
            open_finger=float(ge.get("open_finger", 0.04)),
            fixed_close_finger=float(ge.get("fixed_close_finger", 0.0)),
            closure_gain=float(ge.get("closure_gain", 1.0)),
            close_threshold=ge.get("close_threshold"),
            open_threshold=ge.get("open_threshold"),
            close_delta=ge.get("close_delta"),
            open_delta=ge.get("open_delta"),
            aperture_noise_tolerance=float(ge.get("aperture_noise_tolerance", 0.0005)),
            one_shot=ge.get("one_shot", False),
            min_hold_steps=int(ge.get("min_hold_steps", 10)))

        # Warmup episode (discarded): the first episode after env build is not the same
        # physics as later ones -- measured, the same demo run 4x gives 8.97/94 steps ->
        # 3.95/109 steps x3. Bare sim.step warmup does not help (the difference is in the
        # first reset sequence), so run one full throwaway episode to exactly reproduce
        # the "second and later" conditions.
        # Must come after policy/gripper_exec construction -- referencing them earlier
        # raises NameError, and Isaac's teardown swallows the exception, presenting as a
        # hung process (the cause of all 12 gate machines dying on 7/26).
        if pol["backend"] == "replay":
            policy.set_episode(handler[f"data/{warm_name}/"
                                       f"{'actions_binary' if ge.get('mode') == 'binary' else 'actions'}"][()])
        warm_w = episode_grip_width(handler, warm_name)
        if ge.get("mode") in RELATIVE_GRIPPER_MODES and warm_w is None:
            raise RuntimeError(
                f"{ge.get('mode')} requires episode gripper_width calibration; "
                f"{warm_name} has none"
            )
        warm_limit = None
        if warm_w is not None:
            warm_limit = clamp_finger_lower_limit(
                env, warm_w, total_width_tighten_m=total_width_tighten_m
            )
        run_episode(env, success_term, policy, gripper_exec,
                    load_initial_state(handler, warm_name,
                                       task.get("removed_source_assets", ""), env.unwrapped.device),
                    episode_seed=paired_episode_seed(
                        args.seed, args.suite, task["id"], warm_name, "warmup"
                    ),
                    suite=suite, control=suite["control"], grip_width=warm_w,
                    finger_lower_limit=warm_limit,
                    evaluation_protocol=evaluation_protocol,
                    fem_asset=(cur_task.get("asset_name")
                               if obj and obj.get("deformable") else None))
        restore_physics(phys_snapshot)
        restore_physics(light_snapshot)
        policy.reset()
        gripper_exec.reset()
        print(f"[runner] task{task['id']} warmup episode done", flush=True)

        # One server + one env run ID and each OOD condition back-to-back: saves the fixed
        # cost of restarting the server (~90s) + rebuilding the env (~20s) per condition.
        # Physical OOD edits USD in place; restore_physics keeps conditions from leaking
        # into each other.
        for cond_label, ood_cfg, ood_hash in conditions:
            if vec_n > 1:
                from softvtbench.evaluation.rollout_vec import run_batch
                mk_policy = lambda: make("openpi", server_host=args.server_host, server_port=args.server_port,
                                         modality=pol["modality"], language_instruction=task["language"],
                                         replan_steps=int(suite["control"]["replan_steps"]),
                                         mosaic_layout=suite.get("mosaic_layout", "rows"))
                mk_ge = lambda: GripperExecutor(
                    ge.get("mode", "binary"), open_finger=float(ge.get("open_finger", 0.04)),
                    fixed_close_finger=float(ge.get("fixed_close_finger", 0.0)),
                    closure_gain=float(ge.get("closure_gain", 1.0)),
                    close_threshold=ge.get("close_threshold"), open_threshold=ge.get("open_threshold"),
                    close_delta=ge.get("close_delta"), open_delta=ge.get("open_delta"),
                    aperture_noise_tolerance=float(
                        ge.get("aperture_noise_tolerance", 0.0005)
                    ),
                    one_shot=ge.get("one_shot", False),
                    min_hold_steps=int(ge.get("min_hold_steps", 10)))
                for g0 in range(0, len(chosen), vec_n):
                    group = chosen[g0:g0 + vec_n]
                    pad = vec_n - len(group)               # short last group: pad by repeating the first episode, discard those results
                    filled = group + group[:1] * pad
                    states, oods_g = [], []
                    for ep_idx, ep_name in filled:
                        states.append(load_initial_state(
                            handler, ep_name, task.get("removed_source_assets", ""), env.unwrapped.device))
                        o = None
                        if ood_cfg is not None:
                            from softvtbench.compat.ood_evaluation import make_ood_episode
                            o = make_ood_episode(ood_cfg, ood_hash, suite=suite["name"],
                                                 task_id=task["id"], episode_key=ep_name)
                        oods_g.append(o)
                    restore_physics(phys_snapshot)   # each batch starts from authored materials
                    widths, lower_limits = [], []
                    for i, (ep_idx, ep_name) in enumerate(filled):
                        a = ep_assets(ep_idx, ep_name)
                        if a is not None:                   # each env places its own episode's distractors
                            from softvtbench.evaluation.envs.build import reposition_visual_assets
                            reposition_visual_assets(env, a, idx=i)
                        w = episode_grip_width(handler, ep_name)   # collection-time safe gripper width (soft only)
                        widths.append(w)
                        if w is not None:
                            lower_limits.append(clamp_finger_lower_limit(
                                env, w, idx=i,
                                total_width_tighten_m=total_width_tighten_m
                            ))
                        else:
                            lower_limits.append(None)
                    rs = run_batch(env, success_term, [mk_policy() for _ in filled],
                                   [mk_ge() for _ in filled], states,
                                   episode_seeds=[paired_episode_seed(
                                       args.seed, args.suite, task["id"], nm, cond_label)
                                                  for _, nm in filled],
                                   suite=suite, control=suite["control"], oods=oods_g,
                                   finger_lower_limits=lower_limits,
                                   fem_asset=(cur_task.get("asset_name")
                                              if obj and obj.get("deformable") else None))
                    for j, ((ep_idx, ep_name), r) in enumerate(list(zip(group, rs))[:len(group)]):
                        r.update({"task_id": task["id"], "episode": ep_name, "policy": args.policy,
                                  "suite": args.suite, "gripper_mode": ge.get("mode", "binary"),
                                  "vec_n": vec_n, "grip_width_m": widths[j],
                                  "finger_lower_limit_m": lower_limits[j],
                                  "total_width_tighten_m": total_width_tighten_m,
                                  "condition": cond_label})
                        emit(r)
                        print(f"[runner] task{task['id']}/{ep_name}: "
                              f"{'SUCCESS' if r['success'] else 'fail'} steps={r['steps']} "
                              f"(vec,{cond_label})", flush=True)
                continue          # next condition

            for ep_idx, ep_name in chosen:
                assets = ep_assets(ep_idx, ep_name)
                if assets is not None:
                    from softvtbench.evaluation.envs.build import reposition_visual_assets
                    reposition_visual_assets(env, assets)
                initial_state = load_initial_state(
                    handler, ep_name, task.get("removed_source_assets", ""), env.unwrapped.device)
                restore_physics(phys_snapshot)      # each episode starts from authored materials (prevents compounding OOD drift)
                restore_physics(light_snapshot)     # lighting OOD likewise restored per episode to the collection baseline 135
                grip_width = episode_grip_width(handler, ep_name)
                if ge.get("mode") in RELATIVE_GRIPPER_MODES and grip_width is None:
                    raise RuntimeError(
                        f"{ge.get('mode')} requires episode gripper_width calibration; "
                        f"{ep_name} has none"
                    )
                finger_lower_limit = None
                if grip_width is not None:
                    finger_lower_limit = clamp_finger_lower_limit(
                        env, grip_width,
                        total_width_tighten_m=total_width_tighten_m
                    )
                if pol["backend"] == "replay":
                    key = "actions_binary" if ge.get("mode") == "binary" else "actions"
                    policy.set_episode(handler[f"data/{ep_name}/{key}"][()])
                ood = None
                if ood_cfg is not None:
                    from softvtbench.compat.ood_evaluation import make_ood_episode
                    ood = make_ood_episode(ood_cfg, ood_hash, suite=suite["name"],
                                           task_id=task["id"], episode_key=ep_name)
                seed = paired_episode_seed(
                    args.seed, args.suite, task["id"], ep_name, cond_label
                )
                debug_dir = str(
                    out_dir / "debug" / f"{cond_label}_task{task['id']}_{ep_name}"
                )
                # Exact episode preflight: perform the same reset sequence as
                # rollout, apply the condition, then read back demo/assets/OOD
                # before policy.reset()/observe()/predict() can send inference.
                prepared_obs = prepare_episode(env, initial_state, suite=suite)
                apply_episode_ood(env, ood, debug_dir)
                episode_task = dict(cur_task)
                if assets is not None:
                    episode_task["extra_assets_inline"] = json.dumps(assets)
                    episode_task.pop("extra_assets_file", None)
                scene_entry = scene_params[ep_name] if scene_params is not None else None
                ep_rec = receipt_mod.episode_receipt(
                    env,
                    suite=suite,
                    task=episode_task,
                    policy_entry=pol,
                    h5file=handler,
                    episode_name=ep_name,
                    condition=cond_label,
                    episode_seed=seed,
                    static_receipt=rec,
                    ood=ood,
                    grip_width=grip_width,
                    scene_params_entry=scene_entry,
                )
                safe_condition = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in cond_label
                )
                ep_receipt_rel = (
                    Path("episode_receipts")
                    / (
                        f"{safe_condition}_{hashlib.sha1(cond_label.encode()).hexdigest()[:8]}"
                        f"_task{task['id']}_{ep_name}.json"
                    )
                )
                ep_receipt_path = out_dir / ep_receipt_rel
                try:
                    receipt_mod.assert_episode_contract(ep_rec, suite=suite)
                except Exception as exc:
                    receipt_mod.write(ep_receipt_path, ep_rec)
                    print(
                        f"[runner] EPISODE RECEIPT PREFLIGHT FAILED: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    results_f.close()
                    os._exit(43)
                ep_receipt_sha = receipt_mod.write(ep_receipt_path, ep_rec)
                r = run_episode(env, success_term, policy, gripper_exec, initial_state, ood=ood,
                                episode_seed=seed,
                                grip_width=grip_width,
                                finger_lower_limit=finger_lower_limit,
                                suite=suite, control=suite["control"],
                                evaluation_protocol=evaluation_protocol,
                                fem_asset=(cur_task.get("asset_name")
                                           if obj and obj.get("deformable") else None),
                                debug_dir=debug_dir, prepared_obs=prepared_obs,
                                ood_already_applied=True)
                r.update({"task_id": task["id"], "episode": ep_name, "policy": args.policy,
                          "suite": args.suite, "gripper_mode": ge.get("mode", "binary"),
                          "evaluation_protocol": evaluation_protocol,
                          "grip_width_m": grip_width,
                          "finger_lower_limit_m": finger_lower_limit,
                          "total_width_tighten_m": total_width_tighten_m,
                          "condition": cond_label,
                          "episode_receipt": str(ep_receipt_rel),
                          "episode_receipt_sha256": ep_receipt_sha,
                          **({"ood": ood.frame_metadata()} if ood is not None and ood.enabled else {})})
                emit(r)
                print(f"[runner] task{task['id']}/{ep_name}: "
                      f"{'SUCCESS' if r['success'] else 'fail'} steps={r['steps']} "
                      f"d_peak={r['d_peak']} [{cond_label}]",
                      flush=True)
        handler.close()
        if task is not task_list[-1]:
            env.close()      # destroy in-process, next task rebuilds (App is not restarted; the last task ends via os._exit)

    summary = metrics.summarize(results)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    # Isaac's env.close()/destructor triggers a pybind weakref crash at shutdown (same
    # issue as the formal stack); results are already on disk, so hard-exit directly
    # (transcribes the SOFTVTBENCH_SKIP_ISAAC_CLEANUP_ON_EXIT semantics).
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
