"""
Optuna Sweep — Tune Reward Weights and PPO Epochs

Single-objective TPE search over the reward knobs that matter for the
"don't collide, everyone reaches target" task. Each trial launches train.py
as a subprocess with FORMATION_* env-var overrides and reads back a metrics
JSON. SQLite storage makes the study resumable — kill and restart freely.

Usage:
    # Overnight sweep (7 h), 2 parallel trials, 1000 eps each
    python sweep_optuna.py

    # Tune budget
    python sweep_optuna.py --timeout 7200 --episodes 1500 --n-jobs 1

    # Resume an existing study
    python sweep_optuna.py --study-name formation_v1

    # Just run the final long-training pass on the best params found so far
    python sweep_optuna.py --final-only

Output layout (under --out-dir, default sweep_results/):
    study.db            — Optuna SQLite storage (resumable)
    trial_{n}/          — per-trial checkpoints and metrics.json
    summary.csv         — one row per completed trial
    best_config.json    — best params + metrics
    best_final/         — long-training checkpoint trained with best params
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable

# Parameters exposed via FORMATION_* env vars in env/formation_config.py
TUNABLE_PARAMS = [
    "COLLISION_PENALTY",
    "SAFE_DIST",
    "APPROACH_WEIGHT",
    "PPO_EPOCHS",
]


def suggest_params(trial: optuna.Trial) -> dict:
    """Search space — only reward weights + PPO epochs, per user scope."""
    return {
        "COLLISION_PENALTY": trial.suggest_float("COLLISION_PENALTY", 2.0, 50.0, log=True),
        "SAFE_DIST": trial.suggest_float("SAFE_DIST", 5.0, 12.0),
        "APPROACH_WEIGHT": trial.suggest_float("APPROACH_WEIGHT", 0.5, 4.0),
        "PPO_EPOCHS": trial.suggest_categorical("PPO_EPOCHS", [3, 4, 5, 7, 10]),
    }


def score(metrics: dict) -> float:
    """
    Higher is better. Encodes user priority: "no collisions AND reach target".
    completion_rate ∈ [0, 1] scaled to 100; collisions/ep subtracted 2x so that
    every collision costs more than reaching the target gains.
    """
    return metrics["completion_rate_last_n"] * 100.0 - 2.0 * metrics["mean_collisions_last_n"]


def run_trial(trial_dir: Path, params: dict, episodes: int, num_agents: int) -> dict:
    trial_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = trial_dir / "metrics.json"

    env = os.environ.copy()
    for k, v in params.items():
        env[f"FORMATION_{k}"] = str(v)

    cmd = [
        PYTHON, str(HERE / "train.py"),
        "--num-agents", str(num_agents),
        "--episodes", str(episodes),
        "--checkpoint-dir", str(trial_dir / "ckpt"),
        "--log-dir", str(trial_dir / "tb"),
        "--metrics-out", str(metrics_path),
        "--quiet",
    ]

    log_path = trial_dir / "train.log"
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)

    if result.returncode != 0 or not metrics_path.exists():
        raise RuntimeError(f"Trial failed (rc={result.returncode}); see {log_path}")

    with open(metrics_path) as f:
        return json.load(f)


def make_objective(out_dir: Path, episodes: int, num_agents: int):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        trial_dir = out_dir / f"trial_{trial.number:04d}"
        trial.set_user_attr("trial_dir", str(trial_dir))

        metrics = run_trial(trial_dir, params, episodes, num_agents)
        for k, v in metrics.items():
            trial.set_user_attr(k, v)

        s = score(metrics)
        print(f"  [trial {trial.number:3d}] score={s:+7.2f}  "
              f"comp={metrics['completion_rate_last_n']*100:5.1f}%  "
              f"coll={metrics['mean_collisions_last_n']:5.1f}  "
              f"params={ {k: round(v, 3) if isinstance(v, float) else v for k, v in params.items()} }",
              flush=True)
        return s

    return objective


def dump_summary_csv(study: optuna.Study, out_csv: Path) -> None:
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": t.number, "score": t.value}
        row.update(t.params)
        for k in ("completion_rate_last_n", "mean_collisions_last_n",
                  "mean_reward_last_n", "mean_form_error_last_n",
                  "mean_distance_last_n"):
            row[k] = t.user_attrs.get(k)
        rows.append(row)
    if not rows:
        return
    rows.sort(key=lambda r: r["score"], reverse=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_final(out_dir: Path, params: dict, episodes: int, num_agents: int) -> dict:
    print(f"\n[final] training best config for {episodes} episodes: {params}", flush=True)
    final_dir = out_dir / "best_final"
    metrics = run_trial(final_dir, params, episodes, num_agents)
    metrics["params"] = params
    with open(out_dir / "best_config.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="sweep_results")
    p.add_argument("--study-name", default="formation_v1")
    p.add_argument("--num-agents", type=int, default=3)
    p.add_argument("--episodes", type=int, default=1000, help="Episodes per trial")
    p.add_argument("--final-episodes", type=int, default=3000,
                   help="Episodes for the final-best retraining run")
    p.add_argument("--timeout", type=int, default=7 * 3600, help="Sweep time budget (s)")
    p.add_argument("--n-trials", type=int, default=None, help="Optional trial count cap")
    p.add_argument("--n-jobs", type=int, default=2, help="Parallel trials")
    p.add_argument("--final-only", action="store_true",
                   help="Skip search; just retrain the current best params")
    p.add_argument("--skip-final", action="store_true",
                   help="Skip the long-training pass after the search")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{out_dir / 'study.db'}"
    sampler = TPESampler(seed=42, n_startup_trials=10, multivariate=True)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )

    # Seed the search with the current baseline and the manually-reasoned tune
    # (only if this study has no trials yet, to avoid duplicating on resume).
    if not study.trials:
        study.enqueue_trial({
            "COLLISION_PENALTY": 2.0, "SAFE_DIST": 5.0,
            "APPROACH_WEIGHT": 2.0, "PPO_EPOCHS": 10,
        })
        study.enqueue_trial({
            "COLLISION_PENALTY": 15.0, "SAFE_DIST": 8.0,
            "APPROACH_WEIGHT": 2.0, "PPO_EPOCHS": 4,
        })
        study.enqueue_trial({
            "COLLISION_PENALTY": 30.0, "SAFE_DIST": 10.0,
            "APPROACH_WEIGHT": 1.5, "PPO_EPOCHS": 4,
        })

    if not args.final_only:
        print(f"[sweep] study={args.study_name} storage={storage}")
        print(f"[sweep] budget: {args.timeout}s  n_jobs={args.n_jobs}  "
              f"eps/trial={args.episodes}")
        t0 = time.time()
        objective = make_objective(out_dir, args.episodes, args.num_agents)
        try:
            study.optimize(
                objective,
                n_trials=args.n_trials,
                timeout=args.timeout,
                n_jobs=args.n_jobs,
                gc_after_trial=True,
                show_progress_bar=False,
            )
        except KeyboardInterrupt:
            print("\n[sweep] interrupted, writing summary…", flush=True)

        dump_summary_csv(study, out_dir / "summary.csv")
        elapsed = time.time() - t0
        print(f"\n[sweep] done in {elapsed/60:.1f} min  "
              f"completed={len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}",
              flush=True)

    try:
        best = study.best_trial
    except ValueError:
        print("[sweep] no completed trials yet; nothing to retrain.")
        return

    print(f"\n[sweep] best trial #{best.number}  score={best.value:+.2f}  "
          f"comp={best.user_attrs.get('completion_rate_last_n', 0)*100:.1f}%  "
          f"coll={best.user_attrs.get('mean_collisions_last_n', 0):.1f}")
    print(f"[sweep] best params: {best.params}")

    if args.skip_final:
        with open(out_dir / "best_config.json", "w") as f:
            json.dump({"params": best.params, **best.user_attrs}, f, indent=2)
        return

    run_final(out_dir, best.params, args.final_episodes, args.num_agents)
    print(f"[sweep] final checkpoint: {out_dir / 'best_final' / 'ckpt' / 'final.pt'}")


if __name__ == "__main__":
    main()
