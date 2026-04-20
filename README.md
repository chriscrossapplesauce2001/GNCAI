# MARL V-Formation Drone Control

Multi-agent reinforcement learning for 3 UAVs flying in a V-shape toward a navigation target while avoiding each other. Built with MAPPO (Multi-Agent PPO) on a custom PettingZoo environment in PyTorch.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick commands

```bash
python train.py --num-agents 3 --episodes 10000 --render-every 2000   # train from scratch
python train.py ... --resume checkpoints/best.pt                        # fine-tune
python evaluate.py  --checkpoint checkpoints/best.pt --num-agents 3    # 100-ep stats + plots
python interactive.py --checkpoint checkpoints/best.pt --num-agents 3   # pygame explorer
tensorboard --logdir runs/                                              # training metrics
```

Any config value wrapped in `_override()` (see `env/formation_config.py`) can be set via `FORMATION_<NAME>=<value>` env var. Example:
```bash
FORMATION_ENTROPY_COEF=0.0 FORMATION_COLLISION_PENALTY=5.0 python train.py ...
```

---

## Environment

**`env/formation_env.py`** — PettingZoo `ParallelEnv`, 100×100 2D world, Euler integration.

| Item | Value | Notes |
|---|---|---|
| World size | 100 × 100 | `WORLD_SIZE` |
| Physics timestep (DT) | 0.1 s | `pos += vel·DT`, `vel += accel·DT` |
| Max speed | 5.0 u/s | hard clamp on `‖v‖` |
| Agent radius | 1.5 u | collision threshold between two agents = 3.0 |
| Max steps / episode | 400 | truncation |
| Target threshold | 12.0 u | all agents within → mission complete |
| Spawn region | x,y ∈ [10, 30] | then agents scatter ±7 around that point |
| Target region | x,y ∈ [20, 80] | re-sampled until ≥40 units from spawn centroid |
| Wall behavior | bounce at radius | velocity × −0.5 on wall contact |

**Observation (per agent, 20-dim, normalized to [-1, 1]):**

| idx | meaning |
|---|---|
| 0-1 | own position (centered) |
| 2-3 | own velocity |
| 4-5 | Δ to nearest neighbor |
| 6-7 | Δ to 2nd nearest neighbor |
| 8-9 | Δ to own formation slot (centroid + offset) |
| 10-11 | Δ to navigation target |
| 12-15 | Δ + velocity of nearest obstacle |
| 16-19 | Δ + velocity of 2nd nearest obstacle |

Fixed at 20 dims so weights transfer when agent count changes (curriculum).

**Action (2-dim):** `(ax, ay)` ∈ [-1, 1] — 2D acceleration, scaled by physics.

**V-formation offsets (3 agents, world frame, never rotated):**
- Agent 0 (leader): `(+5, 0)` — front
- Agent 1 (L wing): `(-5, -5)`
- Agent 2 (R wing): `(-5, +5)`

---

## Reward function

Per agent, per step (`env/formation_env.py::_compute_reward`):

```
approach_reward  = APPROACH_WEIGHT  ·  Δdistance_to_target / (MAX_SPEED·DT)     # range [-APPROACH_WEIGHT, +APPROACH_WEIGHT]
proximity_penalty = COLLISION_PENALTY · max(0, (SAFE_DIST − min_neighbor_dist) / SAFE_DIST)
reward = approach_reward − proximity_penalty
```

On episode termination when all agents within `TARGET_THRESHOLD`:
```
reward += COMPLETION_BONUS   # per agent
```

**Currently dead (defined but not wired):** `FORMATION_WEIGHT`, `TIME_PENALTY`, `OBSTACLE_PENALTY`, `SMOOTHNESS_WEIGHT`, `_get_formation_error()`. Setting them has no effect unless the reward function is extended.

---

## Hyperparameters

All in `env/formation_config.py`. Values marked ✦ are env-var-overridable via `FORMATION_<NAME>`.

### Reward weights
| Name | Default | Effect |
|---|---|---|
| `APPROACH_WEIGHT` ✦ | 2.0 | Scale of the "move toward target" gradient. Bigger → stronger pull. +2/step max. |
| `COLLISION_PENALTY` ✦ | 2.0 | Scale of the proximity penalty. Bigger → more spacing, but >8 breaks training. |
| `SAFE_DIST` ✦ | 5.0 | Distance at which proximity penalty starts ramping. Setting > spawn radius (±7) unfairly punishes spawn geometry. |
| `COMPLETION_BONUS` ✦ | 500.0 | One-time bonus (per agent) when all reach target. Dominates per-step signals by design. |

### PPO
| Name | Default | Effect |
|---|---|---|
| `LEARNING_RATE` ✦ | 1e-4 | Adam LR for actor + critic. Use 3e-5 for long fine-tunes to prevent policy drift. |
| `GAMMA` | 0.99 | Discount factor. 0.99 ≈ effective horizon of ~100 steps. |
| `GAE_LAMBDA` | 0.95 | GAE bias-variance tradeoff. 0.95 is the PPO default. |
| `CLIP_EPSILON` | 0.2 | PPO ratio-clip range. Caps policy update magnitude. |
| `PPO_EPOCHS` ✦ | 10 | Optimization passes per batch of rollouts. |
| `ENTROPY_COEF` ✦ | 0.01 | Entropy bonus weight. **Setting to 0.0 eliminates late-stage policy drift** (important — see Lessons). |
| `VALUE_LOSS_COEF` | 1.0 | Scales the critic loss. |
| `EPISODES_PER_UPDATE` | 10 | Rollouts collected before each PPO update. |
| `BATCH_SIZE` | 0 | 0 = use the full rollout as a single batch (MAPPO default). |

### Architecture / world
| Name | Default | Effect |
|---|---|---|
| `OBS_DIM` | 20 | Observation vector size. Changing breaks checkpoints. |
| `ACTION_DIM` | 2 | 2D acceleration. |
| `HIDDEN_DIM` | 128 | Hidden width of actor and critic MLPs. |
| `NUM_AGENTS` | 3 | Swarm size. Formation offsets defined for 3 and 5 only. |

---

## Network architecture

**Actor** (`algo/actor.py`) — shared across all agents, decentralized at execution.
```
obs (20) → Linear(20→128) → ReLU → Linear(128→128) → ReLU → Linear(128→2) → tanh
                                                            ↓
                                                learnable log_std ∈ [-2.0, 0.5]
                                                            ↓
                                                   Normal(μ, σ), tanh-squashed
```
- Orthogonal init (`gain=√2` hidden, `gain=0.01` output head).
- Log-prob is corrected for the tanh squash (SAC paper, Appendix C).
- Total params ≈ 19k.

**Critic** (`algo/critic.py`) — centralized (sees all agents), training-only.
```
global_state (num_agents × 20 = 60) → Linear(60→128) → ReLU → Linear(128→128) → ReLU → Linear(128→1)
```
- Orthogonal init, output `gain=1.0`.
- Value targets normalized by running mean/std (`ValueNormalizer` in `algo/mappo.py`).

**CTDE split:** at training time the critic reads the global state (concat of all 3 obs); at deployment only the actor runs per-agent, no communication.

---

## Training loop (`algo/mappo.py`)

1. **Collect `EPISODES_PER_UPDATE=10` episodes.** Each step: every agent queries the shared actor; env steps once with the combined action dict; transitions stored in `RolloutBuffer`.
2. **Compute GAE advantages + returns** (`algo/buffer.py::compute_returns_and_advantages`) going backwards through the buffer with `GAMMA` and `GAE_LAMBDA`.
3. **PPO update for `PPO_EPOCHS=10`**:
   - Actor loss = PPO clipped objective (`CLIP_EPSILON=0.2`) − `ENTROPY_COEF` · entropy
   - Critic loss = Huber(δ=10) on value-normalized returns
   - Grad clip to max-norm 10 for both
4. Log to TensorBoard + print summary every 50 eps. Save `checkpoint_N.pt` every 200 eps; overwrite `best.pt` whenever episode reward is a new high.

---

## Reproducing the best result

Best model so far: **100% completion, 52 collisions / ep** at ep ~33000 of the 60k-episode run.

```bash
FORMATION_ENTROPY_COEF=0.0 \
FORMATION_COLLISION_PENALTY=5.0 \
FORMATION_LEARNING_RATE=3e-5 \
python train.py --num-agents 3 --episodes 60000 \
    --resume checkpoints_run5_pen5/best.pt --render-every 10000
```
- Resumes from the 100%/67 model (`checkpoints_run5_pen5/best.pt`).
- `ENTROPY_COEF=0.0` prevents the late-training policy collapse.
- `COLLISION_PENALTY=5.0` bumps spacing pressure without breaking training (3 and 5 work; ≥8 breaks).
- `LEARNING_RATE=3e-5` keeps the long fine-tune stable. Best snapshot is around ep 33k; after that the policy drifts.

Use `checkpoints/best.pt` (it captures the peak automatically).

---

## Key lessons (from actual runs)

- **Default `ENTROPY_COEF=0.01` causes policy collapse late in training** when the reward surface flattens — `log_std` drifts to its clamp ceiling, entropy climbs, reward falls. Fix: `FORMATION_ENTROPY_COEF=0.0`.
- **Huge collision penalties break training** — `COLLISION_PENALTY=8.0, SAFE_DIST=10.0` stuck the policy at 30% completion because the `SAFE_DIST=10` zone overlapped the spawn radius (±7). The baseline reward becomes ~−960/ep from spawn geometry alone, drowning out the learning signal. Keep `SAFE_DIST ≤ 7`.
- **Long fine-tunes drift without LR decay.** Even at `LR=3e-5`, a 60k-ep fine-tune peaked at ep 33k then regressed back toward worse performance. `best.pt` is load-bearing.
- **Scalar penalty bumps alone can't push collisions below ~50.** Changing `COLLISION_PENALTY` from 2→3→5→10 got us from ~90 to ~52, but penalty=10 was no better than 5. Further gains need structural changes (formation reward, stable-arrival check).
- **The sweep used a misleading objective.** `sweep_optuna.py::score = completion·100 − 2·collisions` — `mean_collisions` is counted per agent-step, so one persistent overlap inflates the number hugely. Treat sweep scores as coarse.

---

## Where this env sits in the landscape

The env in this project is an **intentionally minimal 2D custom env** — point masses, no rotation, no sensors, no aero, no ROS. That's a feature: you can iterate on the RL algorithm in seconds-per-episode without fighting a simulator. When you're ready to move up, here's the progression.

### 2D MARL benchmarks (same complexity class, standardized)

| Benchmark | What it is | Why you'd switch |
|---|---|---|
| **PettingZoo MPE** (`simple_spread`) | 2D particle env almost identical to this one | Published baselines; compare your algorithm to papers |
| **PettingZoo MAgent2** | Thousands of agents, grid-world battles | Scaling MARL algorithms to many agents |
| **SMAC / SMACv2** | StarCraft II mini-games | De-facto MARL benchmark; discrete actions, partial obs |
| **Melting Pot (DeepMind)** | Social-dilemma / cooperative scenarios | Generalization across partners |

### 3D drone simulators

The step up from "2D particle" to "actual drone" is big. You now deal with **6-DOF rigid body** (pos + orientation), **quaternion dynamics**, and a **PID inner-loop** (the RL policy typically outputs target velocity or attitude setpoints, not raw motor commands). Observations can include IMU, GPS, depth cams, LiDAR.

| Simulator | Physics | Speed | Render | RL-native? | Sweet spot |
|---|---|---|---|---|---|
| **gym-pybullet-drones** | PyBullet (rigid body) | ~1000 Hz | OpenGL | Yes (Gymnasium API) | Single/multi-drone RL, fast iteration |
| **MuJoCo + Menagerie drones** | MuJoCo (fast, accurate) | ~2000 Hz | OpenGL | Yes (Gymnasium/dm_control) | Low-level control RL, sim2real research |
| **Isaac Lab (NVIDIA)** | PhysX on GPU | **100k+ Hz** (parallel envs) | Omniverse | Yes (RL APIs built-in) | Massive parallel training, modern SOTA |
| **Flightmare (ETH)** | Unity render + Flightlib physics | ~20× real-time | Unity (photorealistic) | Yes | Vision-based control, drone racing |
| **Gazebo + PX4 SITL** | ODE/Bullet | ~real-time | OGRE | Indirectly (via ROS) | Pre-hardware validation, production SITL |
| **AirSim** (deprecated, still used) | PhysX | ~real-time | Unreal (photoreal) | Via plugin | Vision-heavy research (cameras, semantic seg) |

**The axis to optimize:**
- **Speed** → Isaac Lab or MuJoCo. Days → minutes of wall-clock training.
- **Fidelity / sim2real** → Gazebo+PX4. This is what your drone's firmware actually runs under.
- **Photorealism** (vision RL) → Flightmare, AirSim, or Isaac Sim with Omniverse RTX.
- **Ease / getting started** → `gym-pybullet-drones`. Works out of pip, multi-agent ready, closest-in-spirit bump from this project.

### What porting this project to 3D would actually involve

1. **State/action grows.** Obs goes from 20-dim to ~30-50 (add orientation quaternion, angular velocity, maybe IMU noise). Action goes from 2D accel to 3D velocity or 4D attitude+thrust.
2. **Normalization changes.** 2D world is 100×100; 3D is typically 20×20×10 m with metric units. Redo all scalings.
3. **Formation offsets become 3D.** V-shape in a horizontal plane is easy; true 3D formations (echelon, diamond) need thought.
4. **Collisions must include altitude.** Current proximity penalty uses 2D distance; 3D needs full 3-vector norms and possibly separate z-tolerance (drones stack vertically easier than horizontally).
5. **The MAPPO code mostly stays the same.** Actor/critic arch, PPO update, CTDE structure — all generic. You re-tune hyperparams (`LEARNING_RATE`, `ENTROPY_COEF`, reward weights) because the scale changes, but the algorithm works identically.
6. **Sim2real gap.** If the goal is a real drone: train in high-fidelity sim (Isaac or Gazebo-PX4) with domain randomization (motor noise, wind, mass perturbation), validate in SITL, then deploy. The RL policy typically outputs setpoints that a PX4 inner-loop tracks — don't have RL drive motors directly.

**Recommended next hop for this project:** `gym-pybullet-drones` — same MARL structure, 3D physics, Gymnasium API so the MAPPO code here drops in with minimal changes.

---

## Files

| Path | Role |
|---|---|
| `env/formation_env.py` | PettingZoo env: physics, observation builder, reward, termination |
| `env/formation_config.py` | All hyperparameters, env-var overrides, V-formation offsets |
| `algo/actor.py` | Policy network (tanh-squashed Gaussian, shared across agents) |
| `algo/critic.py` | Centralized value network (sees concatenated obs) |
| `algo/buffer.py` | Rollout buffer + GAE computation |
| `algo/mappo.py` | Training loop, PPO clipped update, checkpointing, value normalization |
| `train.py` | CLI trainer — used for every run in this project |
| `curriculum_train.py` | 3-stage curriculum (3 agents → 5 agents → 5 agents + obstacles) |
| `evaluate.py` | 100-ep deterministic eval + 4-panel matplotlib plot |
| `interactive.py` | Pygame explorer — drag agents, place obstacles, take manual control |
| `visualize.py` | Pygame renderer used by `train.py` and `interactive.py` |
| `sweep_optuna.py` | TPE hyperparam sweep (see "Key lessons" caveat) |

Checkpoints from successful runs live in `checkpoints_run*` folders; superseded ones are archived under `_superseded/`.
