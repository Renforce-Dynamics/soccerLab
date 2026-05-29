# SoccerLab — mjlab Backend Port (As Built)

End-to-end notes for the mjlab (MuJoCo-Warp) port of soccerLab. Companion
to the top-level README. Covers install, asset choices, the two ported
tasks, AMP integration, the upstream beyondAMP bugs we patched along the
way, and the train wrapper.

The IsaacLab side under `source/soccerTask/` is unchanged by this work —
the mjlab path lives in `source/soccer_tasks_mjlab/` alongside it.

---

## 1. Install

### 1.1 mjlab core (PyPI)

```bash
pip install --user mjlab   # tested against 1.4.0
```

Pulled-in versions (will upgrade your system if older):

| Package | Required | Note |
|:---|:---|:---|
| torch | >= 2.7.0 | upgrades from 2.4.x |
| mujoco | ~= 3.8.0 | upgrades from 3.3.x |
| mujoco-warp | any | MuJoCo-Warp GPU backend |
| warp-lang | >= 1.12.0 | NVIDIA Warp compute |
| rsl-rl-lib | == 5.2.0 | base PPO trainer |

### 1.2 beyondAMP (mjlab backend + AMP runner)

beyondAMP ships both an IsaacLab and an mjlab backend in one repo. We
vendor it under `source/beyondAMP/` so the patches stay reproducible:

```bash
cd soccerLab/source/beyondAMP
pip install -e source/beyondAMP source/amp_tasks_mjlab source/rsl_rl_amp
```

Upstream: <https://github.com/Renforce-Dynamics/beyondAMP> (master).

Sanity check:
```bash
python -c "import mjlab, beyondAMP; from beyondAMP.mjlab.rsl_rl import AMPEnvWrapper, AMPRunnerCfg; print('ok')"
```

### 1.3 soccer_tasks_mjlab

```bash
pip install -e soccerLab/source/soccer_tasks_mjlab
```

Registers the four kick variants + the dribble task at import time.

---

## 2. Tasks (as registered)

Tasks stay **separated by robot** — kicking on G1, dribbling on T1. No
robot unification was attempted; the original training rigs are kept.

| Task ID | Robot | Algo | Source repo |
|:---|:---|:---|:---|
| `Soccer-Mjlab-Dribble-Flat-T1` | T1 (14-DoF subset) | pure PPO | BackupDribbling |
| `Soccer-Mjlab-Kick-Bootstrap-G1` | G1 (29-DoF) | AMP-PPO (A0→A1) | G1_kicking |
| `Soccer-Mjlab-Kick-AMP-G1` | G1 (29-DoF) | AMP-PPO (Stage A) | G1_kicking |
| `Soccer-Mjlab-Kick-StageB-G1` | G1 (29-DoF) | AMP-PPO (Stage B) | G1_kicking |
| `Soccer-Mjlab-Kick-StageC-G1` | G1 (29-DoF) | AMP-PPO (Stage C) | G1_kicking |

`Mjlab-Velocity-Flat-Unitree-G1` (and the rough variant) come from mjlab
itself and are usable out of the box — no soccerLab code required.

### 2.1 Layout

```
soccerLab/
  source/
    soccer_tasks_mjlab/
      soccer_tasks_mjlab/
        __init__.py           # imports dribbling + kicking
        dribbling/
          t1/
            __init__.py       # register Soccer-Mjlab-Dribble-Flat-T1
            dribbling_env_cfg.py
            agents/ppo_cfg.py
            mdp/{observations,rewards}.py
        kicking/
          g1/
            __init__.py       # register 4 Kick variants
            env_cfg.py        # 4 factories: basic / bootstrap / stage_b / stage_c
            agents/amp_ppo_cfg.py
            mdp/{commands,curriculums,events,observations,rewards,terminations}.py
  data/
    assets/
      t1/                     # T1 23dof MJCF + 63 STL meshes (16 MB)
      ball/soccer_ball.xml    # shared FIFA ball: r=0.115, m=0.43
    datasets/
      g1_kick_skill/wo_cf_shoot_74_06.npz   # AMP motion (101 frames × 29 DoF, ~179 KB)
  scripts/factoryMjlab/
    train.py                  # AMP-aware launcher (see §5)
```

### 2.2 Observation / action dimensions

| Task | Action dim | Actor obs | Critic obs | AMP obs |
|:---|:---:|:---:|:---:|:---:|
| Dribble-Flat-T1 | 14 | 77 | 77 | — |
| Kick-Bootstrap-G1 | 29 | 112 | 448 | 58 |
| Kick-AMP-G1 / StageB / StageC | 29 | 112 | 448 | 58 |

---

## 3. Robot / asset choices

### 3.1 G1 — use mjlab's bundled MJCF

All three known G1 sources (mjlab MJCF, G1_kicking URDF, G1_kicking USD)
share the **same 29 joint names** and motor specs (5020 / 7520-14 /
7520-22 / 4010). We standardise on mjlab's bundled
`asset_zoo/robots/unitree_g1/xmls/g1.xml` because:

1. mjlab-first port — native MJCF avoids URDF→USD→MJCF conversion drift.
2. `g1_constants.py` derives armature from planetary-gear specs instead
   of hardcoding it.
3. Richer collision model: 33 collision geoms (7 per foot) vs 31 in the
   URDF.
4. Joint names match the URDF/USD exactly — no remapping needed if the
   IsaacLab side ever wants to share a checkpoint format.

Joint groups (identical across sources):
- **Legs (12)**: `{left,right}_hip_{pitch,roll,yaw}_joint`, `{left,right}_knee_joint`, `{left,right}_ankle_{pitch,roll}_joint`
- **Waist (3)**: `waist_{yaw,roll,pitch}_joint`
- **Arms (14)**: `{left,right}_shoulder_{pitch,roll,yaw}_joint`, `{left,right}_elbow_joint`, `{left,right}_wrist_{roll,pitch,yaw}_joint`

One mjlab-ism: waist pitch/roll and the ankles are modelled as 4-bar
linkages and get doubled actuator stiffness/damping/effort/armature.

### 3.2 T1 — shipped as MJCF in this repo

mjlab does not ship T1, so we vendor it at `data/assets/t1/` (23 DoF
MJCF + 63 STL meshes, ~16 MB). The dribble task actuates 14 joints
(12 lower body + 2 head) and capitalises joint names (`Left_Hip_Pitch`)
to match the T1 MJCF — these names are baked into
`joint_pos_reward_stage1`'s T1 mapping.

### 3.3 Ball — one shared MJCF for both tasks

`data/assets/ball/soccer_ball.xml` — FIFA spec sphere
(r = 0.115 m, m = 0.43 kg). Both dribbling and kicking reference it.

### 3.4 AMP motion (kicking only)

`data/datasets/g1_kick_skill/wo_cf_shoot_74_06.npz` — single 101-frame
right-foot shoot clip from G1_kicking, full G1 state:

```
joint_pos      (101, 29)
joint_vel      (101, 29)
body_pos_w     (101, 30, 3)
body_quat_w    (101, 30, 4)
body_lin_vel_w (101, 30, 3)
body_ang_vel_w (101, 30, 3)
fps            scalar
```

The dribbling task uses **no motion reference** — pure PPO with a hand-
designed gait-phase clock observation (sin/cos, cycle_time = 0.8 s).

---

## 4. IsaacLab → mjlab API mapping (cheat sheet)

Came up repeatedly during the port; keep handy when adapting more
IsaacLab code:

| IsaacLab | mjlab |
|:---|:---|
| `robot.data.body_pos_w` | `robot.data.body_link_pos_w` |
| `robot.data.body_quat_w` | `robot.data.body_link_quat_w` |
| `robot.data.body_lin_vel_w` | `robot.data.body_link_lin_vel_w` |
| `robot.data.body_ang_vel_w` | `robot.data.body_link_ang_vel_w` |
| `robot.data.root_pos_w` | `robot.data.root_link_pos_w` |
| `robot.data.root_quat_w` | `robot.data.root_link_quat_w` |
| `robot.data.root_lin_vel_w` | `robot.data.root_link_lin_vel_w` |
| `robot.data.root_ang_vel_w` | `robot.data.root_link_ang_vel_w` |
| `robot.data.soft_joint_pos_limits` | `robot.data.joint_pos_limits` |
| `Entity.device` | `env.device` |
| `write_root_pose_to_sim(...)` | `write_root_link_pose_to_sim(...)` |
| `write_root_velocity_to_sim(...)` | `write_root_link_velocity_to_sim(...)` |
| `@configclass` | `@dataclass(kw_only=True)` |
| `ManagerBasedRLEnvCfg` | `ManagerBasedRlEnvCfg` |
| `ObsGroup` (class attrs) | `ObservationGroupCfg(terms={...})` |
| `CommandTermCfg.class_type` | `CommandTermCfg.build()` |
| `gym.register(...)` | `register_mjlab_task(...)` |
| Obs group name `"policy"` | Obs group name `"actor"` |

Pure-PyTorch terms (reward functions, AMP discriminator, curriculum
logic) port unchanged. The bulk of the work is plumbing class names and
field access patterns.

---

## 5. Train wrapper (`scripts/factoryMjlab/train.py`)

The wrapper inspects the task's `rl_cfg`. If it's an `AMPRunnerCfg`, it
wraps the env in `AMPEnvWrapper` and uses `AMPOnPolicyRunner` from
`rsl_rl_amp`; otherwise it falls back to mjlab's
`MjlabOnPolicyRunner` + `RslRlVecEnvWrapper`. Before that fix the wrapper
unconditionally went down the standard path and crashed at runner
construction when it hit `'AMPPPO'`.

```bash
# Pure-PPO (dribbling)
python scripts/factoryMjlab/train.py Soccer-Mjlab-Dribble-Flat-T1 \
    --headless --num_envs 64

# AMP (kicking)
python scripts/factoryMjlab/train.py Soccer-Mjlab-Kick-Bootstrap-G1 \
    --agent.amp-data.motion-files data/datasets/g1_kick_skill/wo_cf_shoot_74_06.npz \
    --headless --num_envs 64
```

---

## 6. Upstream beyondAMP patches (pushed to master)

Three real bugs in `rsl_rl_amp` were blocking end-to-end AMP training on
the mjlab backend. All three are fixed on `master` of
<https://github.com/Renforce-Dynamics/beyondAMP>:

| Commit | File | Bug |
|:---|:---|:---|
| `bc6922a` | `runners/amp_on_policy_runner.py:124` | Stray `[0]` subscript on `predict_amp_reward()` return value → "too many values to unpack". |
| `cee88cd` | `modules/amp_discriminator.py:69` | `d` / `amp_reward` returned shape `[N,1]`; runner consumed them as `[N]`, silently broadcasting to `[N,N]` during reward accumulation. Both now `.squeeze()`. |
| `cee88cd` | `runners/amp_on_policy_runner.py:177–197` | `log()` assumed every `ep_infos` dict carried the same keys; mjlab emits per-step dicts with varying keys (reward terms only appear on episode end). Now collects the union of keys and skips per-dict misses. |

If you sync `source/beyondAMP/` from upstream you'll get these for
free; if you pin an older revision you'll need to cherry-pick.

---

## 7. Tiptoe / post-kick stability fixes (kicking task)

Two known failure modes from the original IsaacLab G1_kicking made it
into the mjlab port as well, so they're addressed in `mdp/rewards.py`
from the start instead of as a post-hoc patch.

### 7.1 Right foot tiptoe during kick

Support foot drifts into plantarflexion as a locally optimal kick-power
posture. Mitigations active from Stage A:

| Term | Stage A | Stage B (B1 / B2) | Stage C |
|:---|:---:|:---:|:---:|
| `penalty_right_ankle_pitch_staged` (th1_deg = 6°) | **-3.0** | -3.0 / -3.0 | -5.0 |
| `reward_support_foot_parallel_rp` | **1.0** (new in A) | 1.0 | 1.0 |
| `reward_support_foot_flat_contact_early` | **-2.0** (new) | -3.0 / -3.5 | -4.5 |

`_ANTI_TOE_FILTER` from the original `kick_amp_data_cfg.py` carries over
unchanged.

### 7.2 Post-kick crouch / instability

After follow-through the robot tends to over-flex the knee and stay
crouched. Mitigations:

| Term | Stage A | Stage B1 | Stage B2 | Stage C |
|:---|:---:|:---:|:---:|:---:|
| `reward_post_kick_upright` | **0.3** | 0.8 | 0.8 | 1.2 |
| `reward_post_kick_upright_early` (new) | 0.3 | 0.8 | 0.8 | 1.5 |
| `penalty_post_kick_crouch` | **-0.5** | **-1.5** | **-3.0** | **-4.0** |

Stage-A weights stay small enough not to compete with kick learning
(ball-speed reward sits at 13.0).

---

## 8. Verification

### 8.1 Dribbling (T1) — smoke train

| Iters | Steps/s | Notes |
|:---:|:---:|:---|
| 50 | ~5755 (RTX 4090, num_envs = 64) | env steps cleanly, rewards non-zero |

### 8.2 Kicking (G1) — Bootstrap smoke train, 30 iters @ RTX 4090

- Steady-state throughput: **~2500 steps/s** (iter 0 was 1399 due to JIT warmup; 30 iters in 18.8 s total)
- Losses at iter 29, all non-NaN: value 0.063, surrogate -0.040, AMP 0.056, AMP-grad 0.068
- AMP discriminator separates expert from policy:
  mean policy logit -0.87 / mean expert logit +0.83
- Tracking rewards active (anchor_pos 0.0044, body_pos 0.0040, body_vel 0.0014); stability penalties active (`bad_ball_contact` -0.0318, `support_foot_flat_contact_early` -0.0198); kick rewards still ~0 at 30 iters — expected for Stage A0
- Only warnings observed: a benign `urllib3/chardet` system version mismatch and the one-time `cuBLAS … no current CUDA context` torch init message

### 8.3 Stages A / B / C

Not yet re-run after the upstream patches. The four factories share
`env_cfg.py` so they're expected to behave the same as Bootstrap modulo
curriculum weights. To validate before a long run, do a 20-iter smoke on
each.

---

## 9. Known gaps

- **Standup / recovery policy** is **not** in this repo. The GIF in the
  README is from an external standup policy. To stand up an end-to-end
  "fall → recover → kick" FSM demo we still need that checkpoint.
- **Only one kick motion clip** (`wo_cf_shoot_74_06.npz`). For varied
  kick directions / power, more clips would need to be added under
  `data/datasets/g1_kick_skill/`.
- Stages A / B / C of the kick curriculum have not been smoke-tested
  yet — only Bootstrap.
- No full-quality training run has been done. The `chii` remote-training
  workflow (autodl5090) is documented in chii's memory; when GPU is
  allocated we can deploy.

---

## 10. Quick reference

| Action | Command |
|:---|:---|
| List registered mjlab tasks | `python -c "from mjlab.tasks.registry import list_tasks; [print(t) for t in list_tasks()]"` |
| Train dribble (smoke) | `python scripts/factoryMjlab/train.py Soccer-Mjlab-Dribble-Flat-T1 --headless --num_envs 64 --agent.max-iterations 50` |
| Train kick bootstrap (smoke) | `python scripts/factoryMjlab/train.py Soccer-Mjlab-Kick-Bootstrap-G1 --headless --num_envs 64 --agent.max-iterations 30` |
| Reinstall after upstream beyondAMP sync | `cd source/beyondAMP && git pull && pip install -e source/beyondAMP source/amp_tasks_mjlab source/rsl_rl_amp` |
