"""CLI extraction of notebook 05 release-v3; one isolated process per episode."""
from pathlib import Path
import os
import time
rollout_wall_start = time.monotonic()
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--demo", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

from pathlib import Path
import json
import hashlib
import importlib.metadata
from datetime import datetime, timezone
from itertools import product
import numpy as np
import pandas as pd
import h5py
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation
from IPython.display import Video, display
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

project = args.project.resolve()
pilot_dir = project / "data/processed" / args.demo
assert project.is_dir()
trajectory_file = pilot_dir / "marker_smoothed.csv"
events_file = pilot_dir / "human_events.json"
trajectory = pd.read_csv(trajectory_file)
events = json.loads(events_file.read_text())["events"]

TASK = "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
ROLLOUT_REVISION = "release-v3"
print("Loaded rollout revision:", ROLLOUT_REVISION)

CONFIG = dict(seed=0, fps=20, max_steps=12000, stage_steps=1000,
              translation_limit=0.02, rotation_limit=0.05,
              insertion_depth_m=0.005, hover_clearance_m=0.020,
              waypoint_spacing_m=0.005, drift_limit_m=0.040, recent_drift_limit_m=0.010,
              contact_loss_limit_steps=10)
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task_id = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == TASK)
task = suite.get_task(task_id)
bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

# Validate inputs before creating a simulation.
sample_times = trajectory["seconds"].to_numpy()
assert np.all(np.diff(sample_times) > 0)
t0 = events["grasp_established"]["time_s"]
t1 = events["placement_contact"]["time_s"]
assert sample_times[0] <= t0 < t1 <= sample_times[-1]
closure_seconds = t0 - events["closure_start"]["time_s"]
assert 0 < closure_seconds < 5
human_times = np.r_[t0, sample_times[(sample_times > t0) & (sample_times < t1)], t1]
human_positions = np.column_stack([
    np.interp(human_times, sample_times, trajectory[k])
    for k in ["x_m", "y_m", "height_m"]
])
assert np.isfinite(human_positions).all()


def collision_box_bounds(body_name):
    body_id = sim.model.body_name2id(body_name)

    first = int(sim.model.body_geomadr[body_id])
    count = int(sim.model.body_geomnum[body_id])

    # Eight combinations: (-1,-1,-1), (-1,-1,+1), ...
    corner_signs = np.array(
        list(product([-1, 1], repeat=3)),
        dtype=float,
    )

    world_corners = []

    for geom_id in range(first, first + count):
        # In MuJoCo, geometry type 6 is a box.
        if int(sim.model.geom_type[geom_id]) != 6:
            continue

        # Exclude geometries that have no collision masks.
        if (sim.model.geom_contype[geom_id] == 0 and sim.model.geom_conaffinity[geom_id] == 0):
            continue

        half_sizes = sim.model.geom_size[geom_id]
        position = sim.data.geom_xpos[geom_id]
        rotation = sim.data.geom_xmat[geom_id].reshape(3, 3)

        local_corners = corner_signs * half_sizes
        corners = local_corners @ rotation.T + position

        world_corners.append(corners)

    assert world_corners, f"No collision boxes found for {body_name}"

    points = np.concatenate(world_corners, axis=0)

    return {
        "min": points.min(axis=0),
        "max": points.max(axis=0),
    }



def pose():
    return (sim.data.site_xpos[site_id].copy(),
            sim.data.site_xmat[site_id].reshape(3, 3).copy())


def errors(target, orientation):
    p, r = pose()
    return target - p, Rotation.from_matrix(orientation @ r.T).as_rotvec()


def command(target, orientation, grip):
    dp, dr = errors(target, orientation)
    return np.r_[np.clip(dp / .05, -CONFIG["translation_limit"], CONFIG["translation_limit"]),
                 np.clip(dr / .5, -CONFIG["rotation_limit"], CONFIG["rotation_limit"]), grip]


def geom_ids(prefix):
    return {i for i in range(sim.model.ngeom)
            if (sim.model.geom_id2name(i) or "").startswith(prefix)}


def touching(a, b):
    for i in range(sim.data.ncon):
        c = sim.data.contact[i]
        if c.dist <= 0.0001 and ((c.geom1 in a and c.geom2 in b) or
                                (c.geom2 in a and c.geom1 in b)):
            return True
    return False


def map_transfer(human, times, start, destination):
    dh = human[-1, :2] - human[0, :2]
    dr = destination[:2] - start[:2]
    assert np.linalg.norm(dh) > 1e-6
    angle = np.arctan2(dr[1], dr[0]) - np.arctan2(dh[1], dh[0])
    r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    scale = np.linalg.norm(dr) / np.linalg.norm(dh)
    u = ((times - times[0]) / (times[-1] - times[0]))[:, None]
    residual = human - ((1-u)*human[0] + u*human[-1])
    residual[:, :2] = residual[:, :2] @ r.T * scale
    path = (1-u)*start + u*destination + residual
    assert np.allclose(path[0], start) and np.allclose(path[-1], destination)
    return path, float(scale)


class Recorder:
    def __init__(self, folder):
        self.h5 = h5py.File(folder / "episode.h5", "w")
        self.video = imageio.get_writer(str(folder / "replay.mp4"), fps=10)
        self.n = 0
        self.h5.attrs["fps"] = CONFIG["fps"]
        self.h5.attrs["language"] = task.language
        self.h5.attrs["image_convention"] = "upright RGB HWC; raw simulator images vertically flipped"
        self.h5.attrs["state_layout"] = "eef_pos_xyz, eef_quat_xyzw, gripper_qpos_2"
        self.h5.attrs["alignment"] = "observation[t], action[t], success_after[t]"
        self.h5.attrs["committed_steps"] = 0
        self.h5.create_dataset("initial_sim_state", data=env.get_sim_state())
        self.h5.create_dataset("model_xml", data=sim.model.get_xml(), dtype=h5py.string_dtype())

    def append(self, values):
        for key, value in values.items():
            a = np.asarray(value)
            if key not in self.h5:
                kwargs = dict(shape=(0,) + a.shape, maxshape=(None,) + a.shape,
                              chunks=(1,) + a.shape, dtype=a.dtype)
                if a.ndim == 3:
                    kwargs.update(compression="lzf")
                if a.dtype.kind == "U":
                    kwargs["dtype"] = h5py.string_dtype()
                self.h5.create_dataset(key, **kwargs)
            ds = self.h5[key]
            ds.resize(self.n + 1, axis=0)
            ds[self.n] = value
        if self.n % 2 == 0:
            self.video.append_data(np.concatenate([values["agentview_image"], values["wrist_image"]], axis=1))
        self.n += 1
        if self.n % 100 == 0:
            print(f"Recorded {self.n} steps | phase={values['phase']} | elapsed={time.monotonic()-rollout_wall_start:.1f}s", flush=True)
        self.h5.attrs["committed_steps"] = self.n
        if self.n % 20 == 0:
            self.h5.flush()

    def close(self):
        try:
            self.h5.flush()
            self.h5.close()
        finally:
            self.video.close()


def step(target, orientation, grip, phase):
    global ob
    if recorder.n >= CONFIG["max_steps"]:
        raise RuntimeError("Episode step budget exhausted")
    low, high = task_env.action_spec
    action = np.clip(command(target, orientation, grip), low, high)
    values = {
        "agentview_image": ob["agentview_image"][::-1].copy(),
        "wrist_image": ob["robot0_eye_in_hand_image"][::-1].copy(),
        "state": np.r_[ob["robot0_eef_pos"], ob["robot0_eef_quat"], ob["robot0_gripper_qpos"]].astype(np.float32),
        "sim_state": np.asarray(env.get_sim_state()).copy(),
        "action": action.astype(np.float32),
        "phase": phase,
        "timestamp_s": recorder.n / CONFIG["fps"],
        "target_position": target.copy(),
    }
    ob, reward, done, info = env.step(action)
    dp, dr = errors(target, orientation)
    values.update(success_after=bool(env.check_success()), done_after=bool(done),
                  reward_after=float(reward), position_error_after=float(np.linalg.norm(dp)))
    recorder.append(values)

    # LIBERO's returned done flag can mean task success, not simulator termination.
    # Preserve it in done_after, but finish release/withdrawal when stepping is valid.
    success = bool(values["success_after"])
    terminal_latch = bool(getattr(task_env, "done", False))
    if terminal_latch or (done and not success):
        raise RuntimeError(
            f"Environment terminated: returned_done={bool(done)}, "
            f"success={success}, terminal_latch={terminal_latch}"
        )
    if success and "first_success_step" not in report:
        report["first_success_step"] = recorder.n
        report["first_success_phase"] = phase
        print("Task predicate reached; continuing through release and withdrawal.", flush=True)
    return np.linalg.norm(dp) < .003 and np.linalg.norm(dr) < np.deg2rad(3)


def move(target, orientation, grip, phase, limit=None, guard=None):
    print(phase, flush=True)
    stable = 0
    for _ in range(limit or CONFIG["stage_steps"]):
        reached = step(target, orientation, grip, phase)
        if guard:
            guard()
        stable = stable + 1 if reached else 0
        if stable >= 5:
            return
    raise RuntimeError(f"{phase}: tracking timed out")


def hold(target, orientation, grip, phase, steps):
    for _ in range(steps):
        step(target, orientation, grip, phase)


assert ROLLOUT_REVISION == "release-v3"
assert CONFIG["drift_limit_m"] == 0.040
assert CONFIG["recent_drift_limit_m"] == 0.010
print("Starting rollout revision:", ROLLOUT_REVISION, flush=True)

# A rerun never continues a partly executed episode.
if "env" in globals():
    env.close()
print("Creating environment/rendering context...", flush=True)
env = OffScreenRenderEnv(bddl_file_name=bddl_path,
                         camera_heights=256,
                         camera_widths=256,
                         horizon=CONFIG["max_steps"],
                         ignore_done=True)
env.seed(CONFIG["seed"])
print("Resetting environment...", flush=True)
ob = env.reset()
print("Environment reset complete.", flush=True)
task_env = env.env
sim = task_env.sim
controller = task_env.robots[0].controller
gripper = task_env.robots[0].gripper
site_id = sim.model.site_name2id(controller.eef_name)
assert controller.use_delta
assert np.allclose(controller.input_max, 1) and np.allclose(controller.input_min, -1)
assert np.allclose(controller.output_max, [.05]*3 + [.5]*3)
assert np.allclose(controller.output_min, [-.05]*3 + [-.5]*3)
assert task_env.control_freq == CONFIG["fps"]

run_dir = args.output
run_dir.mkdir(parents=True, exist_ok=False)
report = dict(rollout_revision=ROLLOUT_REVISION, status="running", training_eligible=False, visual_review_required=True,
              task=TASK, config=CONFIG, source_events=events,
              source_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [trajectory_file, events_file]},
              provenance="Human transfer shape; engineered grasp, orientation, placement and timing adaptation",
              versions={})
for package in ["lerobot", "libero", "robosuite", "mujoco", "numpy", "h5py"]:
    try:
        report["versions"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        report["versions"][package] = "unavailable"
(run_dir / "report.json").write_text(json.dumps(report, indent=2))
print("Starting local recording:", run_dir, flush=True)
recorder = Recorder(run_dir)

try:
    # Settle the scene while opening, then measure geometry in that state.
    p, r = pose()
    hold(p, r, -1, "settle_open", 30)
    bowl_id = sim.model.body_name2id("akita_black_bowl_1_main")
    bowl_geoms = geom_ids("akita_black_bowl_1_")
    plate_geoms = geom_ids("plate_1_")
    finger_geoms = {sim.model.geom_name2id(name)
                    for names in gripper.important_geoms.values() for name in names}
    b = collision_box_bounds("akita_black_bowl_1_main")
    p, r = pose()
    centre = (b["min"][:2] + b["max"][:2]) / 2
    radial = p[:2] - centre
    radial /= np.linalg.norm(radial)
    x = np.r_[radial, 0.]
    z = np.array([0., 0., -1.])
    orientation = np.column_stack([x, np.cross(z, x), z])
    pad_centres = [sim.data.geom_xpos[sim.model.geom_name2id(gripper.important_geoms[f"{side}_fingerpad"][0])].copy()
                   for side in ["left", "right"]]
    pad_offset = r.T @ (np.mean(pad_centres, axis=0) - p)
    radius = np.mean(b["max"][:2] - b["min"][:2]) / 2 - .006
    grasp_pad = np.r_[centre + radius*radial, b["max"][2] - CONFIG["insertion_depth_m"]]
    grasp_target = grasp_pad - orientation @ pad_offset
    high = grasp_target.copy()
    high[2] = max(p[2], b["max"][2] + .10)
    move(p, orientation, -1, "orient")
    vertical = p.copy(); vertical[2] = high[2]
    move(vertical, orientation, -1, "raise_for_approach")
    move(high, orientation, -1, "above_rim")
    approach = grasp_target.copy(); approach[2] += .075
    move(approach, orientation, -1, "approach")
    move(grasp_target, orientation, -1, "insert")
    p, _ = pose()
    hold(p, orientation, 1, "close", max(20, int(np.ceil(closure_seconds*20))))

    if not touching(finger_geoms, bowl_geoms):
        raise RuntimeError("No finger–bowl contact after closure")

    start, _ = pose()
    b = collision_box_bounds("akita_black_bowl_1_main")
    plate = collision_box_bounds("plate_1_main")
    destination = start.copy()
    destination[:2] += (plate["min"][:2]+plate["max"][:2]-b["min"][:2]-b["max"][:2])/2
    destination[2] += plate["max"][2] + CONFIG["hover_clearance_m"] - b["min"][2]
    path, scale = map_transfer(human_positions, human_times, start, destination)
    np.savez_compressed(run_dir / "mapped_path.npz", path=path, human_times=human_times,
                        human_positions=human_positions, orientation=orientation)
    report["horizontal_scale"] = scale

    from collections import deque
    left_geoms = {
        sim.model.geom_name2id(name)
        for name in gripper.important_geoms["left_finger"]
    }
    right_geoms = {
        sim.model.geom_name2id(name)
        for name in gripper.important_geoms["right_finger"]
    }

    def relative_bowl_position():
        wrist_position, wrist_rotation = pose()
        return wrist_rotation.T @ (
            sim.data.body_xpos[bowl_id] - wrist_position
        )

    initial_relative_position = relative_bowl_position().copy()
    # 21 samples span one second at the asserted 20 Hz control rate.
    relative_history = deque([initial_relative_position.copy()], maxlen=21)
    contact_loss_steps = 0

    def grasp_guard():
        global contact_loss_steps
        relative_position = relative_bowl_position()
        relative_history.append(relative_position.copy())
        both_fingers_touching = (
            touching(left_geoms, bowl_geoms)
            and touching(right_geoms, bowl_geoms)
        )
        contact_loss_steps = 0 if both_fingers_touching else contact_loss_steps + 1
        total_shift = np.linalg.norm(relative_position - initial_relative_position)
        recent_shift = (
            np.linalg.norm(relative_position - relative_history[0])
            if len(relative_history) == relative_history.maxlen else 0.0
        )
        report["grasp_monitor"] = {
            "total_shift_mm": float(total_shift * 1000),
            "recent_shift_mm": float(recent_shift * 1000),
            "contact_loss_steps": contact_loss_steps,
        }
        if contact_loss_steps >= CONFIG["contact_loss_limit_steps"]:
            raise RuntimeError("Persistent loss of two-finger contact")
        if recent_shift > CONFIG["recent_drift_limit_m"]:
            raise RuntimeError("Rapid bowl movement relative to wrist")
        if total_shift > CONFIG["drift_limit_m"]:
            raise RuntimeError("Bowl shifted over total displacement limit")

    waypoints = [path[0]]
    for point in path[1:]:
        if np.linalg.norm(point-waypoints[-1]) >= CONFIG["waypoint_spacing_m"]:
            waypoints.append(point)
    if not np.allclose(waypoints[-1], path[-1]):
        waypoints.append(path[-1])
    for i, target in enumerate(waypoints[1:], 1):
        move(target, orientation, 1, f"human_transfer_{i}", limit=400, guard=grasp_guard)
    report["transfer_endpoint_error_m"] = float(np.linalg.norm(pose()[0]-path[-1]))

    # The grasp may have settled or tilted: use current held-bowl geometry.
    bowl_now = collision_box_bounds("akita_black_bowl_1_main")
    plate_now = collision_box_bounds("plate_1_main")
    corrected_hover, _ = pose()
    bowl_center = (bowl_now["min"][:2] + bowl_now["max"][:2]) / 2
    plate_center = (plate_now["min"][:2] + plate_now["max"][:2]) / 2
    corrected_hover[:2] += plate_center - bowl_center
    corrected_hover[2] += (
        plate_now["max"][2] + CONFIG["hover_clearance_m"] - bowl_now["min"][2]
    )
    raise_target, _ = pose()
    raise_target[2] = max(raise_target[2], corrected_hover[2])
    move(raise_target, orientation, 1, "restore_hover_clearance", guard=grasp_guard)
    move(corrected_hover, orientation, 1, "align_held_bowl_over_plate", guard=grasp_guard)

    # Contact-based placement: no assumption that plate AABB top is its support plane.
    p, _ = pose()
    bottom_limit = p[2] - .050
    descent_target = p.copy()
    contact_steps = 0

    for _ in range(1000):
        if touching(bowl_geoms, plate_geoms):
            contact_steps += 1
            descent_target = pose()[0]
        else:
            contact_steps = 0
            if np.linalg.norm(pose()[0]-descent_target) < .002:
                descent_target[2] -= .001

        if descent_target[2] < bottom_limit:
            raise RuntimeError("No stable bowl–plate contact within placement descent limit")

        step(descent_target, orientation, 1, "place_until_contact")
        if contact_steps >= 10:
            break
    else:
        raise RuntimeError("Placement contact timed out")

    p, _ = pose()
    print("release", flush=True)

    hold(p, orientation, -1, "release", 30)
    move(p + [0, 0, .080], orientation, -1, "withdraw")

    p, _ = pose()
    successes = []

    for _ in range(40):
        step(p, orientation, -1, "settle_after_release")
        successes.append(bool(env.check_success()))

    report.update(final_task_success=successes[-1], success_last_second=all(successes[-20:]),
                  finger_bowl_contact=touching(finger_geoms, bowl_geoms),
                  bowl_plate_contact=touching(bowl_geoms, plate_geoms),
                  release_sequence_completed=True)

    passed = (report["success_last_second"] and report["bowl_plate_contact"]
              and not report["finger_bowl_contact"])

    report["status"] = "passed_automatic_checks" if passed else "failed_final_checks"
    # Visual review is a separate, explicit dataset admission step.
    report["training_eligible"] = False
    recorder.h5.attrs["automatic_checks_passed"] = passed

except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    print(report["error"])

finally:
    report["steps"] = recorder.n
    report["simulated_duration_s"] = recorder.n / CONFIG["fps"]
    recorder.close()
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    print("Saved:", run_dir)
    print("Status:", report["status"])

    env.close()

if report["status"] != "passed_automatic_checks":
    raise SystemExit(1)
