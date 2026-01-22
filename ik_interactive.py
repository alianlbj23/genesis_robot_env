import numpy as np
import genesis as gs
import sys
import select

# --- WSL viewer 常在文字 overlay OOM，先禁用（不影響 IK） ---
import genesis.ext.pyrender.renderer as _r

if hasattr(_r, "Renderer") and hasattr(_r.Renderer, "render_texts"):
    _r.Renderer.render_texts = lambda *args, **kwargs: None


# ---------------- init ----------------
def _init_genesis_quiet():
    try:
        gs.init(backend=gs.gpu, log_level="error")
        return
    except TypeError:
        pass
    try:
        gs.init(backend=gs.gpu, logging_level="error")
        return
    except TypeError:
        pass
    gs.init(backend=gs.gpu)
    for attr in ("logger", "logging", "log"):
        logger = getattr(gs, attr, None)
        if logger is None:
            continue
        for name in ("set_level", "setLevel", "set_log_level", "setLogLevel"):
            func = getattr(logger, name, None)
            if callable(func):
                try:
                    func("error")
                    return
                except Exception:
                    try:
                        import logging

                        func(logging.ERROR)
                        return
                    except Exception:
                        pass


_init_genesis_quiet()

# ---------------- scene ----------------
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, -1.5, 1.5),
        camera_lookat=(0.0, 0.0, 0.6),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,
)

scene.add_entity(gs.morphs.Plane())

# ---------------- load your URDF ----------------
robot = scene.add_entity(
    gs.morphs.URDF(
        file="assets/angle_excurate_arm_ver2_to_unity/angle_excurate_arm_ver2_clean.urdf",
        fixed=True,
        pos=(0, 0, 0),
        euler=(0, 0, 0),
        recompute_inertia=True,
        convexify=True,
        merge_fixed_links=False,
    )
)

scene.build()


def _viewer_closed(viewer) -> bool:
    if viewer is None:
        return True
    for attr in ("is_closed", "closed"):
        val = getattr(viewer, attr, None)
        if callable(val):
            try:
                if val():
                    return True
            except Exception:
                pass
        elif isinstance(val, bool) and val:
            return True
    for attr in ("is_open", "opened", "is_alive"):
        val = getattr(viewer, attr, None)
        if callable(val):
            try:
                if not val():
                    return True
            except Exception:
                pass
        elif isinstance(val, bool) and not val:
            return True
    return False


def _viewer_key_pressed(viewer, key: str) -> bool:
    if viewer is None:
        return False
    for func_name in (
        "is_key_pressed",
        "key_pressed",
        "get_key_pressed",
        "get_key_down",
    ):
        func = getattr(viewer, func_name, None)
        if callable(func):
            try:
                if func(key):
                    return True
            except TypeError:
                try:
                    if func(ord(key)):
                        return True
                except Exception:
                    pass
            except Exception:
                pass
    return False


def _read_stdin_commands():
    cmds = []
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if ready:
            line = sys.stdin.readline()
            cmds = list(line.strip())
    except Exception:
        pass
    return cmds


def _get_link_pos(link):
    for name in ("get_pose", "get_world_pose", "get_transform", "get_world_transform"):
        func = getattr(link, name, None)
        if callable(func):
            try:
                pose = func()
                if isinstance(pose, np.ndarray):
                    if pose.shape == (4, 4):
                        return np.array(pose[:3, 3], dtype=np.float32)
                    if pose.ndim == 1 and pose.shape[0] >= 3:
                        return np.array(pose[:3], dtype=np.float32)
                if isinstance(pose, (list, tuple)) and len(pose) >= 3:
                    return np.array(pose[:3], dtype=np.float32)
                if isinstance(pose, dict):
                    if "pos" in pose:
                        return np.array(pose["pos"], dtype=np.float32)
                    if "position" in pose:
                        return np.array(pose["position"], dtype=np.float32)
                    if "translation" in pose:
                        return np.array(pose["translation"], dtype=np.float32)
                if hasattr(pose, "pos"):
                    return np.array(pose.pos, dtype=np.float32)
            except Exception:
                pass
    for name in ("get_pos", "get_position", "get_world_pos", "get_world_position"):
        func = getattr(link, name, None)
        if callable(func):
            try:
                return np.array(func(), dtype=np.float32)
            except Exception:
                pass
    pos = getattr(link, "pos", None)
    if pos is not None:
        try:
            return np.array(pos, dtype=np.float32)
        except Exception:
            pass
    return None


def _set_marker_pos(marker, pos: np.ndarray) -> None:
    for name in ("set_pos", "set_position", "set_world_pos", "set_world_position"):
        func = getattr(marker, name, None)
        if callable(func):
            try:
                func(pos)
                return
            except Exception:
                pass


def _get_link_world_pos(robot, link):
    link_name = getattr(link, "name", None) or getattr(link, "link_name", None)
    for name in (
        "get_link_state",
        "get_link_states",
        "get_link_pose",
        "get_link_poses",
        "get_link_transform",
        "get_link_transforms",
        "get_link_world_pos",
        "get_link_world_position",
    ):
        func = getattr(robot, name, None)
        if callable(func):
            try:
                state = func(link_name) if link_name is not None else func(link)
                if isinstance(state, dict):
                    for key in (
                        "pos",
                        "position",
                        "world_pos",
                        "world_position",
                        "translation",
                    ):
                        if key in state:
                            return np.array(state[key], dtype=np.float32)
                if isinstance(state, np.ndarray) and state.shape == (4, 4):
                    return np.array(state[:3, 3], dtype=np.float32)
                if isinstance(state, (list, tuple, np.ndarray)) and len(state) >= 3:
                    return np.array(state[:3], dtype=np.float32)
            except Exception:
                pass
    return _get_link_pos(link)


# ---------------- control gains ----------------
n = robot.n_dofs
robot.set_dofs_kp(np.ones(n) * 1200.0)
robot.set_dofs_kv(np.ones(n) * 120.0)

# ---------------- IK + interactive target ----------------
ee = robot.get_link("robot_ver7_grap2_2_v1_1")  # ✅ 你的末端

# warm up a few steps so link world poses are updated
for _ in range(5):
    scene.step()

# target (red) and EE (green) markers
target_pos = np.array([0.20, -0.15, 0.15], dtype=np.float32)

target_marker = scene.draw_debug_sphere(
    pos=target_pos, radius=0.01, color=(1.0, 0.0, 0.0, 1.0)
)

ee_pos = _get_link_world_pos(robot, ee)
if ee_pos is not None:
    ee_marker = scene.draw_debug_sphere(
        pos=ee_pos, radius=0.008, color=(0.0, 1.0, 0.0, 1.0)
    )
else:
    ee_marker = None

print(
    "Controls: WASD move in XY, R/F move in Z. Close window to exit. "
    "(If viewer keys don't work, type keys in terminal then ENTER.)"
)

# 目標姿態：固定姿態
target_quat = np.array([0, 1, 0, 0], dtype=np.float32)

ik_max_iters = 1000
move_step = 0.002  # meters per frame

while True:
    viewer = getattr(scene, "viewer", None)
    if _viewer_closed(viewer):
        break

    moved = False
    if _viewer_key_pressed(viewer, "w") or _viewer_key_pressed(viewer, "W"):
        target_pos[1] += move_step
        moved = True
    if _viewer_key_pressed(viewer, "s") or _viewer_key_pressed(viewer, "S"):
        target_pos[1] -= move_step
        moved = True
    if _viewer_key_pressed(viewer, "a") or _viewer_key_pressed(viewer, "A"):
        target_pos[0] -= move_step
        moved = True
    if _viewer_key_pressed(viewer, "d") or _viewer_key_pressed(viewer, "D"):
        target_pos[0] += move_step
        moved = True
    if _viewer_key_pressed(viewer, "r") or _viewer_key_pressed(viewer, "R"):
        target_pos[2] += move_step
        moved = True
    if _viewer_key_pressed(viewer, "f") or _viewer_key_pressed(viewer, "F"):
        target_pos[2] -= move_step
        moved = True

    for key in _read_stdin_commands():
        if key in ("w", "W"):
            target_pos[1] += move_step
            moved = True
        elif key in ("s", "S"):
            target_pos[1] -= move_step
            moved = True
        elif key in ("a", "A"):
            target_pos[0] -= move_step
            moved = True
        elif key in ("d", "D"):
            target_pos[0] += move_step
            moved = True
        elif key in ("r", "R"):
            target_pos[2] += move_step
            moved = True
        elif key in ("f", "F"):
            target_pos[2] -= move_step
            moved = True

    if moved:
        _set_marker_pos(target_marker, target_pos)

    _ik_kwargs = {
        "link": ee,
        "pos": target_pos,
        "quat": target_quat,
    }
    try:
        qpos_goal = robot.inverse_kinematics(**_ik_kwargs, max_iters=ik_max_iters)
    except TypeError:
        try:
            qpos_goal = robot.inverse_kinematics(**_ik_kwargs, max_iter=ik_max_iters)
        except TypeError:
            try:
                qpos_goal = robot.inverse_kinematics(
                    **_ik_kwargs, num_iters=ik_max_iters
                )
            except TypeError:
                qpos_goal = robot.inverse_kinematics(**_ik_kwargs)

    robot.control_dofs_position(qpos_goal)
    scene.step()

    if ee_marker is not None:
        ee_pos = _get_link_world_pos(robot, ee)
        if ee_pos is not None:
            _set_marker_pos(ee_marker, ee_pos)
