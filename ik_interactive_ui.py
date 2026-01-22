import sys
import tkinter as tk

import numpy as np
import genesis as gs

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


def _set_marker_pos(marker, pos: np.ndarray) -> bool:
    for name in ("set_pos", "set_position", "set_world_pos", "set_world_position"):
        func = getattr(marker, name, None)
        if callable(func):
            try:
                func(pos)
                return True
            except Exception:
                pass
    if marker is not None and hasattr(marker, "pos"):
        try:
            marker.pos = pos
            return True
        except Exception:
            pass
    return False


def _replace_marker(scene, marker, pos: np.ndarray, radius: float, color):
    for name in ("remove", "destroy", "delete", "detach"):
        func = getattr(marker, name, None)
        if callable(func):
            try:
                func()
                break
            except Exception:
                pass
    try:
        if marker is not None and hasattr(scene, "remove_entity"):
            scene.remove_entity(marker)
    except Exception:
        pass
    return scene.draw_debug_sphere(pos=pos, radius=radius, color=color)


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

# shared target state
_target_pos = np.array([0.20, -0.15, 0.15], dtype=np.float32)
_target_quat = np.array([0, 1, 0, 0], dtype=np.float32)

# target (red) and EE (green) markers
_target_marker = scene.draw_debug_sphere(
    pos=_target_pos, radius=0.01, color=(1.0, 0.0, 0.0, 1.0)
)

_ee_pos = _get_link_world_pos(robot, ee)
if _ee_pos is not None:
    _ee_marker = scene.draw_debug_sphere(
        pos=_ee_pos, radius=0.008, color=(0.0, 1.0, 0.0, 1.0)
    )
else:
    _ee_marker = None

ik_max_iters = 200
_last_target_pos = _target_pos.copy()
_last_qpos_goal = None


# ---------------- UI ----------------
root = tk.Tk()
root.title("IK Target Controls")

x_var = tk.DoubleVar(value=float(_target_pos[0]))
y_var = tk.DoubleVar(value=float(_target_pos[1]))
z_var = tk.DoubleVar(value=float(_target_pos[2]))


def _add_slider(label: str, var: tk.DoubleVar, frm: tk.Frame, lo: float, hi: float):
    tk.Label(frm, text=label, width=2).pack(side=tk.LEFT)
    tk.Scale(
        frm,
        variable=var,
        from_=lo,
        to=hi,
        resolution=0.001,
        orient=tk.HORIZONTAL,
        length=300,
    ).pack(side=tk.LEFT, padx=6)
    tk.Label(frm, textvariable=var, width=8).pack(side=tk.LEFT)


row1 = tk.Frame(root)
row1.pack(padx=8, pady=6)
_add_slider("X", x_var, row1, -0.2, 0.6)

row2 = tk.Frame(root)
row2.pack(padx=8, pady=6)
_add_slider("Y", y_var, row2, -0.3, 0.3)

row3 = tk.Frame(root)
row3.pack(padx=8, pady=6)
_add_slider("Z", z_var, row3, 0.0, 0.6)


def _on_close():
    root.destroy()


root.protocol("WM_DELETE_WINDOW", _on_close)

try:
    while True:
        viewer = getattr(scene, "viewer", None)
        if _viewer_closed(viewer):
            break
        if not root.winfo_exists():
            break

        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            break

        _target_pos[0] = float(x_var.get())
        _target_pos[1] = float(y_var.get())
        _target_pos[2] = float(z_var.get())

        if not _set_marker_pos(_target_marker, _target_pos):
            _target_marker = _replace_marker(
                scene, _target_marker, _target_pos, 0.01, (1.0, 0.0, 0.0, 1.0)
            )

        if np.linalg.norm(_target_pos - _last_target_pos) > 1e-6:
            _ik_kwargs = {
                "link": ee,
                "pos": _target_pos,
                "quat": _target_quat,
            }
            try:
                _last_qpos_goal = robot.inverse_kinematics(
                    **_ik_kwargs, max_iters=ik_max_iters
                )
            except TypeError:
                try:
                    _last_qpos_goal = robot.inverse_kinematics(
                        **_ik_kwargs, max_iter=ik_max_iters
                    )
                except TypeError:
                    try:
                        _last_qpos_goal = robot.inverse_kinematics(
                            **_ik_kwargs, num_iters=ik_max_iters
                        )
                    except TypeError:
                        _last_qpos_goal = robot.inverse_kinematics(**_ik_kwargs)
            _last_target_pos[:] = _target_pos

        if _last_qpos_goal is not None:
            robot.control_dofs_position(_last_qpos_goal)
        scene.step()

        if _ee_marker is not None:
            ee_pos = _get_link_world_pos(robot, ee)
            if ee_pos is not None:
                if not _set_marker_pos(_ee_marker, ee_pos):
                    _ee_marker = _replace_marker(
                        scene, _ee_marker, ee_pos, 0.008, (0.0, 1.0, 0.0, 1.0)
                    )
finally:
    try:
        if root.winfo_exists():
            root.destroy()
    except Exception:
        pass

if _viewer_closed(getattr(scene, "viewer", None)):
    sys.exit(0)
