import numpy as np
import genesis as gs

# --- WSL viewer 常在文字 overlay OOM，先禁用（不影響 IK） ---
import genesis.ext.pyrender.renderer as _r

if hasattr(_r, "Renderer") and hasattr(_r.Renderer, "render_texts"):
    _r.Renderer.render_texts = lambda *args, **kwargs: None

# ---------------- init ----------------
gs.init(backend=gs.gpu)

# ---------------- scene ----------------
scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=0.01),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.5, -1.5, 1.5),
        camera_lookat=(0.0, 0.0, 0.6),
        camera_fov=30,
        max_FPS=60,
    ),
    show_viewer=True,  # 如果 viewer 還是炸：改 False
)

scene.add_entity(gs.morphs.Plane())

# ---------------- load your URDF ----------------
robot = scene.add_entity(
    gs.morphs.URDF(
        file="assets/angle_excurate_arm_ver2_to_unity/angle_excurate_arm_ver2_clean.urdf",  # ✅ 推薦：noinertial + 重算
        fixed=True,
        pos=(0, 0, 0),
        euler=(0, 0, 0),
        # ✅ 避免 inertial_quat shape 問題
        recompute_inertia=True,
        # ✅ 穩定性
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


# ---------------- (optional) control gains ----------------
# 參考官方教學：不同機器人要自己 tune kp/kv。:contentReference[oaicite:1]{index=1}
# 先給一組保守值，讓它能動起來；之後你再依實際穩定度調大/調小。
n = robot.n_dofs
robot.set_dofs_kp(np.ones(n) * 1200.0)
robot.set_dofs_kv(np.ones(n) * 120.0)

# ---------------- IK + motion planning ----------------
ee = robot.get_link("robot_ver7_link_4_v1_1")  # ✅ 你的末端

# 隨機取一個「大概在手臂可達範圍」的目標點
# (如果解不出來，你就把範圍縮小一點)
target_pos = np.array(
    [
        np.random.uniform(0.20, 0.45),  # x
        np.random.uniform(-0.15, 0.15),  # y
        np.random.uniform(0.15, 0.40),  # z
    ],
    dtype=np.float32,
)

# 目標姿態：直接先用教學裡的 quat (0,1,0,0) 當固定姿態示例 :contentReference[oaicite:2]{index=2}
target_quat = np.array([0, 1, 0, 0], dtype=np.float32)

print("target_pos:", target_pos)

# draw a small red marker at the target position
target_marker = scene.draw_debug_sphere(
    pos=target_pos, radius=0.01, color=(1.0, 0.0, 0.0, 1.0)
)

qpos_goal = robot.inverse_kinematics(
    link=ee,
    pos=target_pos,
    quat=target_quat,
)

# 用 motion planner 走到目標（教學用 plan_path + waypoints 執行）:contentReference[oaicite:3]{index=3}
path = robot.plan_path(
    qpos_goal=qpos_goal,
    num_waypoints=200,
)

for waypoint in path:
    robot.control_dofs_position(waypoint)
    scene.step()

# 走完路徑後，固定目標關節並讓 PD controller 收斂
for _ in range(300):
    robot.control_dofs_position(qpos_goal)
    scene.step()

while True:
    viewer = getattr(scene, "viewer", None)
    if _viewer_closed(viewer):
        break
    scene.step()
