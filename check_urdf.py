import genesis as gs

gs.init(backend=gs.gpu)  # 你有 4050，GPU OK

scene = gs.Scene(show_viewer=True)
scene.add_entity(gs.morphs.Plane())

robot = scene.add_entity(
    gs.morphs.URDF(
        file="assets/angle_excurate_arm_ver2_to_unity/angle_excurate_arm_ver2_clean.urdf",  # 相對路徑可以；Genesis 也支援絕對路徑
        fixed=True,  # 讓 base link 固定在世界座標（不然會是 free joint）
        pos=(0, 0, 0),
        euler=(0, 0, 0),
        # 若你的 mesh 座標系不是 Z-up，可調整：
        # file_meshes_are_zup=True/False,
        # 若你 URDF 沒寫 inertia 或寫得怪，可以先讓它重算：
        # recompute_inertia=True,
        # 若碰撞很複雜常出問題，可先試 convexify：
        # convexify=True,
        # 若你有 IK 需求，通常保留預設 requires_jac_and_IK=True 就好
    )
)

scene.build()
while True:
    scene.step()
