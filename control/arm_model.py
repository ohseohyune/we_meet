"""arm_model.py - MuJoCo arm/model utilities for the inspection robot scene.

Joint lookups, site poses, and collision queries for the active 6-DOF DH arm.
"""

from __future__ import annotations

import numpy as np


ARM_JOINT_NAMES = [f"q{i}" for i in range(1, 7)]
ROBOT_COLLISION_BIT = 2


def joint_qpos_indices(model, mujoco_module):
    "각 관절들의 qpos 인덱스(주로)를 배열로 반환"
    "id : 1~6번 관절 번호"
    "qpos Index : 각 관절의 qpos 인덱스"
    ids = [mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    if any(i < 0 for i in ids):
        missing = [n for n, i in zip(ARM_JOINT_NAMES, ids) if i < 0]
        raise ValueError(f"Missing arm joints in model: {missing}")
    return np.array([model.jnt_qposadr[i] for i in ids], dtype=int)


def joint_limits(model, mujoco_module):
    "한계가 정해져 있으면 그 범위를 가져오고, 한계가 없는 무한 회전 관절이면 임의로 [−π,π]로 범위 제한 "
    ids = [mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
    limits = []
    for joint_id in ids:
        if model.jnt_limited[joint_id]:
            limits.append(model.jnt_range[joint_id])
        else:
            limits.append([-np.pi, np.pi])
    return np.array(limits, dtype=float)


def set_arm_qpos(model, data, mujoco_module, q):
    "로봇 초기 자세 설정"
    idx = joint_qpos_indices(model, mujoco_module)
    data.qpos[idx] = q
    mujoco_module.mj_forward(model, data) # 전체 동기화


def get_arm_qpos(model, data, mujoco_module):
    "로봇 관절값 가져오기"
    idx = joint_qpos_indices(model, mujoco_module)
    return data.qpos[idx].copy()


def site_pose(model, data, mujoco_module, site_name="ee_site"):
    "ee_site의 pose를 가져오기"
    sid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise ValueError(f"Site '{site_name}' not found")
    T = np.eye(4)
    T[:3, 3] = data.site_xpos[sid]
    T[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    return T


def _object_name(model, mujoco_module, obj_type, obj_id):
    "이름표 달아주기"
    name = mujoco_module.mj_id2name(model, obj_type, int(obj_id))
    obj_label = getattr(obj_type, "name", str(obj_type)).lower()
    return name if name is not None else f"{obj_label}_{int(obj_id)}"


def _is_robot_collision_geom(model, geom_id):
    "로봇의 contype를 검사해서 로봇이면 true, 아니면 false를 반환"
    return bool(int(model.geom_contype[int(geom_id)]) & ROBOT_COLLISION_BIT)


def collision_contacts_for_q(
    model,
    data,
    mujoco_module,
    q,
    qidx=None,
    collision_margin=0.0,
    max_pairs=12,
):
    """Return robot-involved contacts for an arm configuration."""
    if qidx is None:
        qidx = joint_qpos_indices(model, mujoco_module)
    data.qpos[qidx] = np.asarray(q, dtype=float)
    mujoco_module.mj_forward(model, data)

    contacts = []
    min_dist = np.inf
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        
        # 둘 중 하나라도 로봇이 아닌경우 
        if not (_is_robot_collision_geom(model, geom1) or _is_robot_collision_geom(model, geom2)):
            continue
        dist = float(contact.dist)
        min_dist = min(min_dist, dist)
        if dist > float(collision_margin):
            continue
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        contacts.append(
            {
                "dist": dist,
                "geom1": _object_name(model, mujoco_module, mujoco_module.mjtObj.mjOBJ_GEOM, geom1),
                "geom2": _object_name(model, mujoco_module, mujoco_module.mjtObj.mjOBJ_GEOM, geom2),
                "body1": _object_name(model, mujoco_module, mujoco_module.mjtObj.mjOBJ_BODY, body1),
                "body2": _object_name(model, mujoco_module, mujoco_module.mjtObj.mjOBJ_BODY, body2),
            }
        )

    contacts = sorted(contacts, key=lambda item: item["dist"])
    return {
        "count": len(contacts),
        "min_dist": min_dist if np.isfinite(min_dist) else np.inf,
        "contacts": contacts[:max(0, int(max_pairs))],
    }


def evaluate_collision_trajectory(
        
    model,
    data,
    mujoco_module,
    Q,
    collision_margin=0.0,
    max_pairs=4,
):
    "위 함수를 궤적 전체로 확장한 버전, Q = [q1, q2, q3, ...]. collision_free 배열이 전부 True인 궤적만 골라내어 실제 로봇에게 명령 "
    qidx = joint_qpos_indices(model, mujoco_module)
    counts = []
    min_dists = []
    pairs = []
    for q in np.asarray(Q, dtype=float):
        summary = collision_contacts_for_q(
            model,
            data,
            mujoco_module,
            q,
            qidx=qidx,
            collision_margin=collision_margin,
            max_pairs=max_pairs,
        )
        counts.append(int(summary["count"]))
        min_dists.append(float(summary["min_dist"]))
        pairs.append(summary["contacts"])
    return {
        "collision_count": np.asarray(counts, dtype=int),
        "min_contact_dist": np.asarray(min_dists, dtype=float),
        "contacts": pairs,
        "collision_free": np.asarray(counts, dtype=int) == 0,
    }
