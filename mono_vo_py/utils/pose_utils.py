import math
import numpy as np
from scipy.spatial.transform import Rotation


def read_file_list(filename):
    """Reads a trajectory from a text file (TUM format).

    File format: "stamp tx ty tz qx qy qz qw" per line.

    Returns:
        list of (stamp, [tx,ty,tz,qx,qy,qz,qw]) sorted by stamp
    """
    with open(filename) as f:
        data = f.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    parsed = [[v.strip() for v in line.split(" ") if v.strip() != ""]
              for line in lines if len(line) > 0 and line[0] != "#"]
    parsed = [(float(l[0]), l[1:]) for l in parsed if len(l) > 1]
    parsed.sort(key=lambda x: x[0])
    return parsed


def _to_mat(tx, ty, tz, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T


def umeyama_alignment(src, dst):
    """Sim3 alignment (Umeyama 1991): finds s, R, t such that dst ≈ s*R*src + t.
    
    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points
    Returns:
        s, R, t - scale, rotation (3x3), translation (3,)
    """
    assert src.shape == dst.shape
    n, d = src.shape

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / n

    H = dst_c.T @ src_c / n
    U, S_vals, Vt = np.linalg.svd(H)

    D = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[d - 1, d - 1] = -1

    R = U @ D @ Vt
    s = np.trace(np.diag(S_vals) @ D) / var_src
    t = mu_dst - s * R @ mu_src

    return s, R, t


def compute_ate(gt, pred, align=True):
    """ATE: absolute translation and rotation error.
    
    When align=True, performs Sim3 alignment (Umeyama) before computing error.
    This is essential for monocular VO where scale is unknown.
    
    Args:
        gt, pred: list of (frame_idx, tx, ty, tz, qx, qy, qz, qw)
        align: if True, do Sim3 alignment first
    Returns:
        (ate_trans, ate_rot) - RMSE in meters and degrees
    """
    gt_dict = {int(p[0]): p for p in gt}
    pr_dict = {int(p[0]): p for p in pred}
    common = sorted(set(gt_dict.keys()) & set(pr_dict.keys()))

    if len(common) < 1:
        return float('nan'), float('nan')

    gt_xyz = np.array([gt_dict[i][1:4] for i in common])
    pr_xyz = np.array([pr_dict[i][1:4] for i in common])

    if align and len(common) >= 3:
        s, R_align, t_align = umeyama_alignment(pr_xyz, gt_xyz)
        pr_xyz = (s * (R_align @ pr_xyz.T).T) + t_align

    trans_errors = np.linalg.norm(pr_xyz - gt_xyz, axis=1)

    rot_errors = []
    for i in common:
        R_gt = Rotation.from_quat(gt_dict[i][4:8]).as_matrix()
        R_pr = Rotation.from_quat(pr_dict[i][4:8]).as_matrix()
        R_err = R_gt.T @ R_pr
        cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors.append(math.degrees(math.acos(cos_angle)))

    ate_trans = np.sqrt(np.mean(trans_errors ** 2))
    ate_rot = np.sqrt(np.mean(np.array(rot_errors) ** 2))
    return ate_trans, ate_rot


def compute_rpe(gt_poses, pr_poses):
    """gt_poses, pr_poses: list of (frame_idx, tx, ty, tz, qx, qy, qz, qw)"""
    gt_dict = {p[0]: _to_mat(*p[1:]) for p in gt_poses}
    pr_dict = {p[0]: _to_mat(*p[1:]) for p in pr_poses}
    common = sorted(set(gt_dict.keys()) & set(pr_dict.keys()))

    if len(common) < 2:
        return False, False

    trans_errors = []
    rot_errors = []
    for i in range(len(common) - 1):
        fi, fj = common[i], common[i + 1]

        gt_rel = np.linalg.inv(gt_dict[fi]) @ gt_dict[fj]
        pr_rel = np.linalg.inv(pr_dict[fi]) @ pr_dict[fj]

        E = np.linalg.inv(gt_rel) @ pr_rel
        trans_errors.append(np.linalg.norm(E[:3, 3]))

        cos_angle = np.clip((np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        rot_errors.append(math.degrees(math.acos(cos_angle)))

    rpe_trans = math.sqrt(np.mean(np.array(trans_errors) ** 2))
    rpe_rot = math.sqrt(np.mean(np.array(rot_errors) ** 2))
    return rpe_trans, rpe_rot
