#!/usr/bin/env python3
"""Evaluate ATE and RPE before/after BA using existing code from ../mono_vo_py"""

import sys
import os
from pathlib import Path
from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt

# Add mono_vo_py to path
mono_vo_py_path = Path(__file__).parent.parent / 'mono_vo_py'
if not mono_vo_py_path.exists():
    mono_vo_py_path = Path('/home/jeffrey/ws/learn_vslam_code/mono_vo_py')
sys.path.insert(0, str(mono_vo_py_path))

# Import from mono_vo_py
sys.path.insert(0, str(mono_vo_py_path))
from utils.pose_utils import compute_ate, compute_rpe, umeyama_alignment
from mono_vo import Visualizer, CoordTransform
from data_loaders.kitti_loader import KITTILoader
import numpy as np
from scipy.spatial.transform import Rotation


def load_trajectory(filepath):
    """Load trajectory: idx x y z qx qy qz qw"""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                t = [float(parts[1]), float(parts[2]), float(parts[3])]
                q = [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]
                poses.append([0.0] + t + q)  # Add dummy timestamp
    return poses


def save_tum_format(poses, filepath):
    """Save poses in TUM format: timestamp x y z qx qy qz qw"""
    with open(filepath, 'w') as f:
        for i, pose in enumerate(poses):
            f.write(f"{i} {' '.join(map(str, pose[1:]))}\n")


def load_ply(filepath):
    """Load PLY point cloud"""
    points, colors = [], []
    with open(filepath, 'r') as f:
        header = True
        for line in f:
            if header:
                if 'end_header' in line:
                    header = False
                continue
            parts = line.strip().split()
            if len(parts) >= 6:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.array(points), np.array(colors)


@dataclass
class Args:
    """Evaluation arguments"""
    seq: str = '00'
    data_root: str = f'/home/{os.environ.get("USER", "user")}/ws/datasets'
    result_dir: str = './build/results/KITTI'


if __name__ == '__main__':
    args = tyro.cli(Args)
    
    # Paths
    result_dir = Path(args.result_dir) / args.seq
    traj_file = result_dir / 'traj_est.txt'
    traj_ba_file = result_dir / 'traj_est_ba.txt'
    traj_pgo_file = result_dir / 'traj_est_pgo.txt'
    pcd_file = result_dir / 'pcd.ply'
    pcd_ba_file = result_dir / 'pcd_ba.ply'
    gt_file = Path(args.data_root) / 'kitti-odom/data_odometry_poses/dataset/poses' / f'{args.seq}.txt'
    
    # Load trajectories
    print(f"Loading before BA: {traj_file}")
    est_poses_before = load_trajectory(traj_file)
    
    est_poses_after = None
    if traj_ba_file.exists():
        print(f"Loading after BA: {traj_ba_file}")
        est_poses_after = load_trajectory(traj_ba_file)
    else:
        print(f"BA trajectory not found: {traj_ba_file}, using before BA")
        est_poses_after = est_poses_before
    
    est_poses_pgo = None
    if traj_pgo_file.exists():
        print(f"Loading after PGO: {traj_pgo_file}")
        est_poses_pgo = load_trajectory(traj_pgo_file)
    else:
        print(f"PGO trajectory not found: {traj_pgo_file}, skipping PGO evaluation")
    

    # Load GT poses if available
    class EvalConfig:
        load_gt_pose = gt_file is not None
    if gt_file:
        loader = KITTILoader(EvalConfig())
        loader.config.data_root = args.data_root
        loader.config.seq = args.seq
        gt_poses = loader.load_gt_pose()
    
    # Convert GT poses to format: (frame_idx, tx, ty, tz, qx, qy, qz, qw)
    gt_poses_vec = []
    for i, gt_pose in enumerate(gt_poses):
        t = gt_pose[:3, 3]
        q = Rotation.from_matrix(gt_pose[:3, :3]).as_quat()  # Returns [x, y, z, w]
        gt_poses_vec.append((i, t[0], t[1], t[2], q[0], q[1], q[2], q[3]))
    
    # Convert estimated poses to format: (frame_idx, tx, ty, tz, qx, qy, qz, qw)
    # pose format: [timestamp, tx, ty, tz, qx, qy, qz, qw]
    est_poses_before_list = []
    for i, pose in enumerate(est_poses_before):
        est_poses_before_list.append((i, pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7]))
    
    est_poses_after_list = []
    for i, pose in enumerate(est_poses_after):
        est_poses_after_list.append((i, pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7]))
    
    est_poses_pgo_list = []
    if est_poses_pgo is not None:
        for i, pose in enumerate(est_poses_pgo):
            est_poses_pgo_list.append((i, pose[1], pose[2], pose[3], pose[4], pose[5], pose[6], pose[7]))
    
    # Evaluate before BA
    ate_trans_before, ate_rot_before = compute_ate(gt_poses_vec, est_poses_before_list)
    rpe_trans_before, rpe_rot_before = compute_rpe(gt_poses_vec, est_poses_before_list)
    print(f"\n[Before BA] ATE: {ate_trans_before:.4f} m, {ate_rot_before:.4f} deg")
    print(f"[Before BA] RPE: {rpe_trans_before:.4f} m, {rpe_rot_before:.4f} deg")
    
    # Evaluate after BA
    ate_trans_after, ate_rot_after = compute_ate(gt_poses_vec, est_poses_after_list)
    rpe_trans_after, rpe_rot_after = compute_rpe(gt_poses_vec, est_poses_after_list)
    print(f"\n[After BA]  ATE: {ate_trans_after:.4f} m, {ate_rot_after:.4f} deg")
    print(f"[After BA]  RPE: {rpe_trans_after:.4f} m, {rpe_rot_after:.4f} deg")
    
    # Evaluate after PGO
    if est_poses_pgo is not None and len(est_poses_pgo_list) > 0:
        ate_trans_pgo, ate_rot_pgo = compute_ate(gt_poses_vec, est_poses_pgo_list)
        rpe_trans_pgo, rpe_rot_pgo = compute_rpe(gt_poses_vec, est_poses_pgo_list)
        print(f"\n[After PGO] ATE: {ate_trans_pgo:.4f} m, {ate_rot_pgo:.4f} deg")
        print(f"[After PGO] RPE: {rpe_trans_pgo:.4f} m, {rpe_rot_pgo:.4f} deg")
    
    # ── Bird's eye view plot (XZ plane, KITTI convention) ──
    # GT is in KITTI CV frame: x=right, y=down, z=forward
    # Estimated is in ROS frame, but Umeyama aligns it to GT frame
    gt_xyz = np.array([[p[1], p[2], p[3]] for p in gt_poses_vec])

    def align_traj(est_list):
        pr_xyz = np.array([[p[1], p[2], p[3]] for p in est_list])
        s, R, t = umeyama_alignment(pr_xyz, gt_xyz[:len(pr_xyz)])
        return (s * (R @ pr_xyz.T).T) + t

    est_before_aligned = align_traj(est_poses_before_list)
    est_pgo_aligned = align_traj(est_poses_pgo_list) if est_poses_pgo_list else None

    fig, ax = plt.subplots(figsize=(10, 10))
    # KITTI bird's eye: x (right) vs z (forward)
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'k-', linewidth=2, label='Ground Truth')
    ax.plot(est_before_aligned[:, 0], est_before_aligned[:, 2], 'b-', linewidth=1, alpha=0.7, label='Before BA/PGO')
    if est_pgo_aligned is not None:
        ax.plot(est_pgo_aligned[:, 0], est_pgo_aligned[:, 2], 'r-', linewidth=1, alpha=0.7, label='After PGO')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(f'Bird\'s Eye View - KITTI Seq {args.seq}')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plot_path = result_dir / 'traj_bev.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved bird's eye view plot to: {plot_path}")
    plt.show()

    viz = Visualizer(EvalConfig())
    
    # Estimated poses (before BA)
    for i, pose in enumerate(est_poses_before):
        t = np.array([pose[1], pose[2], pose[3]])
        q = np.array([pose[4], pose[5], pose[6], pose[7]])  # qx, qy, qz, qw
        R = Rotation.from_quat(q).as_matrix()
        T_wc_ros = np.eye(4)
        T_wc_ros[:3, :3] = R
        T_wc_ros[:3, 3] = t
        viz.add_cam_frame(T_wc_ros, "estimated_before_ba", i)
        gt_poses_ros = CoordTransform.T_cv_to_ros @ gt_poses[i]
        viz.add_cam_frame(gt_poses_ros, "gt_frames", i)
    
    # Optimized poses (after BA)
    for i, pose in enumerate(est_poses_after):
        t = np.array([pose[1], pose[2], pose[3]])
        q = np.array([pose[4], pose[5], pose[6], pose[7]])
        R = Rotation.from_quat(q).as_matrix()
        T_wc_ros = np.eye(4)
        T_wc_ros[:3, :3] = R
        T_wc_ros[:3, 3] = t
        viz.add_cam_frame(T_wc_ros, "estimated_after_ba", i)
    
    # Optimized poses (after PGO)
    if est_poses_pgo is not None:
        for i, pose in enumerate(est_poses_pgo):
            t = np.array([pose[1], pose[2], pose[3]])
            q = np.array([pose[4], pose[5], pose[6], pose[7]])
            R = Rotation.from_quat(q).as_matrix()
            T_wc_ros = np.eye(4)
            T_wc_ros[:3, :3] = R
            T_wc_ros[:3, 3] = t
            viz.add_cam_frame(T_wc_ros, "estimated_after_pgo", i)
    
    # Point cloud
    if pcd_file.exists():
        points, colors = load_ply(pcd_file)
        pcd_handle = viz.get_pcd_handle("pointcloud")
        pcd_handle.points = points
        pcd_handle.colors = colors / 255.0
    
    # optimized point cloud
    if pcd_ba_file.exists():
        points, colors = load_ply(pcd_ba_file)
        pcd_handle = viz.get_pcd_handle("pointcloud_optimized")
        pcd_handle.points = points
        pcd_handle.colors = colors / 255.0
    
    print(f"\nViser: http://localhost:8080")
    input("Press Enter to exit...")
