#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

// other files
#include "backend.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

class MonoVO 
{
public:
    Config config;
    cv::Mat K;
    std::vector<std::string> img_files;
    int num_frames;
    std::vector<Eigen::Matrix4d> gt_pose;
    std::vector<Eigen::Matrix4d> T_wc_list;

    // point cloud
    std::vector<std::array<double,3>> pcd_all;
    std::vector<std::array<uint8_t,3>> pcd_colors_all;

    // Bundle adjustment data
    std::vector<Eigen::Matrix4d> kf_poses;
    std::vector<Eigen::MatrixX2d> kf_keypoints;
    std::vector<std::vector<int>> kf_point_ids;
    std::vector<Eigen::Vector3d> points_3d;
    std::vector<int> kf_frame_indices;  // Track which frames are keyframes
    std::unordered_map<int, size_t> point_id_to_pcd_idx;  // Map point ID to pcd_all index

    // Optical flow tracking state
    std::vector<cv::Point2f> prev_tracked_pts;
    std::vector<int> prev_tracked_pids;  // -1 = no 3D point yet
    cv::Mat prev_img_gray;

    std::string out_pose_file;
    std::string out_pose_ba_file;  // After BA
    std::string out_ply_file;
    std::string out_ply_ba_file;  // Optimized after BA

    // online pose graph optimization
    PoseGraph pose_graph;
    LoopDetector loop_detector;
    SaveIO save_io;

    MonoVO(const Config& cfg) : config(cfg) 
    {
        KITTILoader loader(config);
        img_files  = loader.load_img_files();
        num_frames = (int)img_files.size();
        K          = loader.load_intrinsics();
        if (config.load_gt_pose)
            gt_pose = loader.load_gt_pose();

        fs::create_directories(config.result_dir);
        out_pose_file = config.result_dir + "/" + "traj_est.txt";
        out_pose_ba_file = config.result_dir + "/" + "traj_est_ba.txt";
        out_ply_file  = config.result_dir + "/" + "pcd.ply";
        out_ply_ba_file = config.result_dir + "/" + "pcd_ba.ply";

        // truncate pose files
        std::ofstream(out_pose_file, std::ios::trunc).close();
        std::ofstream(out_pose_ba_file, std::ios::trunc).close();

        std::cout << "MonoVO initialized " << std::endl;
        std::cout << "num of frames: " << num_frames << std::endl;
        
        // Initialize loop detector
        loop_detector = LoopDetector(K);
    }


    void detect_features(const cv::Mat& img, std::vector<cv::Point2f>& pts, int max_pts = 5000)
    {
        cv::goodFeaturesToTrack(img, pts, max_pts, 0.005, 10);
        
        if (!pts.empty()) {
            cv::cornerSubPix(img, pts, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01));
        }
    }


    void track_features(
        const cv::Mat& prev_gray, const cv::Mat& curr_img_gray,
        std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& curr_pts,
        std::vector<int>& prev_pids, std::vector<int>& curr_pids)
    {
        if (prev_pts.empty()) 
        { 
            curr_pts.clear(); 
            curr_pids.clear(); 
            return; 
        }

        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_gray, curr_img_gray, prev_pts, curr_pts, status, err);

        // Bidirectional check
        std::vector<cv::Point2f> back_pts;
        std::vector<uchar> status_back;
        cv::calcOpticalFlowPyrLK(curr_img_gray, prev_gray, curr_pts, back_pts, status_back, err);

        std::vector<cv::Point2f> good_prev, good_curr;
        std::vector<int> good_pids;
        for (size_t i = 0; i < status.size(); i++) 
        {
            if (!status[i] || !status_back[i]) continue;
            double dist = cv::norm(prev_pts[i] - back_pts[i]);
            if (dist > 1.0) 
                continue;
            good_prev.push_back(prev_pts[i]);
            good_curr.push_back(curr_pts[i]);
            good_pids.push_back(prev_pids[i]);
        }
        prev_pts = good_prev;
        curr_pts = good_curr;
        curr_pids = good_pids;
    }


    Eigen::Matrix4d compute_pose(
        std::vector<cv::Point2f>& pts1,
        std::vector<cv::Point2f>& pts2,
        std::vector<int>& pids)
    {
        double fx = K.at<double>(0,0);
        double cx = K.at<double>(0,2);
        double cy = K.at<double>(1,2);

        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(pts1, pts2, fx, cv::Point2d(cx, cy),
                                         cv::RANSAC, 0.999, 1.0, mask);

        // filter to inliers
        std::vector<cv::Point2f> p1_in, p2_in;
        std::vector<int> pids_in;
        for (int j = 0; j < mask.rows; j++)
        {
            if (mask.at<uchar>(j)) 
            {
                p1_in.push_back(pts1[j]);
                p2_in.push_back(pts2[j]);
                pids_in.push_back(pids[j]);
            }
        }
        pts1 = p1_in;
        pts2 = p2_in;
        pids = pids_in;

        cv::Mat R_cv, t_cv;
        cv::recoverPose(E, pts1, pts2, R_cv, t_cv, fx, cv::Point2d(cx, cy));

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        cv::cv2eigen(R_cv, R);
        cv::cv2eigen(t_cv, t);

        Eigen::Matrix4d T_21 = Eigen::Matrix4d::Identity();
        T_21.block<3,3>(0,0) = R;
        T_21.block<3,1>(0,3) = t;

        return T_21.inverse();
    }

    // ─────────── triangulate with cheirality check ───────────
    // T_1w, T_2w are world-to-camera (T_wc) poses
    // returns 3D points (3×N) and valid mask
    Eigen::MatrixXd triangulate(
        const std::vector<cv::Point2f>& kp1,
        const std::vector<cv::Point2f>& kp2,
        const Eigen::Matrix4d& T_1w,
        const Eigen::Matrix4d& T_2w,
        std::vector<bool>& valid_mask)
    {
        // build projection matrices: P = K @ inv(T_wc)[:3,:]
        Eigen::Matrix4d T_1w_inv = T_1w.inverse();
        Eigen::Matrix4d T_2w_inv = T_2w.inverse();

        cv::Mat K_cv = K;
        cv::Mat P1_34, P2_34;
        Eigen::Matrix<double, 3, 4> E1 = T_1w_inv.block<3,4>(0,0);
        Eigen::Matrix<double, 3, 4> E2 = T_2w_inv.block<3,4>(0,0);

        cv::Mat E1_cv(3, 4, CV_64F), E2_cv(3, 4, CV_64F);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++) {
                E1_cv.at<double>(r,c) = E1(r,c);
                E2_cv.at<double>(r,c) = E2(r,c);
            }

        P1_34 = K_cv * E1_cv;
        P2_34 = K_cv * E2_cv;

        // convert keypoints to 2×N
        int N = (int)kp1.size();
        cv::Mat pts1_2d(2, N, CV_64F), pts2_2d(2, N, CV_64F);
        for (int i = 0; i < N; i++) {
            pts1_2d.at<double>(0, i) = kp1[i].x;
            pts1_2d.at<double>(1, i) = kp1[i].y;
            pts2_2d.at<double>(0, i) = kp2[i].x;
            pts2_2d.at<double>(1, i) = kp2[i].y;
        }

        cv::Mat X_homo;
        cv::triangulatePoints(P1_34, P2_34, pts1_2d, pts2_2d, X_homo);

        // convert to 3D (3×N)
        Eigen::MatrixXd X(3, N);
        for (int i = 0; i < N; i++) {
            double w = X_homo.at<double>(3, i);
            X(0, i) = X_homo.at<double>(0, i) / w;
            X(1, i) = X_homo.at<double>(1, i) / w;
            X(2, i) = X_homo.at<double>(2, i) / w;
        }

        // cheirality check in both camera frames
        valid_mask.resize(N);
        for (int i = 0; i < N; i++) {
            Eigen::Vector4d Xh;
            Xh << X(0,i), X(1,i), X(2,i), 1.0;

            double z1 = (T_1w_inv.row(2) * Xh)(0);
            double z2 = (T_2w_inv.row(2) * Xh)(0);
            bool valid = (z1 > 0) && (z2 > 0);

            // depth clamp in second camera
            if (valid && config.max_depth > 0)
                valid = valid && (z2 < config.max_depth);

            valid_mask[i] = valid;
        }

        return X; // 3×N, caller filters by valid_mask
    }

    // ─────────── map points to ROS frame with colors ─────────
    void mapping_points(
        const std::vector<cv::Point2f>& pts,
        const Eigen::MatrixXd& pcd, // 3×M in CV world frame
        const cv::Mat& curr_img)
    {
        int M = (int)pcd.cols();
        Eigen::Matrix4d T_cv_to_ros = CoordTransform::T_cv_to_ros();

        for (int i = 0; i < M; i++) 
        {
            // Transform from CV world frame to ROS world frame
            Eigen::Vector4d X_cv_world;
            X_cv_world << pcd(0,i), pcd(1,i), pcd(2,i), 1.0;
            Eigen::Vector4d X_ros_world = T_cv_to_ros * X_cv_world;
            pcd_all.push_back({X_ros_world(0), X_ros_world(1), X_ros_world(2)});

            // sample color from image (BGR → RGB)
            int u = std::clamp((int)pts[i].x, 0, curr_img.cols - 1);
            int v = std::clamp((int)pts[i].y, 0, curr_img.rows - 1);
            cv::Vec3b bgr = curr_img.at<cv::Vec3b>(v, u);
            pcd_colors_all.push_back({bgr[2], bgr[1], bgr[0]}); // RGB
        }
    }

    bool meet_keyframe_criteria(
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        const Eigen::Matrix4d& T_cc,
        double min_parallax = 1.0,
        double min_translation = 0.1)
    {
        double translation = T_cc.block<3,1>(0,3).norm();
        if (translation < min_translation) return false;

        // median parallax
        std::vector<double> par;
        for (size_t i = 0; i < pts1.size(); i++) {
            double dx = pts2[i].x - pts1[i].x;
            double dy = pts2[i].y - pts1[i].y;
            par.push_back(std::sqrt(dx*dx + dy*dy));
        }
        std::nth_element(par.begin(), par.begin() + par.size()/2, par.end());
        return par[par.size()/2] >= min_parallax;
    }

    void run() 
    {
        Eigen::Matrix4d prev_T_wc = Eigen::Matrix4d::Identity();
        bool is_kf;
        // for (int frame_idx = 0; frame_idx < num_frames; frame_idx++)
        for (int frame_idx = 0; frame_idx < 500; frame_idx++)
        {
            cv::Mat curr_img = cv::imread(img_files[frame_idx], cv::IMREAD_COLOR);
            cv::Mat curr_img_gray;
            cv::cvtColor(curr_img, curr_img_gray, cv::COLOR_BGR2GRAY);
            Eigen::Matrix4d curr_T_wc;
            is_kf = false;

            if (frame_idx == 0) 
            {
                curr_T_wc = Eigen::Matrix4d::Identity();
                is_kf = true;
                
                // detect initial features
                detect_features(curr_img_gray, prev_tracked_pts);
                prev_tracked_pids.assign(prev_tracked_pts.size(), -1);
                prev_img_gray = curr_img_gray;

                // Add frame 0 as first keyframe
                kf_poses.push_back(curr_T_wc);
                kf_frame_indices.push_back(0);
                kf_keypoints.push_back(Eigen::MatrixX2d(0, 2));  // Empty keypoints for frame 0
                kf_point_ids.push_back(std::vector<int>());  // Empty point IDs for frame 0
                
                // add frame 0, the 1st keyframe to pose graph
                pose_graph.add_keyframe(0, curr_T_wc, Sophus::SE3d());
                loop_detector.add_keyframe(curr_img_gray); 
            } 
            else 
            {
                // Track features with optical flow
                std::vector<cv::Point2f> prev_pts = prev_tracked_pts;
                std::vector<cv::Point2f> curr_pts;
                std::vector<int> prev_pids = prev_tracked_pids;
                std::vector<int> curr_pids;
                track_features(prev_img_gray, curr_img_gray, prev_pts, curr_pts, prev_pids, curr_pids);

                if (prev_pts.size() < 10) 
                {
                    std::cerr << "frame " << frame_idx << ": too few tracks\n";
                    curr_T_wc = prev_T_wc;
                    // Re-detect
                    detect_features(curr_img_gray, prev_tracked_pts);
                    prev_tracked_pids.assign(prev_tracked_pts.size(), -1);
                    prev_img_gray = curr_img_gray;
                    continue;
                }

                Eigen::Matrix4d T_prev_curr = compute_pose(prev_pts, curr_pts, curr_pids);

                curr_T_wc = prev_T_wc * T_prev_curr;
                T_wc_list.push_back(curr_T_wc);

                // keyframe check → triangulate
                is_kf = meet_keyframe_criteria(prev_pts, curr_pts, T_prev_curr);
                if (is_kf) 
                {
                    std::vector<bool> valid_mask;
                    Eigen::MatrixXd pcd = triangulate(prev_pts, curr_pts, prev_T_wc, curr_T_wc, valid_mask);

                    std::vector<int> point_ids;
                    std::vector<cv::Point2f> kps_valid;

                    // For colored point cloud — only new points
                    Eigen::MatrixXd pcd_new(3, 0);
                    std::vector<cv::Point2f> kps_new;
                    std::vector<int> new_pids;  // track which pids are new

                    // Also collect observations for previous keyframe
                    std::vector<int> prev_kf_new_pids;
                    std::vector<cv::Point2f> prev_kf_new_kps;

                    for (int j = 0; j < (int)valid_mask.size(); j++) 
                    {
                        if (!valid_mask[j]) continue;

                        int pid = curr_pids[j];
                        if (pid < 0) {
                            // New point — triangulate and assign pid
                            pid = static_cast<int>(points_3d.size());
                            points_3d.push_back(Eigen::Vector3d(pcd(0,j), pcd(1,j), pcd(2,j)));
                            curr_pids[j] = pid;

                            pcd_new.conservativeResize(Eigen::NoChange, pcd_new.cols() + 1);
                            pcd_new.col(pcd_new.cols() - 1) = pcd.col(j);
                            kps_new.push_back(curr_pts[j]);
                            new_pids.push_back(pid);

                            // Add observation to previous keyframe too
                            prev_kf_new_pids.push_back(pid);
                            prev_kf_new_kps.push_back(prev_pts[j]);
                        }
                        point_ids.push_back(pid);
                        kps_valid.push_back(curr_pts[j]);
                    }

                    // Append new observations to previous keyframe
                    if (!prev_kf_new_pids.empty() && kf_keypoints.size() > 0) 
                    {
                        size_t prev_kf = kf_keypoints.size() - 1;
                        
                        // Append to kf_point_ids
                        for (int pid : prev_kf_new_pids)
                            kf_point_ids[prev_kf].push_back(pid);
                        
                        // Append to kf_keypoints
                        int old_rows = kf_keypoints[prev_kf].rows();
                        int new_rows = old_rows + (int)prev_kf_new_kps.size();
                        Eigen::MatrixX2d expanded(new_rows, 2);
                        if (old_rows > 0)
                            expanded.topRows(old_rows) = kf_keypoints[prev_kf];
                        for (int i = 0; i < (int)prev_kf_new_kps.size(); i++) {
                            expanded(old_rows + i, 0) = prev_kf_new_kps[i].x;
                            expanded(old_rows + i, 1) = prev_kf_new_kps[i].y;
                        }
                        kf_keypoints[prev_kf] = expanded;
                    }

                    // Color mapping for new points only
                    if (pcd_new.cols() > 0) 
                    {
                        size_t pcd_start_idx = pcd_all.size();
                        mapping_points(kps_new, pcd_new, curr_img);

                        for (size_t i = 0; i < new_pids.size(); i++) {
                            point_id_to_pcd_idx[new_pids[i]] = pcd_start_idx + i;
                        }
                    }

                    // Always add to BA arrays to stay in sync with pose_graph / loop_detector
                    kf_poses.push_back(curr_T_wc);
                    kf_frame_indices.push_back(frame_idx);
                    if (!kps_valid.empty())
                    {
                        Eigen::MatrixX2d kp_mat(kps_valid.size(), 2);
                        for (size_t i = 0; i < kps_valid.size(); i++) {
                            kp_mat(i, 0) = kps_valid[i].x;
                            kp_mat(i, 1) = kps_valid[i].y;
                        }
                        kf_keypoints.push_back(kp_mat);
                        kf_point_ids.push_back(point_ids);
                    }
                    else
                    {
                        kf_keypoints.push_back(Eigen::MatrixX2d(0, 2));
                        kf_point_ids.push_back(std::vector<int>());
                    }

                    Sophus::SE3d T_ij_measured(T_prev_curr);
                    pose_graph.add_keyframe(frame_idx, curr_T_wc, T_ij_measured);

                    // Loop closure detection
                    loop_detector.add_keyframe(curr_img_gray);
                    auto [loop_kf, T_loop] = loop_detector.detect();
                    if (loop_kf >= 0)
                    {
                        int curr_pg_idx = pose_graph.size() - 1;
                        std::cout << "[Loop] Loop detection found! frame " << frame_idx 
                                  << " <-> frame " << kf_frame_indices[loop_kf] 
                                  << " (pg " << loop_kf << " <-> " << curr_pg_idx << ")\n";

                        // Simple loop closure: use recoverPose rotation
                        Eigen::Matrix4d T_meas = Eigen::Matrix4d::Identity();
                        T_meas.block<3,3>(0,0) = T_loop.rotationMatrix();
                        Sophus::SE3d T_loop_edge(T_meas);

                        Eigen::Matrix<double, 7, 7> loop_info = Eigen::Matrix<double, 7, 7>::Zero();
                        loop_info.block<3,3>(0,0) = 50.0 * Eigen::Matrix3d::Identity();  // translation
                        loop_info.block<3,3>(3,3) = 5.0 * Eigen::Matrix3d::Identity();   // rotation
                        loop_info(6,6) = 10.0;                                            // scale

                        pose_graph.add_loop_edge(loop_kf, curr_pg_idx, T_loop_edge, loop_info);
                    }
                }
                
                // Replenish lost tracks
                if (curr_pts.size() < 1500) 
                {
                    std::vector<cv::Point2f> new_pts;
                    detect_features(curr_img_gray, new_pts);
                    for (auto& np : new_pts) {
                        bool too_close = false;
                        for (auto& ep : curr_pts) {
                            if (cv::norm(np - ep) < 20) { too_close = true; break; }
                        }
                        if (!too_close) {
                            curr_pts.push_back(np);
                            curr_pids.push_back(-1);
                        }
                    }
                }

                prev_tracked_pts = curr_pts;
                prev_tracked_pids = curr_pids;
                prev_img_gray = curr_img_gray;
            }

            // show keypoints
            std::vector<cv::KeyPoint> kps_draw;
            for (auto& p : prev_tracked_pts)
                kps_draw.push_back(cv::KeyPoint(p, 1.f));
            cv::Mat img_kp;
            cv::drawKeypoints(curr_img, kps_draw, img_kp, cv::Scalar(0, 255, 0));
            cv::imshow("keypoints from current image", img_kp);
            cv::waitKey(1);
            
            prev_T_wc = curr_T_wc;
        }

        // Bundle adjustment
        std::cout << "\n[BA] Starting bundle adjustment with " << kf_poses.size() 
                    << " keyframes and " << points_3d.size() << " points\n";
        
        // Store original poses for comparison
        std::vector<Eigen::Matrix4d> kf_poses_before = kf_poses;
        
        Eigen::Matrix3d K_eigen;
        K_eigen << K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2),
                    K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2),
                    K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2);

        // Print observation stats
        std::unordered_map<int, int> obs_count;
        for (size_t k = 0; k < kf_point_ids.size(); k++)
            for (int pid : kf_point_ids[k])
                obs_count[pid]++;
        int multi_obs = 0;
        for (auto& [pid, cnt] : obs_count)
            if (cnt >= 2) multi_obs++;
        std::cout << "[BA] Points with 2+ observations: " << multi_obs 
                  << " / " << obs_count.size() << "\n";

        run_ba(kf_poses, kf_keypoints, kf_point_ids, points_3d, K_eigen);
            
        // Save original point cloud before updating with optimized points
        if (!pcd_all.empty()) 
        {
            save_io.save_ply(out_ply_file, pcd_all, pcd_colors_all);
            std::cout << "[BA] Saved " << pcd_all.size() << " original points → " << out_ply_file << "\n";
        }
        
        // Update pcd_all with optimized points (convert CV world to ROS world)
        Eigen::Matrix4d T_cv_to_ros_pcd = CoordTransform::T_cv_to_ros();
        for (size_t pid = 0; pid < points_3d.size(); pid++) 
        {
            auto it = point_id_to_pcd_idx.find(static_cast<int>(pid));
            if (it != point_id_to_pcd_idx.end()) {
                size_t pcd_idx = it->second;
                if (pcd_idx < pcd_all.size()) {
                    // points_3d[pid] is in CV world frame, transform to ROS world frame
                    Eigen::Vector4d p_cv_world(points_3d[pid](0), points_3d[pid](1), points_3d[pid](2), 1.0);
                    Eigen::Vector4d p_ros_world = T_cv_to_ros_pcd * p_cv_world;
                    pcd_all[pcd_idx] = {p_ros_world(0), p_ros_world(1), p_ros_world(2)};
                }
            }
        }
        
        // Truncate files
        std::ofstream(out_pose_file, std::ios::trunc).close();
        std::ofstream(out_pose_ba_file, std::ios::trunc).close();

        // Save output pose before and after BA
        for (size_t i = 0; i < kf_poses.size(); i++) 
        {
            save_io.save_output_pose(kf_frame_indices[i], kf_poses_before[i], out_pose_file);
            save_io.save_output_pose(kf_frame_indices[i], kf_poses[i], out_pose_ba_file);
        }
        
        // Save optimized point cloud
        std::cout << "[BA] About to save optimized point cloud to: " << out_ply_ba_file << "\n";
        std::cout << "[BA] pcd_all.size() = " << pcd_all.size() << "\n";
        if (!pcd_all.empty()) 
        {
            save_io.save_ply(out_ply_ba_file, pcd_all, pcd_colors_all);
            std::cout << "[BA] Saved " << pcd_all.size() << " optimized points → " << out_ply_ba_file << "\n";
        } 
        else 
        {
            std::cerr << "[BA] ERROR: pcd_all is empty, cannot save optimized point cloud!\n";
        }

        // After BA saving, run final PGO optimization and save
        std::cout << "\n[PGO] Running final pose graph optimization with "
                  << pose_graph.size() << " keyframes and " << pose_graph.num_edges() << " edges\n";
        pose_graph.run_sim3_pgo(pose_graph.get_poses(), pose_graph.get_edges());
        
        std::string out_pose_pgo_file = config.result_dir + "/traj_est_pgo.txt";
        std::ofstream(out_pose_pgo_file, std::ios::trunc).close();
        const auto& pgo_poses = pose_graph.get_poses();
        const auto& pgo_indices = pose_graph.get_kf_indices();
        for (size_t i = 0; i < pgo_poses.size(); i++) 
        {
            save_io.save_output_pose(pgo_indices[i], pgo_poses[i], out_pose_pgo_file);
        }
        std::cout << "[PGO] Saved " << pgo_poses.size() << " optimized poses → " << out_pose_pgo_file << "\n";
        
        cv::destroyAllWindows();
        
        std::cout << "Processing complete.\n";
    }
};


int main(int argc, char** argv) 
{
    Config config;
    // simple arg override: ./mono_vo [seq]
    if (argc >= 2) config.seq = argv[1];

    MonoVO vo(config);
    vo.run();
    
    return 0;
}