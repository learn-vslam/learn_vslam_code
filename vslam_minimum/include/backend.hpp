#pragma once

#include <vector>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


// bundle adjustment
struct ReprojectionError 
{
    ReprojectionError(double obs_u, double obs_v,
                      double fx, double fy, double cx, double cy)
        : obs_u_(obs_u), obs_v_(obs_v),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy){}

    template <typename T>
    bool operator()(const T* const se3_params,
                    const T* const point,
                    T* residuals) const 
    {
        Eigen::Map<Sophus::SE3<T> const> T_cw(se3_params);
        Eigen::Matrix<T, 3, 1> X_w(point[0], point[1], point[2]);
        Eigen::Matrix<T, 3, 1> X_c = T_cw * X_w;

        T inv_z = T(1.0) / X_c[2];
        T u_proj = T(fx_) * X_c[0] * inv_z + T(cx_);
        T v_proj = T(fy_) * X_c[1] * inv_z + T(cy_);

        residuals[0] = u_proj - T(obs_u_);
        residuals[1] = v_proj - T(obs_v_);
        return true;
    }

    static ceres::CostFunction* Create(double obs_u, double obs_v,
                                        double fx, double fy,
                                        double cx, double cy) 
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, Sophus::SE3d::num_parameters, 3>
        (
            new ReprojectionError(obs_u, obs_v, fx, fy, cx, cy)
        );
    }

    double obs_u_, obs_v_;
    double fx_, fy_, cx_, cy_;
};


inline void run_ba(std::vector<Eigen::Matrix4d>& kf_poses,
                const std::vector<Eigen::MatrixX2d>& kf_keypoints,
                const std::vector<std::vector<int>>& kf_point_ids,
                std::vector<Eigen::Vector3d>& points_3d,
                const Eigen::Matrix3d& K,
                int steps = 50)
{
    const int num_kf = static_cast<int>(kf_poses.size());
    const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);

    // Convert poses to SE3 (T_cw for projection)
    std::vector<Sophus::SE3d> cam_poses(num_kf);
    for (int i = 0; i < num_kf; ++i) 
    {
        cam_poses[i] = Sophus::SE3d(kf_poses[i].inverse());
    }

    // Copy points to contiguous array
    std::vector<std::array<double, 3>> pt_params(points_3d.size());
    for (size_t i = 0; i < points_3d.size(); ++i) {
        pt_params[i] = {points_3d[i](0), points_3d[i](1), points_3d[i](2)};
    }

    // Build Ceres problem
    ceres::Problem problem;
    ceres::Manifold* se3_manifold = new Sophus::Manifold<Sophus::SE3>();

    for (int k = 0; k < num_kf; ++k) 
    {
        problem.AddParameterBlock(cam_poses[k].data(), Sophus::SE3d::num_parameters, se3_manifold);
    }

    for (int k = 0; k < num_kf; ++k) 
    {
        for (int j = 0; j < static_cast<int>(kf_point_ids[k].size()); ++j) 
        {
            int pid = kf_point_ids[k][j];
            double u = kf_keypoints[k](j, 0);
            double v = kf_keypoints[k](j, 1);

            ceres::CostFunction* cost = ReprojectionError::Create(u, v, fx, fy, cx, cy);
            ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(cost, loss, cam_poses[k].data(), pt_params[pid].data());
        }
    }

    problem.SetParameterBlockConstant(cam_poses[0].data());

    // Solve
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = steps;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Print loss
    int n_obs = 0;
    for (int k = 0; k < num_kf; ++k) 
    {
        n_obs += static_cast<int>(kf_point_ids[k].size());
    }
    double rms_error = std::sqrt(summary.final_cost / n_obs);
    std::cout << "[BA] final RMS reprojection error: " << rms_error << " px" << std::endl;

    // Write back
    for (int i = 0; i < num_kf; ++i) 
    {
        kf_poses[i] = cam_poses[i].matrix().inverse();
    }
    for (size_t i = 0; i < points_3d.size(); ++i) 
    {
        points_3d[i] = Eigen::Vector3d(pt_params[i][0], pt_params[i][1], pt_params[i][2]);
    }
}


// pose graph optimization
struct RelativeSim3Error
{
    RelativeSim3Error(const Sophus::Sim3d& S_ij_measured,
                      const Eigen::Matrix<double, 7, 7>& sqrt_info)
        : S_ij_measured_(S_ij_measured), sqrt_info_(sqrt_info) {}

    template <typename T>
    bool operator()(const T* const sim3_i,
                    const T* const sim3_j,
                    T* residuals) const
    {
        Eigen::Map<Sophus::Sim3<T> const> S_wi(sim3_i);
        Eigen::Map<Sophus::Sim3<T> const> S_wj(sim3_j);

        Sophus::Sim3<T> S_ij_meas = S_ij_measured_.cast<T>();
        Sophus::Sim3<T> S_ij_est = S_wi.inverse() * S_wj;
        Sophus::Sim3<T> err = S_ij_meas.inverse() * S_ij_est;

        Eigen::Matrix<T, 7, 1> error = err.log();
        Eigen::Map<Eigen::Matrix<T, 7, 1>> res(residuals);
        res = sqrt_info_.cast<T>() * error;
        return true;
    }

    static ceres::CostFunction* Create(const Sophus::Sim3d& S_ij,
                                        const Eigen::Matrix<double, 7, 7>& sqrt_info)
    {
        return new ceres::AutoDiffCostFunction<RelativeSim3Error, 7,
                                                Sophus::Sim3d::num_parameters,
                                                Sophus::Sim3d::num_parameters>(
            new RelativeSim3Error(S_ij, sqrt_info));
    }

    Sophus::Sim3d S_ij_measured_;
    Eigen::Matrix<double, 7, 7> sqrt_info_;
};


struct Sim3Edge
{
    int i, j;
    Sophus::Sim3d S_ij;
    Eigen::Matrix<double, 7, 7> info;
};


class PoseGraph
{
public:
    Sophus::Sim3d se3_to_sim3(const Sophus::SE3d& T, double scale = 1.0)
    {
        return Sophus::Sim3d(Sophus::RxSO3d(scale, T.rotationMatrix()), T.translation());
    }

    // Sim3 tangent order: [upsilon(3), omega(3), sigma(1)]
    //                      translation, rotation, log_scale
    PoseGraph(double trans_weight = 10.0, double rot_weight = 100.0, double scale_weight = 10.0)
    {
        odom_info_ = Eigen::Matrix<double, 7, 7>::Zero();
        odom_info_.block<3,3>(0,0) = trans_weight * Eigen::Matrix3d::Identity();  // upsilon
        odom_info_.block<3,3>(3,3) = rot_weight * Eigen::Matrix3d::Identity();    // omega
        odom_info_(6,6) = scale_weight;                                           // sigma
    }

    void add_keyframe(int kf_idx, const Eigen::Matrix4d& T_wc,
                      const Sophus::SE3d& T_ij_measured)
    {
        kf_indices_.push_back(kf_idx);
        poses_.push_back(T_wc);

        if (poses_.size() > 1)
        {
            int i = static_cast<int>(poses_.size()) - 2;
            int j = static_cast<int>(poses_.size()) - 1;
            edges_.push_back({i, j, se3_to_sim3(T_ij_measured), odom_info_});
        }
    }

    void add_loop_edge(int i, int j, const Sophus::SE3d& T_ij,
                       const Eigen::Matrix<double, 7, 7>& info)
    {
        edges_.push_back({i, j, se3_to_sim3(T_ij), info});
        std::cout << "[PGO] Loop edge added: " << i << " <-> " << j << "\n";
    }

    const std::vector<Eigen::Matrix4d>& get_poses() const { return poses_; }
    std::vector<Eigen::Matrix4d>& get_poses() { return poses_; }
    const std::vector<int>& get_kf_indices() const { return kf_indices_; }
    const std::vector<Sim3Edge>& get_edges() const { return edges_; }
    int size() const { return static_cast<int>(poses_.size()); }
    int num_edges() const { return static_cast<int>(edges_.size()); }

    void run_sim3_pgo(std::vector<Eigen::Matrix4d>& kf_poses,
                        const std::vector<Sim3Edge>& edges,
                        int steps = 50)
    {
        const int num_kf = static_cast<int>(kf_poses.size());
        if (num_kf < 2 || edges.empty()) 
            return;

        std::vector<Sophus::Sim3d> poses(num_kf);
        for (int i = 0; i < num_kf; ++i)
        poses[i] = se3_to_sim3(Sophus::SE3d(kf_poses[i]));

        ceres::Problem problem;
        ceres::Manifold* manifold = new Sophus::Manifold<Sophus::Sim3>();

        for (int k = 0; k < num_kf; ++k)
            problem.AddParameterBlock(poses[k].data(), Sophus::Sim3d::num_parameters, manifold);

        for (const auto& edge : edges)
        {
            Eigen::Matrix<double, 7, 7> sqrt_info = edge.info.llt().matrixL();
            ceres::CostFunction* cost = RelativeSim3Error::Create(edge.S_ij, sqrt_info);
            problem.AddResidualBlock(cost, new ceres::HuberLoss(5.0),
                                poses[edge.i].data(), poses[edge.j].data());
        }

        problem.SetParameterBlockConstant(poses[0].data());

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = steps;
        options.minimizer_progress_to_stdout = true;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        std::cout << "[PGO-Sim3] final cost: " << summary.final_cost << "\n";

        double min_s = 1e9, max_s = 0;
        for (int i = 0; i < num_kf; ++i) 
        {
            double s = poses[i].scale();
            min_s = std::min(min_s, s);
            max_s = std::max(max_s, s);
        }
        std::cout << "[PGO-Sim3] scale range: [" << min_s << ", " << max_s << "]\n";

        for (int i = 0; i < num_kf; ++i) 
        {
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            T.block<3,3>(0,0) = poses[i].rotationMatrix();
            T.block<3,1>(0,3) = poses[i].translation();
            kf_poses[i] = T;
        }
    }

private:
    std::vector<Eigen::Matrix4d> poses_;
    std::vector<int> kf_indices_;
    std::vector<Sim3Edge> edges_;      // ONE vector: sequential + loop
    Eigen::Matrix<double, 7, 7> odom_info_;
};


// loop detection
class LoopDetector
{
public:
    LoopDetector() {}
    LoopDetector(const cv::Mat& K, int min_kf_gap = 100,
                 double sim_threshold = 0.96, int min_inliers = 80,
                 int cooldown_frames = 50, int required_consecutive = 3,
                 double desc_scale = 0.1)
        : K_(K), min_kf_gap_(min_kf_gap),
          sim_threshold_(sim_threshold), min_inliers_(min_inliers),
          cooldown_frames_(cooldown_frames), required_consecutive_(required_consecutive),
          desc_scale_(desc_scale) {}

    void add_keyframe(const cv::Mat& gray)
    {
        kf_images_.push_back(gray.clone());

        cv::Mat small;
        cv::resize(gray, small, cv::Size(0, 0), desc_scale_, desc_scale_);
        small.convertTo(small, CV_32F);
        cv::Mat desc = small.reshape(1, 1);
        cv::normalize(desc, desc);
        descriptors_.push_back(desc);

        if (cooldown_ > 0)
            cooldown_--;
    }

    std::pair<int, Sophus::SE3d> detect()
    {
        int curr = static_cast<int>(descriptors_.size()) - 1;
        if (curr < min_kf_gap_)
            return {-1, Sophus::SE3d()};

        if (cooldown_ > 0)
            return {-1, Sophus::SE3d()};

        int best = -1;
        double best_score = -1.0;
        for (int c = 0; c <= curr - min_kf_gap_; ++c)
        {
            double score = descriptors_[curr].dot(descriptors_[c]);
            if (score > best_score)
            {
                best_score = score;
                best = c;
            }
        }

        if (best_score < sim_threshold_ || best < 0)
        {
            consecutive_count_ = 0;
            last_candidate_ = -1;
            return {-1, Sophus::SE3d()};
        }

        if (last_candidate_ >= 0 && std::abs(best - last_candidate_) < 5)
            consecutive_count_++;
        else
            consecutive_count_ = 1;
        last_candidate_ = best;

        if (consecutive_count_ < required_consecutive_)
            return {-1, Sophus::SE3d()};

        Sophus::SE3d T_ij;
        if (!verify(curr, best, T_ij))
        {
            consecutive_count_ = 0;
            return {-1, Sophus::SE3d()};
        }

        consecutive_count_ = 0;
        last_candidate_ = -1;
        cooldown_ = cooldown_frames_;

        return {best, T_ij};
    }

    int num_keyframes() const { return static_cast<int>(descriptors_.size()); }

private:
    bool verify(int curr, int cand, Sophus::SE3d& T_ij)
    {
        std::vector<cv::Point2f> pts_cand;
        cv::goodFeaturesToTrack(kf_images_[cand], pts_cand, 3000, 0.01, 10);
        if ((int)pts_cand.size() < min_inliers_) return false;

        std::vector<cv::Point2f> pts_curr;
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(kf_images_[cand], kf_images_[curr],
                                  pts_cand, pts_curr, status, err);

        std::vector<cv::Point2f> back_pts;
        std::vector<uchar> status_back;
        cv::calcOpticalFlowPyrLK(kf_images_[curr], kf_images_[cand],
                                  pts_curr, back_pts, status_back, err);

        std::vector<cv::Point2f> good_cand, good_curr;
        for (size_t i = 0; i < status.size(); i++)
        {
            if (!status[i] || !status_back[i]) continue;
            if (cv::norm(pts_cand[i] - back_pts[i]) > 1.0) continue;
            good_cand.push_back(pts_cand[i]);
            good_curr.push_back(pts_curr[i]);
        }

        if ((int)good_cand.size() < min_inliers_) return false;

        double fx = K_.at<double>(0, 0);
        double cx = K_.at<double>(0, 2);
        double cy = K_.at<double>(1, 2);

        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(good_curr, good_cand, fx,
                                          cv::Point2d(cx, cy), cv::RANSAC, 0.999, 1.0, mask);

        cv::Mat R_cv, t_cv;
        int inliers = cv::recoverPose(E, good_curr, good_cand, R_cv, t_cv,
                                       fx, cv::Point2d(cx, cy), mask);

        if (inliers < min_inliers_) return false;

        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        cv::cv2eigen(R_cv, R);
        cv::cv2eigen(t_cv, t);

        Eigen::Matrix4d T_curr_cand = Eigen::Matrix4d::Identity();
        T_curr_cand.block<3,3>(0,0) = R;
        T_curr_cand.block<3,1>(0,3) = t;
        T_ij = Sophus::SE3d(T_curr_cand.inverse());

        std::cout << "[Loop] Verified: " << inliers << " inliers\n";
        return true;
    }

    cv::Mat K_;
    int min_kf_gap_ = 100;
    double sim_threshold_ = 0.96;
    int min_inliers_ = 80;
    int cooldown_frames_ = 50;
    int required_consecutive_ = 3;
    double desc_scale_ = 0.1;

    std::vector<cv::Mat> kf_images_;
    std::vector<cv::Mat> descriptors_;

    int last_candidate_ = -1;
    int consecutive_count_ = 0;
    int cooldown_ = 0;
};
