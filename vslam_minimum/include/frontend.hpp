#pragma once

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

#include "utils.hpp"


class Tracking
{
public:
    Config config;
    
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
        std::vector<int>& pids,
        const cv::Mat& K)
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
        std::vector<bool>& valid_mask,
        const cv::Mat& K)
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
};


class Mapping
{
public:
    Config config;
    std::vector<std::array<double,3>> pcd_all;
    std::vector<std::array<uint8_t,3>> pcd_colors_all;
    
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
};