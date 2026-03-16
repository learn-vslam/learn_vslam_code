#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace fs = std::filesystem;


class CoordTransform 
{
public:
    static Eigen::Matrix4d T_cv_to_ros() 
    {
        Eigen::Matrix4d T;
        T <<  0,  0,  1,  0,
             -1,  0,  0,  0,
              0, -1,  0,  0,
              0,  0,  0,  1;
        return T;
    }
};


class Config {
public:
    std::string data_root  = "/home/" + std::string(getenv("USER")) + "/ws/datasets";
    std::string dataset_type = "KITTI";
    bool   load_gt_pose    = true;
    std::string seq        = "09";
    int    key_frame_interval = 5;
    double max_depth       = 100.0;
    std::string result_dir; // set in constructor

    Config() 
    {
        result_dir = "./results/" + dataset_type + "/" + seq;
    }
};


class KITTILoader 
{
public:
    Config cfg;

    KITTILoader(const Config& c) : cfg(c) {}

    std::vector<std::string> load_img_files() 
    {
        std::string img_dir = cfg.data_root +
            "/kitti-odom/data_odometry_color/dataset/sequences/" +
            cfg.seq + "/image_2";
        std::vector<std::string> files;
        for (auto& e : fs::directory_iterator(img_dir))
            if (e.path().extension() == ".png")
                files.push_back(e.path().string());
        std::sort(files.begin(), files.end());
        return files;
    }

    // Loads from config/kitti_odom.yaml (relative to cwd)
    // Falls back to hardcoded KITTI seq 00-10 intrinsics if file missing
    cv::Mat load_intrinsics() 
    {
        double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;

        std::ifstream f("./config/kitti_odom.yaml");
        if (f.is_open()) {
            std::string line;
            while (std::getline(f, line)) {
                std::istringstream ss(line);
                std::string key;
                ss >> key;
                if (key == "fx:") ss >> fx;
                else if (key == "fy:") ss >> fy;
                else if (key == "cx:") ss >> cx;
                else if (key == "cy:") ss >> cy;
            }
        }

        return (cv::Mat_<double>(3,3) <<
            fx, 0, cx,
            0, fy, cy,
            0,  0,  1);
    }

    std::vector<Eigen::Matrix4d> load_gt_pose() 
    {
        std::string gt_path = cfg.data_root +
            "/kitti-odom/data_odometry_poses/dataset/poses/" +
            cfg.seq + ".txt";
        std::vector<Eigen::Matrix4d> poses;
        std::ifstream f(gt_path);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 4; c++)
                    ss >> T(r, c);
            poses.push_back(T);
        }
        return poses;
    }
};


class SaveIO
{
public:
    void save_output_pose(int idx, const Eigen::Matrix4d& T_wc, const std::string& filename) 
    {
        Eigen::Matrix4d T_wc_ros = CoordTransform::T_cv_to_ros() * T_wc * Eigen::Matrix4d::Identity().transpose();
        Eigen::Vector3d t = T_wc_ros.block<3,1>(0,3);
        Eigen::Quaterniond q(T_wc_ros.block<3,3>(0,0));
        std::ofstream f(filename, std::ios::app);
        f << idx << " "
        << t.x() << " " << t.y() << " " << t.z() << " "
        << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }


    void save_ply(const std::string& path,
                const std::vector<std::array<double,3>>& pts,
                const std::vector<std::array<uint8_t,3>>& colors)
    {
        std::ofstream f(path);
        f << "ply\nformat ascii 1.0\n"
        << "element vertex " << pts.size() << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        << "end_header\n";
        for (size_t i = 0; i < pts.size(); i++)
            f << pts[i][0] << " " << pts[i][1] << " " << pts[i][2] << " "
            << (int)colors[i][0] << " " << (int)colors[i][1] << " " << (int)colors[i][2] << "\n";
    }
};