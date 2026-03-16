cmake_minimum_required(VERSION 3.14)
project(mono_vo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)

add_executable(mono_vo src/mono.cpp)
target_include_directories(mono_vo PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  include
)

# Link libraries
target_link_libraries(mono_vo 
  ${OpenCV_LIBS} 
  Eigen3::Eigen
  Ceres::ceres
  Sophus::Sophus
)