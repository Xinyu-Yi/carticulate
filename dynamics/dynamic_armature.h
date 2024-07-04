#pragma once

#include <string>
#include <vector>
#include "Eigen/Core"


struct DynamicArmature {
    int n_joints;                          // number of joints
    std::string name;                      // armature name
    std::vector<int> parent;               // parent joint index (must satisfying parent[i] < i)
    std::vector<Eigen::Vector3f> bone;     // joint local position in the parent frame
    Eigen::Vector3f gravity;               // gravitional acceleration in the world frame
    std::vector<Eigen::Vector3f> com;      // body center of mass in the joint frame
    std::vector<float> mass;               // body mass
#ifndef USE_DIAGONAL_INERTIA
    std::vector<Eigen::Matrix3f> inertia;  // body inertia in the joint frame (Ixx, Iyy, Izz, Ixy, Iyz, Ixz)
#else
    std::vector<Eigen::Vector3f> inertia;  // body inertia in the joint frame (Ixx, Iyy, Izz)
#endif

    DynamicArmature() : n_joints(0) {}
    DynamicArmature(const std::string &armature_file);
    void print() const;   // print armature information
};
