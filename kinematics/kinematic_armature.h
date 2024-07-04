#pragma once

#include <string>
#include <vector>
#include "Eigen/Core"


struct KinematicArmature {
    int n_joints;       // number of joints
    std::string name;       // armature name
    std::vector<int> parent;    // parent joint index (must satisfying parent[i] < i)
    std::vector<Eigen::Vector3f> bone;  // joint local position expressed in the parent frame
    
    KinematicArmature() : n_joints(0) {}
    KinematicArmature(const std::string &armature_file);
    void print() const;   // print armature information
};
