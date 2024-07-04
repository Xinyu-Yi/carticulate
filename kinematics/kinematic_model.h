#pragma once

#include "kinematic_armature.h"
#include "sophus/se3.hpp"

class KinematicModel {
public:
    explicit KinematicModel(const std::string &armature_file) : armature(armature_file) { TPB.resize(armature.n_joints); TWB.resize(armature.n_joints); }
    explicit KinematicModel(const KinematicArmature &armature) : armature(armature) { TPB.resize(armature.n_joints); TWB.resize(armature.n_joints); }
    explicit KinematicModel(KinematicArmature &&armature) : armature(std::move(armature)) { TPB.resize(this->armature.n_joints); TWB.resize(this->armature.n_joints); }

    void print() const { armature.print(); }                                                     // print armature information
    KinematicArmature &get_armature() { return armature; }									     // get the armature
    const KinematicArmature &get_armature() const { return armature; }						     // get the armature
    void set_state_R(const std::vector<Eigen::Matrix3f> &pose, const Eigen::Vector3f &tran);     // set the pose and translation (rotation matrix)
    void set_state_q(const std::vector<Eigen::Quaternionf> &pose, const Eigen::Vector3f &tran);  // set the pose and translation (quaternion)
    void get_state_R(std::vector<Eigen::Matrix3f> &pose, Eigen::Vector3f &tran) const;           // get the pose and translation (rotation matrix)
    void get_state_q(std::vector<Eigen::Quaternionf> &pose, Eigen::Vector3f &tran) const;        // get the pose and translation (quaternion)
    void update_state(const Eigen::VectorXf &delta);                                             // update the pose and translation (right pertubation, translation first)

    Eigen::Vector3f get_position(int joint_idx, const Eigen::Vector3f &local_position = Eigen::Vector3f::Zero()) const;                             // get position in the world frame
    Eigen::Matrix3f get_orientation_R(int joint_idx, const Eigen::Matrix3f &local_orientation = Eigen::Matrix3f::Identity()) const;                 // get orientation in the world frame (rotation matrix)
    Eigen::Quaternionf get_orientation_q(int joint_idx, const Eigen::Quaternionf &local_orientation = Eigen::Quaternionf::Identity()) const;        // get orientation in the world frame (quaternion)
    Eigen::MatrixXf get_position_Jacobian(int joint_idx, const Eigen::Vector3f &local_position = Eigen::Vector3f::Zero()) const;                    // get position Jacobian: p(state + delta) = p(state) + J * delta
    Eigen::MatrixXf get_orientation_Jacobian_R(int joint_idx, const Eigen::Matrix3f &local_orientation = Eigen::Matrix3f::Identity()) const;        // get orientation (rotation matrix) Jacobian: R(state + delta) = R(state) + J * delta. R is flatten to 9x1 by concatenating three column vectors.
    Eigen::MatrixXf get_orientation_Jacobian_q(int joint_idx, const Eigen::Quaternionf &local_orientation = Eigen::Quaternionf::Identity()) const;  // get orientation (quaternion) Jacobian: q(state + delta) = q(state) + J * delta. q is 4x1 in (w, x, y, z) order in the Jacobian.

private:
    KinematicArmature armature;
    std::vector<Sophus::SE3f> TPB;   // transformation from parent to bone
    std::vector<Sophus::SE3f> TWB;   // transformation from world to bone
};