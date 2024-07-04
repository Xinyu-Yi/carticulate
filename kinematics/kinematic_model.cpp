#include <iostream>
#include "kinematic_model.h"
#include "sophus/so3.hpp"


static inline Eigen::Matrix4f left_quaternion(const Eigen::Quaternionf &q)
{
    Eigen::Matrix4f L;
    L << q.w(), -q.x(), -q.y(), -q.z(),
         q.x(), q.w(), -q.z(), q.y(),
         q.y(), q.z(), q.w(), -q.x(),
         q.z(), -q.y(), q.x(), q.w();
    return L;
}

static inline Eigen::Matrix4f right_quaternion(const Eigen::Quaternionf &q)
{
    Eigen::Matrix4f R;
    R << q.w(), -q.x(), -q.y(), -q.z(),
         q.x(), q.w(), q.z(), -q.y(),
         q.y(), -q.z(), q.w(), q.x(),
         q.z(), q.y(), -q.x(), q.w();
    return R;
}

void KinematicModel::set_state_R(const std::vector<Eigen::Matrix3f> &pose, const Eigen::Vector3f &tran)
{
    if (pose.size() != armature.n_joints) {
        throw std::runtime_error("Error: Pose size mismatch");
    }

    for (int i = 0; i < armature.n_joints; i++) {
        if (armature.parent[i] < 0) {   // root joint
            TPB[i] = Sophus::SE3f(pose[i], armature.bone[i] + tran);
            TWB[i] = TPB[i];
        }
        else {   // non-root joint
            TPB[i] = Sophus::SE3f(pose[i], armature.bone[i]);
            TWB[i] = TWB[armature.parent[i]] * TPB[i];
        }
    }
}

void KinematicModel::set_state_q(const std::vector<Eigen::Quaternionf> &pose, const Eigen::Vector3f &tran)
{
    if (pose.size() != armature.n_joints) {
        throw std::runtime_error("Error: Pose size mismatch");
    }

    for (int i = 0; i < armature.n_joints; i++) {
        if (armature.parent[i] < 0) {   // root joint
            TPB[i] = Sophus::SE3f(pose[i], armature.bone[i] + tran);
            TWB[i] = TPB[i];
        }
        else {   // non-root joint
            TPB[i] = Sophus::SE3f(pose[i], armature.bone[i]);
            TWB[i] = TWB[armature.parent[i]] * TPB[i];
        }
    }
}

void KinematicModel::get_state_R(std::vector<Eigen::Matrix3f> &pose, Eigen::Vector3f &tran) const
{
    pose.resize(armature.n_joints);
    for (int i = 0; i < armature.n_joints; i++) {
        if (armature.parent[i] < 0) {   // root joint
            pose[i] = TPB[i].rotationMatrix();
            tran = TPB[i].translation() - armature.bone[i];
        }
        else {   // non-root joint
            pose[i] = TPB[i].rotationMatrix();
        }
    }
}

void KinematicModel::get_state_q(std::vector<Eigen::Quaternionf> &pose, Eigen::Vector3f &tran) const
{
    pose.resize(armature.n_joints);
    for (int i = 0; i < armature.n_joints; i++) {
        if (armature.parent[i] < 0) {   // root joint
            pose[i] = TPB[i].unit_quaternion();
            tran = TPB[i].translation() - armature.bone[i];
        }
        else {   // non-root joint
            pose[i] = TPB[i].unit_quaternion();
        }
    }
}

void KinematicModel::update_state(const Eigen::VectorXf &delta)
{
    if (delta.size() != armature.n_joints * 3 + 3) {
        throw std::runtime_error("Error: Delta size mismatch");
    }

    for (int i = 0; i < armature.n_joints; i++) {
        TPB[i].so3() *= Sophus::SO3f::exp(delta.segment<3>(i * 3 + 3));
        if (armature.parent[i] < 0) {   // root joint
            TPB[i].translation() += delta.segment<3>(0);
            TWB[i] = TPB[i];
        }
        else {   // non-root joint
            TWB[i] = TWB[armature.parent[i]] * TPB[i];
        }
    }
}

Eigen::Vector3f KinematicModel::get_position(int joint_idx, const Eigen::Vector3f &local_position) const
{
    return TWB[joint_idx] * local_position;
}

Eigen::Matrix3f KinematicModel::get_orientation_R(int joint_idx, const Eigen::Matrix3f &local_orientation) const
{
    return TWB[joint_idx].rotationMatrix() * local_orientation;
}

Eigen::Quaternionf KinematicModel::get_orientation_q(int joint_idx, const Eigen::Quaternionf &local_orientation) const
{
    return TWB[joint_idx].unit_quaternion() * local_orientation;
}

Eigen::MatrixXf KinematicModel::get_position_Jacobian(int joint_idx, const Eigen::Vector3f &local_position) const
{
    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(3, armature.n_joints * 3 + 3);
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        const Eigen::Matrix3f R = TWB[i].rotationMatrix();
        const Eigen::Vector3f p = TWB[i].inverse() * (TWB[joint_idx] * local_position);
        J.block<3, 3>(0, i * 3 + 3) = -R * Sophus::SO3f::hat(p);
    }
    J.block<3, 3>(0, 0).setIdentity();
    return J;
}

Eigen::MatrixXf KinematicModel::get_orientation_Jacobian_R(int joint_idx, const Eigen::Matrix3f &local_orientation) const
{
    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(9, armature.n_joints * 3 + 3);
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        const Eigen::Matrix3f Rl = TWB[i].rotationMatrix();
        const Eigen::Matrix3f Rr = TWB[i].rotationMatrix().transpose() * (TWB[joint_idx].rotationMatrix() * local_orientation);
        J.block<3, 3>(0, i * 3 + 3) = -Rl * Sophus::SO3f::hat(Rr.col(0));
        J.block<3, 3>(3, i * 3 + 3) = -Rl * Sophus::SO3f::hat(Rr.col(1));
        J.block<3, 3>(6, i * 3 + 3) = -Rl * Sophus::SO3f::hat(Rr.col(2));
    }
    return J;
}

Eigen::MatrixXf KinematicModel::get_orientation_Jacobian_q(int joint_idx, const Eigen::Quaternionf &local_orientation) const
{
    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(4, armature.n_joints * 3 + 3);
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        const Eigen::Quaternionf ql = TWB[i].unit_quaternion();
        const Eigen::Quaternionf qr = TWB[i].unit_quaternion().conjugate() * (TWB[joint_idx].unit_quaternion() * local_orientation);
        J.block<4, 3>(0, i * 3 + 3) = (0.5 * left_quaternion(ql) * right_quaternion(qr)).rightCols(3);
    }
    return J;
}

