#include <iostream>
#include "dynamic_model.h"
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

void DynamicModel::set_state_R(const std::vector<Eigen::Matrix3f> &pose, const Eigen::Vector3f &tran, const Eigen::VectorXf &vel)
{
    if (pose.size() != armature.n_joints || vel.size() != armature.n_joints * 3 + 3) {
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

    this->vel = vel;
}

void DynamicModel::set_state_q(const std::vector<Eigen::Quaternionf> &pose, const Eigen::Vector3f &tran, const Eigen::VectorXf &vel)
{
    if (pose.size() != armature.n_joints || vel.size() != armature.n_joints * 3 + 3) {
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

    this->vel = vel;
}

void DynamicModel::get_state_R(std::vector<Eigen::Matrix3f> &pose, Eigen::Vector3f &tran, Eigen::VectorXf &vel) const
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
    vel = this->vel;
}

void DynamicModel::get_state_q(std::vector<Eigen::Quaternionf> &pose, Eigen::Vector3f &tran, Eigen::VectorXf &vel) const
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
    vel = this->vel;
}

void DynamicModel::update_state(const Eigen::VectorXf &acc, float delta_t)
{
    if (acc.size() != armature.n_joints * 3 + 3) {
        throw std::runtime_error("Error: Acc size mismatch");
    }

    const Eigen::VectorXf delta = vel * delta_t + 0.5 * acc * delta_t * delta_t;
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

    vel += acc * delta_t;
}

Eigen::Vector3f DynamicModel::get_position(int joint_idx, const Eigen::Vector3f &local_position) const
{
    return TWB[joint_idx] * local_position;
}

Eigen::Matrix3f DynamicModel::get_orientation_R(int joint_idx, const Eigen::Matrix3f &local_orientation) const
{
    return TWB[joint_idx].rotationMatrix() * local_orientation;
}

Eigen::Quaternionf DynamicModel::get_orientation_q(int joint_idx, const Eigen::Quaternionf &local_orientation) const
{
    return TWB[joint_idx].unit_quaternion() * local_orientation;
}

Eigen::Vector3f DynamicModel::get_linear_velocity(int joint_idx, const Eigen::Vector3f &local_position) const
{
    Eigen::Vector3f v = vel.segment<3>(0);
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        const Eigen::Matrix3f R = TWB[i].rotationMatrix();
        const Eigen::Vector3f p = TWB[i].inverse() * (TWB[joint_idx] * local_position);
        v += R * vel.segment<3>(i * 3 + 3).cross(p);
    }
    return v;
}

Eigen::Vector3f DynamicModel::get_angular_velocity(int joint_idx) const
{
    Eigen::Vector3f w = Eigen::Vector3f::Zero();
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        w += TWB[i].rotationMatrix() * vel.segment<3>(i * 3 + 3);
    }
    return w;
}

Eigen::MatrixXf DynamicModel::get_linear_Jacobian(int joint_idx, const Eigen::Vector3f &local_position) const
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

Eigen::MatrixXf DynamicModel::get_angular_Jacobian(int joint_idx) const
{
    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(3, armature.n_joints * 3 + 3);
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        J.block<3, 3>(0, i * 3 + 3) = TWB[i].rotationMatrix();
    }
    return J;
}

Eigen::MatrixXf DynamicModel::get_linear_Jacobian_dot(int joint_idx, const Eigen::Vector3f &local_position) const
{
#if 0   // this is the slow version, but it is clear in logic

    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(3, armature.n_joints * 3 + 3);
    Eigen::Vector3f tWO = get_position(joint_idx, local_position);
    Eigen::Vector3f tWO_dot = get_linear_velocity(joint_idx, local_position);
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        Eigen::Vector3f tWB = get_position(i);
        Eigen::Vector3f tWB_dot = get_linear_velocity(i);
        Eigen::Matrix3f RWB = get_orientation_R(i);
        Eigen::Vector3f wWB = get_angular_velocity(i);
        J.block<3, 3>(0, i * 3 + 3) = -Sophus::SO3f::hat(tWO_dot - tWB_dot) * RWB - Sophus::SO3f::hat(tWO - tWB) * Sophus::SO3f::hat(wWB) * RWB;
    }
    return J;

#else   // this is faster

    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(3, armature.n_joints * 3 + 3);
    const Eigen::Vector3f tWO = TWB[joint_idx] * local_position;
    std::vector<Eigen::Vector3f> wWX(root_to_joint_chain[joint_idx].size());
    auto pwWB = wWX.begin();
    Eigen::Vector3f w = Eigen::Vector3f::Zero();
    for (int i : root_to_joint_chain[joint_idx]) {
        w += TWB[i].rotationMatrix() * vel.segment<3>(i * 3 + 3);
       *pwWB++ = w;
    }
    Eigen::Vector3f dt = Eigen::Vector3f::Zero();
    Eigen::Vector3f tBC = local_position;
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        const Eigen::Vector3f wWB = *--pwWB;
        const Eigen::Vector3f tWB = TWB[i].translation();
        const Eigen::Matrix3f RWB = TWB[i].rotationMatrix();
        dt += wWB.cross(RWB * tBC);
        tBC = TPB[i].translation();
        J.block<3, 3>(0, i * 3 + 3) = -(Sophus::SO3f::hat(dt) + Sophus::SO3f::hat(tWO - tWB) * Sophus::SO3f::hat(wWB)) * RWB;
    }
    return J;

#endif
}

Eigen::MatrixXf DynamicModel::get_angular_Jacobian_dot(int joint_idx) const
{
#if 0   // this is the slow version, but it is clear in logic

    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(3, armature.n_joints * 3 + 3);
    for (int i = joint_idx; i >= 0; i = armature.parent[i]) {
        Eigen::Matrix3f RWB = get_orientation_R(i);
        Eigen::Vector3f wWB = get_angular_velocity(i);
        J.block<3, 3>(0, i * 3 + 3) = Sophus::SO3f::hat(wWB) * RWB;
    }
    return J;

#else   // this is faster

    Eigen::MatrixXf J = Eigen::MatrixXf::Zero(3, armature.n_joints * 3 + 3);
    Eigen::Vector3f wWB = Eigen::Vector3f::Zero();
    for (int i : root_to_joint_chain[joint_idx]) {
        const Eigen::Matrix3f RWB = TWB[i].rotationMatrix();
        wWB += RWB * vel.segment<3>(i * 3 + 3);
        J.block<3, 3>(0, i * 3 + 3) = Sophus::SO3f::hat(wWB) * RWB;
	}
    return J;

#endif
}

Eigen::MatrixXf DynamicModel::mass_matrix() const
{
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(armature.n_joints * 3 + 3, armature.n_joints * 3 + 3);

    std::vector<float> m(armature.n_joints, 0);
    for (int i = armature.n_joints - 1; i >= 0; i--) {
        m[i] += armature.mass[i];
        M.block<3, 3>(i * 3 + 3, 0) += Sophus::SO3f::hat(armature.com[i]) * TWB[i].rotationMatrix().transpose() * armature.mass[i];
        if (armature.parent[i] >= 0) {
            m[armature.parent[i]] += m[i];
            M.block<3, 3>(armature.parent[i] * 3 + 3, 0) += TPB[i].rotationMatrix() * M.block<3, 3>(i * 3 + 3, 0) + Sophus::SO3f::hat(TPB[i].translation()) * TWB[armature.parent[i]].rotationMatrix().transpose() * m[i];
        }
        else {
            M(0, 0) = M(1, 1) = M(2, 2) = m[i];
        }
    }

    for (int j = 0; j < armature.n_joints; j++) {
        std::vector<Eigen::Matrix3f> fBB(armature.n_joints, Eigen::Matrix3f::Zero());
        for (int i : leaf_to_joint_chain[j]) {
            const Eigen::Matrix3f Rij = TWB[i].rotationMatrix().transpose() * TWB[j].rotationMatrix();
            const Eigen::Vector3f tij = TWB[i].rotationMatrix().transpose() * (TWB[j].translation() - TWB[i].translation());
#ifndef USE_DIAGONAL_INERTIA
            M.block<3, 3>(i * 3 + 3, j * 3 + 3) += armature.inertia[i] * Rij;
#else
            M.block<1, 3>(i * 3 + 3, j * 3 + 3) += Rij.row(0) * armature.inertia[i].x();
            M.block<1, 3>(i * 3 + 4, j * 3 + 3) += Rij.row(1) * armature.inertia[i].y();
            M.block<1, 3>(i * 3 + 5, j * 3 + 3) += Rij.row(2) * armature.inertia[i].z();
#endif
            const Eigen::Matrix3f fcom = armature.mass[i] * Sophus::SO3f::hat(tij - armature.com[i]) * Rij;
            M.block<3, 3>(i * 3 + 3, j * 3 + 3) += Sophus::SO3f::hat(armature.com[i]) * fcom;
            const Eigen::Matrix3f fPB = TPB[i].rotationMatrix() * (fBB[i] + fcom);
            if (armature.parent[i] >= 0) {
                fBB[armature.parent[i]] += fPB;
                M.block<3, 3>(armature.parent[i] * 3 + 3, j * 3 + 3) += TPB[i].rotationMatrix() * M.block<3, 3>(i * 3 + 3, j * 3 + 3) + Sophus::SO3f::hat(TPB[i].translation()) * fPB;
            }
            else {
                M.block<3, 3>(0, j * 3 + 3) = fPB;
            }
        }
        for (int i = armature.parent[j]; i >= 0; i = armature.parent[i]) {
            const Eigen::Matrix3f fPB = TPB[i].rotationMatrix() * fBB[i];
            if (armature.parent[i] >= 0) {
                fBB[armature.parent[i]] += fPB;
                M.block<3, 3>(armature.parent[i] * 3 + 3, j * 3 + 3) += TPB[i].rotationMatrix() * M.block<3, 3>(i * 3 + 3, j * 3 + 3) + Sophus::SO3f::hat(TPB[i].translation()) * fPB;
            }
            else {
                M.block<3, 3>(0, j * 3 + 3) = fPB;
            }
		}
    }
    return M;
}

Eigen::VectorXf DynamicModel::forward_dynamics(const Eigen::VectorXf &force, const std::vector<ExternalForce> &external_force, const std::vector<ExternalTorque> &external_torque) const
{
    Eigen::VectorXf h = inverse_dynamics(Eigen::VectorXf::Zero(armature.n_joints * 3 + 3), external_force, external_torque);
    Eigen::MatrixXf M = mass_matrix();
    Eigen::VectorXf acc = M.partialPivLu().solve(force - h);
    return acc;
}

Eigen::VectorXf DynamicModel::inverse_dynamics(const Eigen::VectorXf &acc, const std::vector<ExternalForce> &external_force, const std::vector<ExternalTorque> &external_torque) const
{
    // Recursive Newton-Euler Inverse Dynamics Algorithm
    // compute external force and torque
    Eigen::VectorXf tauBB = Eigen::VectorXf::Zero(armature.n_joints * 3 + 3);
    Eigen::VectorXf fBB = Eigen::VectorXf::Zero(armature.n_joints * 3 + 3);
    for (const auto &force: external_force) {
        const Eigen::Vector3f local_force = TWB[force.joint_idx].rotationMatrix().transpose() * force.force;
        tauBB.segment<3>(force.joint_idx * 3 + 3) -= force.local_position.cross(local_force);
        fBB.segment<3>(force.joint_idx * 3 + 3) -= local_force;
	}
    for (const auto &torque : external_torque) {
        tauBB.segment<3>(torque.joint_idx * 3 + 3) -= TWB[torque.joint_idx].rotationMatrix().transpose() * torque.torque;
	}

    // forward iteration that computes angular velocity and acceleration
    std::vector<Eigen::Vector3f> wBB(armature.n_joints);
    std::vector<Eigen::Vector3f> vBB(armature.n_joints);
    std::vector<Eigen::Vector3f> wBB_dot(armature.n_joints);
    std::vector<Eigen::Vector3f> vBB_dot(armature.n_joints);
    for (int i = 0; i < armature.n_joints; i++) {
        if (armature.parent[i] < 0) {   // root joint
            wBB[i] = vel.segment<3>(3);
            wBB_dot[i] = acc.segment<3>(3);
            vBB[i] = Eigen::Vector3f::Zero();                                                             // TWB[i].rotationMatrix().transpose() * vel.segment<3>(0);
            vBB_dot[i] = TWB[i].rotationMatrix().transpose() * (acc.segment<3>(0) - armature.gravity);    // TWB[i].rotationMatrix().transpose() * (acc.segment<3>(0) - armature.gravity) + vBB[i].cross(wBB[i]);
        }
        else {   // non-root joint
            wBB[i] = TPB[i].rotationMatrix().transpose() * wBB[armature.parent[i]] + vel.segment<3>(i * 3 + 3);
            wBB_dot[i] = TPB[i].rotationMatrix().transpose() * wBB_dot[armature.parent[i]] + wBB[i].cross(vel.segment<3>(i * 3 + 3)) + acc.segment<3>(i * 3 + 3);
            vBB[i] = TPB[i].rotationMatrix().transpose() * (vBB[armature.parent[i]] + wBB[armature.parent[i]].cross(TPB[i].translation()));
            vBB_dot[i] = TPB[i].rotationMatrix().transpose() * (vBB_dot[armature.parent[i]] + wBB_dot[armature.parent[i]].cross(TPB[i].translation())) + vBB[i].cross(vel.segment<3>(i * 3 + 3));
        }
    }

    // backward iteration that computes generalized force
    for (int i = armature.n_joints - 1; i >= 0; i--) {
#ifndef USE_DIAGONAL_INERTIA
        tauBB.segment<3>(i * 3 + 3) += armature.inertia[i] * wBB_dot[i] + wBB[i].cross(armature.inertia[i] * wBB[i]);
#else
        const Eigen::Vector3f a_(armature.inertia[i].z() - armature.inertia[i].y(), armature.inertia[i].x() - armature.inertia[i].z(), armature.inertia[i].y() - armature.inertia[i].x());
        const Eigen::Vector3f b_(wBB[i].z() * wBB[i].y(), wBB[i].x() * wBB[i].z(), wBB[i].y() * wBB[i].x());
        tauBB.segment<3>(i * 3 + 3) += armature.inertia[i].cwiseProduct(wBB_dot[i]) + a_.cwiseProduct(b_);
#endif
        const Eigen::Vector3f fcom = armature.mass[i] * (vBB_dot[i] + wBB[i].cross(vBB[i]) + wBB[i].cross(wBB[i].cross(armature.com[i])) + wBB_dot[i].cross(armature.com[i]));
        tauBB.segment<3>(i * 3 + 3) += armature.com[i].cross(fcom);
        fBB.segment<3>(i * 3 + 3) += fcom;
        const Eigen::Vector3f fPB = TPB[i].rotationMatrix() * fBB.segment<3>(i * 3 + 3);
        const Eigen::Vector3f tauPB = TPB[i].rotationMatrix() * tauBB.segment<3>(i * 3 + 3);
        if (armature.parent[i] >= 0) {
            fBB.segment<3>(armature.parent[i] * 3 + 3) += fPB;
            tauBB.segment<3>(armature.parent[i] * 3 + 3) += tauPB + TPB[i].translation().cross(fPB);
        }
        else {
            fBB.segment<3>(0) = tauBB.segment<3>(0) = fPB;
        }
    }

    return tauBB;
}

void DynamicModel::init()
{
    const int N = armature.n_joints; 
    TPB.resize(N); 
    TWB.resize(N);
    root_to_joint_chain.resize(N);
    leaf_to_joint_chain.resize(N);
	for (int i = N - 1; i >= 0; i--) {
		for (int j = i; j >= 0; j = armature.parent[j]) {
			leaf_to_joint_chain[j].push_back(i);
            root_to_joint_chain[i].push_back(j);
		}
        std::reverse(root_to_joint_chain[i].begin(), root_to_joint_chain[i].end());
	}
	vel = Eigen::VectorXf::Zero(N * 3 + 3);
}

#ifdef DEBUG
Eigen::MatrixXf DynamicModel::mass_matrix_slow()
{
    Eigen::MatrixXf M = Eigen::MatrixXf::Zero(armature.n_joints * 3 + 3, armature.n_joints * 3 + 3);
    Eigen::VectorXf vel = this->vel;
    Eigen::VectorXf gravity = this->armature.gravity;
    this->vel.setZero();
    this->armature.gravity.setZero();
    for (int i = 0; i < armature.n_joints * 3 + 3; i++) {
        M.col(i) = inverse_dynamics(Eigen::VectorXf::Unit(armature.n_joints * 3 + 3, i));
    }
    this->vel = vel;
    this->armature.gravity = gravity;
    return M;
}

Eigen::VectorXf DynamicModel::forward_dynamics_slow(const Eigen::VectorXf &force, const std::vector<ExternalForce> &external_force, const std::vector<ExternalTorque> &external_torque)
{
    Eigen::VectorXf h = inverse_dynamics(Eigen::VectorXf::Zero(armature.n_joints * 3 + 3), external_force, external_torque);
    Eigen::MatrixXf M = mass_matrix_slow();
    Eigen::VectorXf acc = M.partialPivLu().solve(force - h);
    return acc;
}
#endif