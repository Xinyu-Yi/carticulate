#pragma once

#include "dynamic_armature.h"
#include "sophus/se3.hpp"

class DynamicModel {
public:
    struct ExternalForce {
        int joint_idx;                     // joint index
		Eigen::Vector3f force;             // force applied to the joint in the world frame
		Eigen::Vector3f local_position;    // position of the force in the joint frame
        ExternalForce() : joint_idx(0) {}
        ExternalForce(int joint_idx, const Eigen::Vector3f &force, const Eigen::Vector3f &local_position) : joint_idx(joint_idx), force(force), local_position(local_position) {}
	};
    struct ExternalTorque {
        int joint_idx;                     // joint index
        Eigen::Vector3f torque;            // torque applied to the joint in the world frame
        ExternalTorque() : joint_idx(0) {}
		ExternalTorque(int joint_idx, const Eigen::Vector3f &torque) : joint_idx(joint_idx), torque(torque) {}
    };

    explicit DynamicModel(const std::string &armature_file) : armature(armature_file) { init(); }
    explicit DynamicModel(const DynamicArmature &armature) : armature(armature) { init(); }
    explicit DynamicModel(DynamicArmature &&armature) : armature(std::move(armature)) { init(); }

    void print() const { armature.print(); }                                // print model information
    DynamicArmature &get_armature() { return armature; }					// get the armature
    const DynamicArmature &get_armature() const { return armature; }		// get the armature
    
    // vel/acc (generalized velocity/acceleration) is the first/second-order time derivate of delta (as defined in KinematicModel)
    void set_state_R(const std::vector<Eigen::Matrix3f> &pose, const Eigen::Vector3f &tran, const Eigen::VectorXf &vel);
    void set_state_q(const std::vector<Eigen::Quaternionf> &pose, const Eigen::Vector3f &tran, const Eigen::VectorXf &vel);
    void get_state_R(std::vector<Eigen::Matrix3f> &pose, Eigen::Vector3f &tran, Eigen::VectorXf &vel) const;
    void get_state_q(std::vector<Eigen::Quaternionf> &pose, Eigen::Vector3f &tran, Eigen::VectorXf &vel) const;
    void update_state(const Eigen::VectorXf &acc, float delta_t);

    Eigen::Vector3f get_position(int joint_idx, const Eigen::Vector3f &local_position = Eigen::Vector3f::Zero()) const;                         // get position in the world frame
    Eigen::Matrix3f get_orientation_R(int joint_idx, const Eigen::Matrix3f &local_orientation = Eigen::Matrix3f::Identity()) const;             // get orientation in the world frame (rotation matrix)
    Eigen::Quaternionf get_orientation_q(int joint_idx, const Eigen::Quaternionf &local_orientation = Eigen::Quaternionf::Identity()) const;    // get orientation in the world frame (quaternion)
    Eigen::Vector3f get_linear_velocity(int joint_idx, const Eigen::Vector3f &local_position = Eigen::Vector3f::Zero()) const;                  // get linear velocity in the world frame
    Eigen::Vector3f get_angular_velocity(int joint_idx) const;                                                                                  // get angular velocity in the world frame
    Eigen::MatrixXf get_linear_Jacobian(int joint_idx, const Eigen::Vector3f &local_position = Eigen::Vector3f::Zero()) const;                  // get linear Jacobian: world-frame linear velocity = J * vel
    Eigen::MatrixXf get_angular_Jacobian(int joint_idx) const;                                                                                  // get angular Jacobian: world-frame angular velocity = J * vel
    Eigen::MatrixXf get_linear_Jacobian_dot(int joint_idx, const Eigen::Vector3f &local_position = Eigen::Vector3f::Zero()) const;              // get the time derivate of linear Jacobian
    Eigen::MatrixXf get_angular_Jacobian_dot(int joint_idx) const;                                                                              // get the time derivate of angular Jacobian

    Eigen::MatrixXf mass_matrix() const;                                                                                                                                                    // compute mass matrix
    Eigen::VectorXf forward_dynamics(const Eigen::VectorXf &force, const std::vector<ExternalForce> &external_force = {}, const std::vector<ExternalTorque> &external_torque = {}) const;   // compute acceleration given generalized force and external force & torque
    Eigen::VectorXf inverse_dynamics(const Eigen::VectorXf &acc, const std::vector<ExternalForce> &external_force = {}, const std::vector<ExternalTorque> &external_torque = {}) const;     // compute generalized force given acceleration and external force & torque

#ifdef DEBUG
    Eigen::MatrixXf mass_matrix_slow();                                                                                                                                                    // compute mass matrix
    Eigen::VectorXf forward_dynamics_slow(const Eigen::VectorXf &force, const std::vector<ExternalForce> &external_force = {}, const std::vector<ExternalTorque> &external_torque = {});   // compute acceleration given generalized force and external force & torque
#endif

private:
    void init();
    DynamicArmature armature;
    std::vector<std::vector<int>> root_to_joint_chain;  // root_to_joint_chain[i] contains the chain from root to joint i
    std::vector<std::vector<int>> leaf_to_joint_chain;  // leaf_to_joint_chain[i] contains the chain from leaf to joint i
    std::vector<Sophus::SE3f> TPB;                      // transformation from parent to bone
    std::vector<Sophus::SE3f> TWB;                      // transformation from world to bone
    Eigen::VectorXf vel;                                // generalized velocity, translation first
};