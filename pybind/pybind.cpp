#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "utils.h"
#include "../kinematics/kinematic_armature.h"
#include "../kinematics/kinematic_model.h"
#include "../kinematics/kinematic_optimizer.h"
#include "../dynamics/dynamic_armature.h"
#include "../dynamics/dynamic_model.h"

namespace py = pybind11;

PYBIND11_MODULE(carticulate, m) {
    m.doc() = "articulate uitls in C++";

    py::class_<KinematicArmature>(m, "KinematicArmature")
        .def(py::init<>(), "initialize an empty kinematic armature")
        .def(py::init<const std::string &>(), "initialize a kinematic armature from file", py::arg("armature_file"))
        .def("print", &KinematicArmature::print, "print armature information")
        .def_readwrite("n_joints", &KinematicArmature::n_joints, "number of joints")
        .def_readwrite("name", &KinematicArmature::name, "armature name")
        .def_readwrite("parent", &KinematicArmature::parent, "parent joint index (must satisfying parent[i] < i)")
        .def_property("bone", [](const KinematicArmature &armature) { return vV3f_to_MXf(armature.bone); }, [](KinematicArmature &armature, const Eigen::MatrixXf &bone) { armature.bone = MXf_to_vV3f(bone); }, "joint local position expressed in the parent frame");

    py::class_<KinematicModel>(m, "KinematicModel")
        .def(py::init<const std::string &>(), "initialize a kinematic model from armature file", py::arg("armature_file"))
        .def(py::init<const KinematicArmature &>(), "initialize a kinematic model from armature", py::arg("armature"))
        .def("print", &KinematicModel::print, "print model information")
        .def("get_armature", py::overload_cast<>(&KinematicModel::get_armature), "get the armature", py::return_value_policy::reference)
        .def("set_state_R", [](KinematicModel &model, const TensorXf &pose, const Eigen::Vector3f &tran) { model.set_state_R(TXf_to_vM3f(pose), tran); }, "set the pose and translation (rotation matrix)", py::arg("pose"), py::arg("tran"))
        .def("set_state_q", [](KinematicModel &model, const Eigen::MatrixXf &pose, const Eigen::Vector3f &tran) { model.set_state_q(MXf_to_vQf(pose), tran); }, "set the pose and translation (quaternion)", py::arg("pose"), py::arg("tran"))
        .def("get_state_R", [](const KinematicModel &model) { std::vector<Eigen::Matrix3f> pose; Eigen::Vector3f tran; model.get_state_R(pose, tran); return std::make_pair(vM3f_to_TXf(pose), tran); }, "get the pose and translation (rotation matrix)")
        .def("get_state_q", [](const KinematicModel &model) { std::vector<Eigen::Quaternionf> pose; Eigen::Vector3f tran; model.get_state_q(pose, tran); return std::make_pair(vQf_to_MXf(pose), tran); }, "get the pose and translation (quaternion)")
        .def("update_state", &KinematicModel::update_state, "update the pose and translation (right pertubation, translation first)", py::arg("delta"))
        .def("get_position", &KinematicModel::get_position, "get position in the world frame", py::arg("joint_idx"), py::arg("local_position") = Eigen::Vector3f::Zero())
        .def("get_orientation_R", &KinematicModel::get_orientation_R, "get orientation in the world frame (rotation matrix)", py::arg("joint_idx"), py::arg("local_orientation") = Eigen::Matrix3f::Identity())
        .def("get_orientation_q", [](const KinematicModel &model, int joint_idx, const Eigen::Vector4f &local_orientation) { return Qf_to_V4f(model.get_orientation_q(joint_idx, V4f_to_Qf(local_orientation))); }, "get orientation in the world frame (quaternion)", py::arg("joint_idx"), py::arg("local_orientation") = Eigen::Vector4f::UnitX())
        .def("get_position_Jacobian", &KinematicModel::get_position_Jacobian, "get position Jacobian: p(state + delta) = p(state) + J * delta", py::arg("joint_idx"), py::arg("local_position") = Eigen::Vector3f::Zero())
        .def("get_orientation_Jacobian_R", &KinematicModel::get_orientation_Jacobian_R, "get orientation (rotation matrix) Jacobian: R(state + delta) = R(state) + J * delta. R is flatten to 9x1 by concatenating three column vectors.", py::arg("joint_idx"), py::arg("local_orientation") = Eigen::Matrix3f::Identity())
        .def("get_orientation_Jacobian_q", [](const KinematicModel &model, int joint_idx, const Eigen::Vector4f &local_orientation) { return model.get_orientation_Jacobian_q(joint_idx, V4f_to_Qf(local_orientation)); }, "get orientation (quaternion) Jacobian: q(state + delta) = q(state) + J * delta. q is 4x1 in (w, x, y, z) order in the Jacobian.", py::arg("joint_idx"), py::arg("local_orientation") = Eigen::Vector4f::UnitX());

    py::enum_<RobustKernel>(m, "RobustKernel")
        .value("NONE", RobustKernel::NONE)
        .value("HUBER", RobustKernel::HUBER)
        .value("CAUCHY", RobustKernel::CAUCHY)
        .value("TUKEY", RobustKernel::TUKEY);

    py::class_<Observation> observation(m, "Observation");

    py::class_<Position3DObservation>(m, "Position3DObservation", observation)
        .def(py::init([](int j, const Eigen::Vector3f &p, const Eigen::Vector3f &obs, RobustKernel rk, float t, float w) { return new Position3DObservation(j, p, obs, rk, t, w); }), "initialize a position 3D observation", py::arg("joint_idx"), py::arg("local_position"), py::arg("observation"), py::arg("robust_kernel") = RobustKernel::NONE, py::arg("robust_kernel_delta") = 1, py::arg("weight") = 1)
        .def("print", &Position3DObservation::print, "print observation information");

    py::class_<Position2DObservation>(m, "Position2DObservation", observation)
        .def(py::init([](int j, const Eigen::Vector3f &p, const Eigen::Vector2f &obs, const Eigen::Matrix<float, 3, 4> &KT, RobustKernel rk, float t, float w) { return new Position2DObservation(j, p, obs, KT, rk, t, w); }), "initialize a position 2D observation", py::arg("joint_idx"), py::arg("local_position"), py::arg("observation"), py::arg("KT"), py::arg("robust_kernel") = RobustKernel::NONE, py::arg("robust_kernel_delta") = 1, py::arg("weight") = 1)
        .def("print", &Position2DObservation::print, "print observation information");
    
    py::class_<OrientationObservation>(m, "OrientationObservation", observation)
        .def(py::init([](int j, const Eigen::Vector4f &q, const Eigen::Vector4f &obs, RobustKernel rk, float t, float w) { return new OrientationObservation(j, V4f_to_Qf(q), V4f_to_Qf(obs), rk, t, w); }), "initialize an orientation observation", py::arg("joint_idx"), py::arg("local_orientation"), py::arg("observation"), py::arg("robust_kernel") = RobustKernel::NONE, py::arg("robust_kernel_delta") = 1, py::arg("weight") = 1)
        .def("print", &OrientationObservation::print, "print observation information");

    py::class_<KinematicOptimizer>(m, "KinematicOptimizer")
        .def(py::init<KinematicModel &, bool, bool>(), "initialize a kinematic optimizer  (please set manage_observations to false as they are managed by pybind)", py::arg("model"), py::arg("manage_observations") = false, py::arg("verbose") = true)
        .def("get_model", py::overload_cast<>(&KinematicOptimizer::get_model), "get the kinematic model")
        .def("print", &KinematicOptimizer::print, "print observation information")
        .def("clear_observations", &KinematicOptimizer::clear_observations, "clear all observations")
        .def("add_observation", &KinematicOptimizer::add_observation, "add observation", py::arg("obs"), py::keep_alive<1, 2>())
        .def("set_constraints", &KinematicOptimizer::set_constraints, "set constraints", py::arg("optimize_pose"), py::arg("optimize_tran"))
        .def("optimize", &KinematicOptimizer::optimize, "optimize the state of the kinematic model", py::arg("iterations"), py::arg("init_lambda") = -1);

    py::class_<DynamicArmature>(m, "DynamicArmature")
        .def(py::init<>(), "initialize an empty dynamic armature")
        .def(py::init<const std::string &>(), "initialize a dynamic armature from file", py::arg("armature_file"))
        .def("print", &DynamicArmature::print, "print armature information")
        .def_readwrite("n_joints", &DynamicArmature::n_joints, "number of joints")
        .def_readwrite("name", &DynamicArmature::name, "armature name")
        .def_readwrite("parent", &DynamicArmature::parent, "parent joint index (must satisfying parent[i] < i)")
        .def_property("bone", [](const DynamicArmature &armature) { return vV3f_to_MXf(armature.bone); }, [](DynamicArmature &armature, const Eigen::MatrixXf &bone) { armature.bone = MXf_to_vV3f(bone); }, "joint local position expressed in the parent frame")
        .def_readwrite("mass", &DynamicArmature::mass, "body mass")
        .def_property("com", [](const DynamicArmature &armature) { return vV3f_to_MXf(armature.com); }, [](DynamicArmature &armature, const Eigen::MatrixXf &com) { armature.com = MXf_to_vV3f(com); }, "body center of mass in the joint frame")
#ifndef USE_DIAGONAL_INERTIA
        .def_property("inertia", [](const DynamicArmature &armature) { return vM3f_to_TXf(armature.inertia); }, [](DynamicArmature &armature, const TensorXf &inertia) { armature.inertia = TXf_to_vM3f(inertia); }, "body inertia in the joint frame (Ixx, Iyy, Izz, Ixy, Iyz, Ixz)")
#else
        .def_property("inertia", [](const DynamicArmature &armature) { return vV3f_to_MXf(armature.inertia); }, [](DynamicArmature &armature, const Eigen::MatrixXf &inertia) { armature.inertia = MXf_to_vV3f(inertia); }, "body inertia in the joint frame (Ixx, Iyy, Izz)")
#endif
        .def_readwrite("gravity", &DynamicArmature::gravity, "gravitional acceleration in the world frame");

    py::class_<DynamicModel> pyDynamicModel(m, "DynamicModel");
    py::class_<DynamicModel::ExternalForce> pyExternalForce(pyDynamicModel, "ExternalForce");
    py::class_<DynamicModel::ExternalTorque> pyExternalTorque(pyDynamicModel, "ExternalTorque");

    pyDynamicModel
        .def(py::init<const std::string &>(), "initialize a dynamic model from armature file", py::arg("armature_file"))
        .def(py::init<const DynamicArmature &>(), "initialize a dynamic model from armature", py::arg("armature"))
        .def("print", &DynamicModel::print, "print model information")
        .def("get_armature", py::overload_cast<>(&DynamicModel::get_armature), "get the armature", py::return_value_policy::reference)
        .def("set_state_R", [](DynamicModel &model, const TensorXf &pose, const Eigen::Vector3f &tran, const Eigen::VectorXf &vel) { model.set_state_R(TXf_to_vM3f(pose), tran, vel); }, "set the pose, translation, and velocity (rotation matrix)", py::arg("pose"), py::arg("tran"), py::arg("vel"))
        .def("set_state_q", [](DynamicModel &model, const Eigen::MatrixXf &pose, const Eigen::Vector3f &tran, const Eigen::VectorXf &vel) { model.set_state_q(MXf_to_vQf(pose), tran, vel); }, "set the pose, translation, and velocity (quaternion)", py::arg("pose"), py::arg("tran"), py::arg("vel"))
        .def("get_state_R", [](const DynamicModel &model) { std::vector<Eigen::Matrix3f> pose; Eigen::Vector3f tran; Eigen::VectorXf vel; model.get_state_R(pose, tran, vel); return std::make_tuple(vM3f_to_TXf(pose), tran, vel); }, "get the pose, translation, and velocity (rotation matrix)")
        .def("get_state_q", [](const DynamicModel &model) { std::vector<Eigen::Quaternionf> pose; Eigen::Vector3f tran; Eigen::VectorXf vel; model.get_state_q(pose, tran, vel); return std::make_tuple(vQf_to_MXf(pose), tran, vel); }, "get the pose, translation, and velocity (quaternion)")
        .def("update_state", &DynamicModel::update_state, "update the pose, translation, and velocity by acceleration", py::arg("acc"), py::arg("delta_t"))
        .def("get_position", &DynamicModel::get_position, "get position in the world frame", py::arg("joint_idx"), py::arg("local_position") = Eigen::Vector3f::Zero())
        .def("get_orientation_R", &DynamicModel::get_orientation_R, "get orientation in the world frame (rotation matrix)", py::arg("joint_idx"), py::arg("local_orientation") = Eigen::Matrix3f::Identity())
        .def("get_orientation_q", [](const DynamicModel &model, int joint_idx, const Eigen::Vector4f &local_orientation) { return Qf_to_V4f(model.get_orientation_q(joint_idx, V4f_to_Qf(local_orientation))); }, "get orientation in the world frame (quaternion)", py::arg("joint_idx"), py::arg("local_orientation") = Eigen::Vector4f::UnitX())
        .def("get_linear_velocity", &DynamicModel::get_linear_velocity, "get linear velocity in the world frame", py::arg("joint_idx"), py::arg("local_position") = Eigen::Vector3f::Zero())
        .def("get_angular_velocity", &DynamicModel::get_angular_velocity, "get angular velocity in the world frame", py::arg("joint_idx"))
        .def("get_linear_Jacobian", &DynamicModel::get_linear_Jacobian, "get linear Jacobian: world-frame linear velocity = J * vel", py::arg("joint_idx"), py::arg("local_position") = Eigen::Vector3f::Zero())
        .def("get_angular_Jacobian", &DynamicModel::get_angular_Jacobian, "get angular Jacobian: world-frame angular velocity = J * vel", py::arg("joint_idx"))
        .def("get_linear_Jacobian_dot", &DynamicModel::get_linear_Jacobian_dot, "get the time derivate of linear Jacobian", py::arg("joint_idx"), py::arg("local_position") = Eigen::Vector3f::Zero())
        .def("get_angular_Jacobian_dot", &DynamicModel::get_angular_Jacobian_dot, "get the time derivate of angular Jacobian", py::arg("joint_idx"))
        .def("mass_matrix", &DynamicModel::mass_matrix, "compute mass matrix")
        .def("forward_dynamics", &DynamicModel::forward_dynamics, "compute acceleration given generalized force and external force & torque", py::arg("force"), py::arg("external_force") = std::vector<DynamicModel::ExternalForce>(), py::arg("external_torque") = std::vector<DynamicModel::ExternalTorque>())
        .def("inverse_dynamics", &DynamicModel::inverse_dynamics, "compute generalized force given acceleration and external force & torque", py::arg("acc"), py::arg("external_force") = std::vector<DynamicModel::ExternalForce>(), py::arg("external_torque") = std::vector<DynamicModel::ExternalTorque>());

    pyExternalForce
        .def(py::init<>(), "initialize an empty enternal force")
        .def(py::init<int, const Eigen::Vector3f &, const Eigen::Vector3f &>(), "initialize an enternal force", py::arg("joint_idx"), py::arg("force"), py::arg("local_position"))
        .def_readwrite("joint_idx", &DynamicModel::ExternalForce::joint_idx, "joint index")
        .def_readwrite("force", &DynamicModel::ExternalForce::force, "force applied to the joint in the world frame")
        .def_readwrite("local_position", &DynamicModel::ExternalForce::local_position, "position of the force in the joint frame");

    pyExternalTorque
        .def(py::init<>(), "initialize an empty enternal torque")
        .def(py::init<int, const Eigen::Vector3f &>(), "initialize an enternal torque", py::arg("joint_idx"), py::arg("torque"))
        .def_readwrite("joint_idx", &DynamicModel::ExternalTorque::joint_idx, "joint index")
        .def_readwrite("torque", &DynamicModel::ExternalTorque::torque, "torque applied to the joint in the world frame");
}
