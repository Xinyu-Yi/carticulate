#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "sophus/so3.hpp"

namespace SO3 {


inline Eigen::Matrix3f hat(const Eigen::Vector3f &v) {
    return Sophus::SO3f::hat(v);
}

inline Eigen::Matrix3f Exp(const Eigen::Vector3f &v) {
    return Sophus::SO3f::exp(v).matrix();
}

inline Eigen::Vector3f Log(const Eigen::Matrix3f &R) {
    return Sophus::SO3f(R).log();
}

inline Eigen::Quaternionf Exp_q(const Eigen::Vector3f &v) {
    return Sophus::SO3f::exp(v).unit_quaternion();
}

inline Eigen::Vector3f Log_q(const Eigen::Quaternionf &q) {
    return Sophus::SO3f(q).log();
}

inline Eigen::MatrixXf dRadq(const Eigen::Quaternionf &q, const Eigen::Vector3f &a) {
    Eigen::Matrix<float, 3, 4> m;
    m.block<3, 1>(0, 0) = 2 * (q.w() * a + q.vec().cross(a));
    m.block<3, 3>(0, 1) = 2 * (q.vec().dot(a) * Eigen::Matrix3f::Identity() + q.vec() * a.transpose() - a * q.vec().transpose() - q.w() * hat(a));
    return m;
}

inline Eigen::Matrix3f dRaddtheta(const Eigen::Quaternionf &q, const Eigen::Vector3f &a) {
    Eigen::Matrix3f m = -q.toRotationMatrix() * hat(a);
    return m;
}


} // namespace SO3
