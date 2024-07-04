#pragma once

#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

using TensorXf = pybind11::array_t<float>;

Eigen::MatrixXf vV3f_to_MXf(const std::vector<Eigen::Vector3f> &data);
std::vector<Eigen::Vector3f> MXf_to_vV3f(const Eigen::MatrixXf &data);

Eigen::Vector4f Qf_to_V4f(const Eigen::Quaternionf &data);
Eigen::Quaternionf V4f_to_Qf(const Eigen::Vector4f &data);

Eigen::MatrixXf vQf_to_MXf(const std::vector<Eigen::Quaternionf> &data);
std::vector<Eigen::Quaternionf> MXf_to_vQf(const Eigen::MatrixXf &data);

TensorXf vM3f_to_TXf(const std::vector<Eigen::Matrix3f> &data);
std::vector<Eigen::Matrix3f> TXf_to_vM3f(const TensorXf &data);