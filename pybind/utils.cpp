#include "utils.h"

Eigen::MatrixXf vV3f_to_MXf(const std::vector<Eigen::Vector3f> &data)
{
	Eigen::MatrixXf mat(data.size(), 3);
	for (int i = 0; i < data.size(); i++)
	{
		mat.row(i) = data[i];
	}
	return mat;
}

std::vector<Eigen::Vector3f> MXf_to_vV3f(const Eigen::MatrixXf &data)
{
	std::vector<Eigen::Vector3f> vec(data.rows());
	for (int i = 0; i < data.rows(); i++)
	{
		vec[i] = data.row(i);
	}
	return vec;
}

Eigen::Vector4f Qf_to_V4f(const Eigen::Quaternionf &data)
{
	return Eigen::Vector4f(data.w(), data.x(), data.y(), data.z());
}

Eigen::Quaternionf V4f_to_Qf(const Eigen::Vector4f &data)
{
	return Eigen::Quaternionf(data(0), data(1), data(2), data(3));
}

Eigen::MatrixXf vQf_to_MXf(const std::vector<Eigen::Quaternionf> &data)
{
	Eigen::MatrixXf mat(data.size(), 4);
	for (int i = 0; i < data.size(); i++)
	{
		mat.row(i) = Qf_to_V4f(data[i]);
	}
	return mat;
}

std::vector<Eigen::Quaternionf> MXf_to_vQf(const Eigen::MatrixXf &data)
{
	std::vector<Eigen::Quaternionf> vec(data.rows());
	for (int i = 0; i < data.rows(); i++)
	{
		vec[i] = V4f_to_Qf(data.row(i));
	}
	return vec;
}

TensorXf vM3f_to_TXf(const std::vector<Eigen::Matrix3f> &data)
{
	float *p = new float[data.size() * 9];
	for (int i = 0; i < data.size(); i++)
	{
		Eigen::Map<Eigen::Matrix3f>(p + i * 9) = data[i].transpose();
	}
	TensorXf r(
		std::vector<ssize_t>({ (ssize_t)data.size(), 3, 3 }),  // shape
		{ 3 * 3 * sizeof(float), 3 * sizeof(float), sizeof(float) },  // strides
		p   // data pointer
	);
	delete [] p;
	return r;
}

std::vector<Eigen::Matrix3f> TXf_to_vM3f(const TensorXf &data)
{
	auto buffer_info = data.request();
	float *pdata = static_cast<float *>(buffer_info.ptr);
	std::vector<Eigen::Matrix3f> vec(buffer_info.shape[0]);
	for (int i = 0; i < vec.size(); i++)
	{
		vec[i] = Eigen::Map<Eigen::Matrix3f>(pdata + i * 9).transpose();
	}
	return vec;
}

