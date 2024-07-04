#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include <iostream>
#include <sstream>

enum class RobustKernel { NONE, HUBER, CAUCHY, TUKEY };   // use a robust kernel to reduce the effect of outliers

struct Observation {
    RobustKernel robust_kernel;       // robust kernel
    float robust_kernel_delta;	      // robust kernel delta parameter (if robust kernel is used)
    float weight;                     // global weight of the observation
    virtual void print() const = 0;   // print observation information

protected:
    Observation(RobustKernel robust_kernel = RobustKernel::NONE, float robust_kernel_delta = 1, float weight = 1) :
        robust_kernel(robust_kernel), robust_kernel_delta(robust_kernel_delta), weight(weight) {}
    std::string loss_info() const {
        std::ostringstream oss;
        oss << "weight=" << weight << "\t robust_kernel=";
        switch (robust_kernel) {
            case RobustKernel::NONE:   oss << "None"; oss << "    "; break;
            case RobustKernel::HUBER:  oss << "Huber";  oss << "(" << robust_kernel_delta << ")"; break;
            case RobustKernel::CAUCHY: oss << "Cauchy"; oss << "(" << robust_kernel_delta << ")"; break;
            case RobustKernel::TUKEY:  oss << "Tukey";  oss << "(" << robust_kernel_delta << ")"; break;
        }
        return oss.str();
    }
};

struct Position3DObservation : public Observation {
    int joint_idx;                    // joint index
    Eigen::Vector3f local_position;   // keypoint local position on joint
    Eigen::Vector3f observation;      // keypoint 3D position
    virtual void print() const override {
        std::cout << "Position3DObservation \t " << loss_info() << "\t joint_idx=" << joint_idx << "\t local_position=(" << local_position.transpose() 
                  << ")\t observation=(" << observation.transpose() << ")\n";
    }

    Position3DObservation() = default;
    Position3DObservation(int joint_idx, const Eigen::Vector3f &local_position, const Eigen::Vector3f &observation, 
        RobustKernel robust_kernel = RobustKernel::NONE, float robust_kernel_delta = 1, float weight = 1)
        : Observation(robust_kernel, robust_kernel_delta, weight), joint_idx(joint_idx), local_position(local_position), observation(observation) {}
};

struct Position2DObservation : public Observation {
    int joint_idx;                         // joint index
    Eigen::Vector3f local_position;        // keypoint local position on joint
    Eigen::Vector2f observation;           // keypoint 2D position
    Eigen::Matrix<float, 3, 4> KT;         // camera matrix  KT(p, 1)^T -> (u, v, 1)^T
    virtual void print() const override {
        std::cout << "Position2DObservation \t " << loss_info() << "\t joint_idx=" << joint_idx << "\t local_position=(" << local_position.transpose() 
                  << ")\t observation=(" << observation.transpose() << ")\t KT=(" << KT(0, 0) << " ... " << KT(2, 3) << ")\n";
    }
    Position2DObservation() = default;
    Position2DObservation(int joint_idx, const Eigen::Vector3f &local_position, const Eigen::Vector2f &observation, const Eigen::Matrix<float, 3, 4> &KT,
        RobustKernel robust_kernel = RobustKernel::NONE, float robust_kernel_delta = 1, float weight = 1)
        : Observation(robust_kernel, robust_kernel_delta, weight), joint_idx(joint_idx), local_position(local_position), observation(observation), KT(KT) {}
};

struct OrientationObservation : public Observation {
    int joint_idx;                          // joint index
    Eigen::Quaternionf local_orientation;   // keypoint local orientation on joint
    Eigen::Quaternionf observation;         // keypoint orientation
    virtual void print() const override {
        std::cout << "OrientationObservation\t " << loss_info() << "\t joint_idx=" << joint_idx << "\t local_orientation=(" << local_orientation 
                  << ")\t observation=(" << observation << ")\n";
    }
    OrientationObservation() = default;
    OrientationObservation(int joint_idx, const Eigen::Quaternionf &local_orientation, const Eigen::Quaternionf &observation,
        RobustKernel robust_kernel = RobustKernel::NONE, float robust_kernel_delta = 1, float weight = 1)
        : Observation(robust_kernel, robust_kernel_delta, weight), joint_idx(joint_idx), local_orientation(local_orientation), observation(observation) {}
};