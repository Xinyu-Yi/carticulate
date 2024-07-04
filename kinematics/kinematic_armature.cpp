#include "kinematic_armature.h"
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>


KinematicArmature::KinematicArmature(const std::string &armature_file)
{
    n_joints = 0;
    std::ifstream file(armature_file);
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string field;
        std::string value;

        std::getline(iss, field, ':');
        std::getline(iss, value);

        if (field == "name") {
            value = value.substr(value.find_first_not_of(" "), value.find_last_not_of(" ") + 1);
            name = value;
        }
        else if (field == "n_joint") {
            n_joints = std::stoi(value);
        }
        else if (field == "parent") {
            if (n_joints == 0) {
                throw std::runtime_error("Error: <n_joint> should be defined before <parent>");
            }
            std::istringstream parent_iss(value);
            std::string parent_id;
            for (int i = 0; i < n_joints; i++) {
                std::getline(parent_iss, parent_id, ',');
                parent.push_back(std::stoi(parent_id));
                if (parent.back() >= i) {
                    throw std::runtime_error("Error: parent[" + std::to_string(i) + "] should be less than " + std::to_string(i));
                }
            }
        }
        else if (field == "bone") {
            if (n_joints == 0) {
                throw std::runtime_error("Error: <n_joint> should be defined before <bone>");
            }
            for (int i = 0; i < n_joints; i++) {
                std::getline(file, line);
                std::istringstream bone_iss(line);
                std::string x, y, z;
                std::getline(bone_iss, x, ',');
                std::getline(bone_iss, y, ',');
                std::getline(bone_iss, z, ',');
                bone.push_back(Eigen::Vector3f(std::stof(x), std::stof(y), std::stof(z)));
            }
        }
        else if (field == "gravity") {
            // not used in kinematics
        }
        else if (field == "com") {
            // not used in kinematics
            if (n_joints == 0) {
                throw std::runtime_error("Error: <n_joint> should be defined before <com>");
            }
            for (int i = 0; i < n_joints; i++) {
                std::getline(file, line);
            }
        }
        else if (field == "mass") {
            // not used in kinematics
        }
        else if (field == "inertia") {
            // not used in kinematics
            if (n_joints == 0) {
                throw std::runtime_error("Error: <n_joint> should be defined before <inertia>");
            }
            for (int i = 0; i < n_joints; i++) {
                std::getline(file, line);
            }
        }
        else {
            throw std::runtime_error("Error: Invalid field " + field);
        }
    }
    if (n_joints == 0) {
        throw std::runtime_error("Error: <n_joint> should be defined");
    }

    file.close();
}

void KinematicArmature::print() const
{
    std::cout << "========== KinematicArmature " << name << " Information ==========" << std::endl;
    std::cout << "n_joint: " << n_joints << std::endl;

    std::cout << "parent: ";
    for (int i = 0; i < n_joints; i++) {
        std::cout << parent[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "bone: " << std::endl;
    for (int i = 0; i < n_joints; i++) {
        std::cout << bone[i].transpose() << std::endl;
    }
}
