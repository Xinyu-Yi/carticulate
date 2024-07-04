#include "dynamic_armature.h"
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>


DynamicArmature::DynamicArmature(const std::string &armature_file)
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
			std::istringstream gravity_iss(value);
			std::string x, y, z;
			std::getline(gravity_iss, x, ',');
			std::getline(gravity_iss, y, ',');
			std::getline(gravity_iss, z, ',');
			gravity = Eigen::Vector3f(std::stof(x), std::stof(y), std::stof(z));
		}
        else if (field == "com") {
            if (n_joints == 0) {
                throw std::runtime_error("Error: <n_joint> should be defined before <com>");
            }
            for (int i = 0; i < n_joints; i++) {
                std::getline(file, line);
                std::istringstream com_iss(line);
                std::string x, y, z;
                std::getline(com_iss, x, ',');
                std::getline(com_iss, y, ',');
                std::getline(com_iss, z, ',');
                com.push_back(Eigen::Vector3f(std::stof(x), std::stof(y), std::stof(z)));
            }
        }
		else if (field == "mass") {
            if (n_joints == 0) {
                throw std::runtime_error("Error: <n_joint> should be defined before <mass>");
            }
			std::istringstream mass_iss(value);
			std::string mass_value;
            for (int i = 0; i < n_joints; i++) {
                std::getline(mass_iss, mass_value, ',');
                mass.push_back(std::stof(mass_value));
            }
		}
		else if (field == "inertia") {
            if (n_joints == 0) {
                throw std::runtime_error("Error: <n_joint> should be defined before <inertia>");
            }
            for (int i = 0; i < n_joints; i++) {
                std::getline(file, line);
                std::istringstream inertia_iss(line);
#ifndef USE_DIAGONAL_INERTIA
                std::string Ixx, Iyy, Izz, Ixy, Iyz, Ixz;
                std::getline(inertia_iss, Ixx, ',');
                std::getline(inertia_iss, Iyy, ',');
                std::getline(inertia_iss, Izz, ',');
                std::getline(inertia_iss, Ixy, ',');
                std::getline(inertia_iss, Iyz, ',');
                std::getline(inertia_iss, Ixz, ',');
                Eigen::Matrix3f I;
                I << std::stof(Ixx), std::stof(Ixy), std::stof(Ixz),
					 std::stof(Ixy), std::stof(Iyy), std::stof(Iyz),
					 std::stof(Ixz), std::stof(Iyz), std::stof(Izz);
#else
                std::string Ixx, Iyy, Izz;
                std::getline(inertia_iss, Ixx, ',');
                std::getline(inertia_iss, Iyy, ',');
                std::getline(inertia_iss, Izz, ',');
                Eigen::Vector3f I;
                I << std::stof(Ixx), std::stof(Iyy), std::stof(Izz);
#endif
                inertia.push_back(I);
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

void DynamicArmature::print() const
{
    std::cout << "========== DynamicArmature " << name << " Information ==========" << std::endl;
    std::cout << "n_joint: " << n_joints << std::endl;
    std::cout << "gravity: " << gravity.transpose() << std::endl;

    std::cout << "parent: ";
    for (int i = 0; i < n_joints; i++) {
        std::cout << parent[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "bone: " << std::endl;
    for (int i = 0; i < n_joints; i++) {
        std::cout << bone[i].transpose() << std::endl;
    }

    std::cout << "com: " << std::endl;
    for (int i = 0; i < n_joints; i++) {
        std::cout << com[i].transpose() << std::endl;
    }

    std::cout << "mass: ";
    for (int i = 0; i < n_joints; i++) {
        std::cout << mass[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "inertia: " << std::endl;
    for (int i = 0; i < n_joints; i++) {
#ifndef USE_DIAGONAL_INERTIA
        std::cout << inertia[i] << "\n----------------------------------" << std::endl;
#else
        std::cout << inertia[i].transpose() << std::endl;
#endif
    }
}
