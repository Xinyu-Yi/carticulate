#include "pch.h"
#include "CppUnitTest.h"
#include "../dynamics/dynamic_model.h"
#include <sstream>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace test_dynamics
{
    TEST_CLASS(test_dynamic_armature)
    {
    public:
        TEST_METHOD(test_smpl_male)
        {
            DynamicArmature armature("C:/Users/Admin/Work/projects/carticulate/armatures/SMPL_male.armature");
            Assert::AreEqual(armature.name, std::string("SMPL_male"));
            Assert::AreEqual(armature.n_joints, 24);
            Assert::AreEqual(armature.bone.size(), (size_t)24);
            Assert::AreEqual(armature.parent.size(), (size_t)24);
            Assert::AreEqual(armature.bone[0].norm(), 0.0f);
        }
    };

    TEST_CLASS(test_dynamic_model)
    {
    private:
        std::vector<Eigen::Quaternionf> pose;
        Eigen::Vector3f tran;
        Eigen::VectorXf vel;
        DynamicModel model;

        Eigen::VectorXf flatten(const std::vector<Eigen::Vector3f> &joint_torque, const Eigen::Vector3f &root_force) const {
            Eigen::VectorXf force(75);
            for (int i = 0; i < 24; i++) {
				force.segment<3>(3 * i + 3) = joint_torque[i];
			}
			force.segment<3>(0) = root_force;
			return force;
		}

    public:
        test_dynamic_model() : model("C:/Users/Admin/Work/projects/carticulate/armatures/SMPL_male.armature") {
            float pose_[24][4] = { 0.406466513872, -0.852810859680,  0.146360948682,  0.293388634920,
                                  0.227000191808, -0.091430991888, -0.126407116652, -0.961318075657,
                                  0.559360682964, -0.335615158081, -0.593093216419, -0.471930682659,
                                  0.520075201988, -0.525091350079,  0.615168750286,  0.274532765150,
                                  0.224615797400, -0.316543459892, -0.916824698448, -0.093704506755,
                                  0.730132281780, -0.406292289495, -0.539213001728,  0.105274818838,
                                  0.317469179630, -0.555329620838, -0.548433601856, -0.538556337357,
                                  0.880365848541,  0.048340465873, -0.085648462176,  0.463986545801,
                                  0.573147833347, -0.461526453495, -0.341062277555, -0.584954261780,
                                  0.627840995789,  0.718061864376, -0.075795009732, -0.290616571903,
                                  0.422042757273, -0.721613466740, -0.499400019646,  0.227494016290,
                                  0.358688920736, -0.041317056865, -0.261775255203, -0.895046889782,
                                  0.716889619827,  0.461497515440,  0.076190352440, -0.516995429993,
                                  0.023606089875, -0.517991423607,  0.271160930395,  0.810925006866,
                                  0.261237382889, -0.328141123056,  0.485250055790,  0.767209708691,
                                  0.874739468098,  0.451379001141,  0.044348072261, -0.170649155974,
                                  0.193883493543, -0.937609195709,  0.288603872061, -0.002450699219,
                                  0.343840390444, -0.663194835186, -0.649738967419, -0.140662044287,
                                  0.554252982140, -0.274869352579, -0.718913674355, -0.316881120205,
                                  0.201899036765, -0.273315548897, -0.397826313972, -0.852214515209,
                                  0.742405295372,  0.472843199968, -0.467474848032, -0.081981733441,
                                  0.444336771965, -0.544762492180, -0.625518560410,  0.338415622711,
                                  0.035868179053, -0.743403136730,  0.667864143848,  0.004765785299,
                                  0.587408542633,  0.357875138521,  0.599712193012,  0.408927708864 };
            for (int i = 0; i < 24; i++) {
                pose.emplace_back(pose_[i][0], pose_[i][1], pose_[i][2], pose_[i][3]);
            }
            tran << -1, 2, 3;
            vel = Eigen::VectorXf::Random(75);
        }
        TEST_METHOD(get_set_state)
        {
            model.set_state_q(pose, tran, vel);
            std::vector<Eigen::Matrix3f> pose_;
            Eigen::Vector3f tran_;
            Eigen::VectorXf vel_;
            model.get_state_R(pose_, tran_, vel_);
            for (int i = 0; i < 24; i++) {
				Assert::IsTrue(pose_[i].isApprox(pose[i].toRotationMatrix(), 1e-6));
			}
            Assert::IsTrue(tran_.isApprox(tran, 1e-6));
			Assert::IsTrue(vel_.isApprox(vel, 1e-6));

			model.set_state_R(pose_, tran, vel);
			std::vector<Eigen::Quaternionf> pose__;
			model.get_state_q(pose__, tran_, vel_);
            for (int i = 0; i < 24; i++) {
				Assert::IsTrue(pose__[i].angularDistance(pose[i]) < 1e-6);
			}
			Assert::IsTrue(tran_.isApprox(tran, 1e-6));
			Assert::IsTrue(vel_.isApprox(vel, 1e-6));
        }
        TEST_METHOD(update_state)
        {
            float delta_t = 1. / 60;
            Eigen::VectorXf acc_ = Eigen::VectorXf::Random(75);
            Eigen::VectorXf vel_ = vel + acc_ * delta_t;
            Eigen::Vector3f tran_ = tran + vel.segment<3>(0) * delta_t + 0.5 * acc_.segment<3>(0) * delta_t * delta_t;
            std::vector<Eigen::Matrix3f> pose_;
            for (int i = 0; i < 24; i++) {
				Sophus::SO3f R(pose[i]);
				Sophus::SO3f Rnew = R * Sophus::SO3f::exp(vel.segment<3>(3 + 3 * i) * delta_t + 0.5 * acc_.segment<3>(3 + 3 * i) * delta_t * delta_t);
				pose_.push_back(Rnew.matrix());
			}
            model.set_state_q(pose, tran, vel);
			model.update_state(acc_, delta_t);
			std::vector<Eigen::Matrix3f> pose__;
			Eigen::Vector3f tran__;
			Eigen::VectorXf vel__;
			model.get_state_R(pose__, tran__, vel__);
            for (int i = 0; i < 24; i++) {
                Assert::IsTrue(pose__[i].isApprox(pose_[i], 1e-6));
            }
            Assert::IsTrue(tran__.isApprox(tran_, 1e-6));
            Assert::IsTrue(vel__.isApprox(vel_, 1e-6));
        }
        TEST_METHOD(get_orientation_R)
        {
            std::vector<Eigen::Matrix3f> poseR;
            for (int i = 0; i < 24; i++) {
                poseR.push_back(pose[i].toRotationMatrix());
            }
            model.set_state_R(poseR, tran, vel);
            Eigen::Matrix3f R = model.get_orientation_R(22);
            Eigen::Matrix3f R_gt;
            R_gt << 0.934936225414, 0.333263248205, 0.121781542897, 0.343607902527, -0.764812767506, -0.544973790646, -0.088479414582, 0.551360368729, -0.829562544823;
            Assert::IsTrue(R.isApprox(R_gt, 1e-6));
        }
        TEST_METHOD(get_orientation_q)
        {
            model.set_state_q(pose, tran, vel);
            Eigen::Quaternionf q = model.get_orientation_q(11);
            Eigen::Quaternionf q_gt1(0.029300743714, -0.835265219212, -0.487387508154, -0.252837836742);
            Eigen::Quaternionf q_gt2(-0.029300743714, 0.835265219212, 0.487387508154, 0.252837836742);
            Assert::IsTrue(q.isApprox(q_gt1, 1e-6) || q.isApprox(q_gt2, 1e-6));
        }
        TEST_METHOD(get_position)
        {
            model.set_state_q(pose, tran, vel);

            Eigen::Vector3f p = model.get_position(15);
            Eigen::Vector3f pgt(-1.284297764301, 2 - 0.241089671850, 3.065410442650);
            Assert::IsTrue(p.isApprox(pgt, 1e-6));

            Eigen::Vector3f p2 = model.get_position(23, Eigen::Vector3f::Ones());
            Eigen::Vector3f pgt2(-2.610363841057, 2.549575448036, 3 - 0.479912221432);
            Assert::IsTrue(p2.isApprox(pgt2, 1e-6));
        }
        TEST_METHOD(get_linear_veolicty_jacobian)
        {
            for (int i = 0; i < 24; i++) {
                Eigen::VectorXf vel = Eigen::VectorXf::Random(75);
                model.set_state_q(pose, tran, vel);
                Eigen::Vector3f local_position = Eigen::Vector3f::Random();
                Eigen::MatrixXf J = model.get_linear_Jacobian(i, local_position);
                Eigen::Vector3f v = model.get_linear_velocity(i, local_position);
                Assert::IsTrue(v.isApprox(J * vel, 1e-5));
            }
        }
        TEST_METHOD(get_angular_veolicty_jacobian)
        {
            for (int i = 0; i < 24; i++) {
                Eigen::VectorXf vel = Eigen::VectorXf::Random(75);
                model.set_state_q(pose, tran, vel);
                Eigen::MatrixXf J = model.get_angular_Jacobian(i);
                Eigen::Vector3f w = model.get_angular_velocity(i);
                Assert::IsTrue(w.isApprox(J * vel, 1e-5));
            }
        }
        TEST_METHOD(get_linear_Jacobian_dot)
        {
            float dt = 1e-4;
            for (int i = 0; i < 24; i++) {
                const Eigen::VectorXf vel = Eigen::VectorXf::Random(75) * 10;
                const Eigen::VectorXf acc = Eigen::VectorXf::Random(75) * 10;
                const Eigen::Vector3f local_position = Eigen::Vector3f::Random();

                model.set_state_q(pose, tran, vel);
                const Eigen::Vector3f v0 = model.get_linear_velocity(i, local_position);

                model.update_state(acc, dt / 2);
                const Eigen::MatrixXf J = model.get_linear_Jacobian(i, local_position);
                const Eigen::MatrixXf Jdot = model.get_linear_Jacobian_dot(i, local_position);
                const Eigen::Vector3f acc_pred = Jdot * (vel + acc * dt / 2) + J * acc;

                model.update_state(acc, dt / 2);
                const Eigen::Vector3f v1 = model.get_linear_velocity(i, local_position);

                const Eigen::Vector3f acc_gt = (v1 - v0) / dt;

                Logger::WriteMessage(("Joint " + std::to_string(i) + " error " + std::to_string((acc_pred - acc_gt).norm() / acc_gt.norm()) + ":\n").c_str());
                Logger::WriteMessage(("\tGT-acc   " + std::to_string(acc_gt[0]) + ", " + std::to_string(acc_gt[1]) + "," + std::to_string(acc_gt[2]) + "\n").c_str());
                Logger::WriteMessage(("\tPred-acc " + std::to_string(acc_pred[0]) + ", " + std::to_string(acc_pred[1]) + "," + std::to_string(acc_pred[2]) + "\n").c_str());
                Assert::IsTrue(acc_gt.isApprox(acc_pred, 1e-2));
            }
        }
        TEST_METHOD(get_angular_Jacobian_dot)
        {
            float dt = 1e-4;
            for (int i = 0; i < 24; i++) {
                const Eigen::VectorXf vel = Eigen::VectorXf::Random(75) * 10;
                const Eigen::VectorXf acc = Eigen::VectorXf::Random(75) * 10;

                model.set_state_q(pose, tran, vel);
                const Eigen::Vector3f v0 = model.get_angular_velocity(i);

                model.update_state(acc, dt / 2);
                const Eigen::MatrixXf J = model.get_angular_Jacobian(i);
                const Eigen::MatrixXf Jdot = model.get_angular_Jacobian_dot(i);
                const Eigen::Vector3f acc_pred = Jdot * (vel + acc * dt / 2) + J * acc;

                model.update_state(acc, dt / 2);
                const Eigen::Vector3f v1 = model.get_angular_velocity(i);

                const Eigen::Vector3f acc_gt = (v1 - v0) / dt;

                Logger::WriteMessage(("Joint " + std::to_string(i) + " error " + std::to_string((acc_pred - acc_gt).norm() / acc_gt.norm()) + ":\n").c_str());
                Logger::WriteMessage(("\tGT-acc   " + std::to_string(acc_gt[0]) + ", " + std::to_string(acc_gt[1]) + "," + std::to_string(acc_gt[2]) + "\n").c_str());
                Logger::WriteMessage(("\tPred-acc " + std::to_string(acc_pred[0]) + ", " + std::to_string(acc_pred[1]) + "," + std::to_string(acc_pred[2]) + "\n").c_str());
                Assert::IsTrue(acc_gt.isApprox(acc_pred, 1e-2));
            }
        }
        TEST_METHOD(test_statics)
        {
            for (int i = 0; i < 24; i++) {
                Eigen::Vector3f f = Eigen::Vector3f::Random();
                Eigen::Vector3f local_position = Eigen::Vector3f::Random();
                model.set_state_q(pose, tran, vel);
                Eigen::MatrixXf J = model.get_linear_Jacobian(i, local_position);
                Eigen::VectorXf tau_pred = J.transpose() * f;

                Eigen::VectorXf tau_gt = Eigen::VectorXf::Zero(75);
                Eigen::Vector3f f_position = model.get_position(i, local_position);
                for (int j = i; j >= 0; j = model.get_armature().parent[j]) {
                    Eigen::Vector3f r = f_position - model.get_position(j);
                    tau_gt.segment<3>(j * 3 + 3) = model.get_orientation_R(j).transpose() * r.cross(f);
                }
                tau_gt.segment<3>(0) = f;

                Logger::WriteMessage(("Joint " + std::to_string(i) + " error " + std::to_string((tau_pred - tau_gt).norm() / tau_gt.norm()) + ":\n").c_str());
                //Logger::WriteMessage("\tGT-tau   ");
                //for (int j = 0; j < 75; j++) {
                //    Logger::WriteMessage((std::to_string(tau_gt[j]) + "\t").c_str());
                //}
                //Logger::WriteMessage("\n\tPred-tau ");
                //for (int j = 0; j < 75; j++) {
                //    Logger::WriteMessage((std::to_string(tau_pred[j]) + "\t").c_str());
                //}
                //Logger::WriteMessage("\n");
                Assert::IsTrue(tau_gt.isApprox(tau_pred, 1e-5));
            }
        }
        TEST_METHOD(test_dynamics)
        {
            std::vector<DynamicModel::ExternalForce> external_force;
            std::vector<DynamicModel::ExternalTorque> external_torque;
			for (int i = 0; i < 24; i++) {
                for (int j = 0; j < 2; j++) {
                    external_force.emplace_back(i, Eigen::Vector3f::Random(), Eigen::Vector3f::Random());
                    external_torque.emplace_back(i, Eigen::Vector3f::Random());
                }
			}
            for (int i = 0; i < 10; i++) {
                model.set_state_q(pose, tran, Eigen::VectorXf::Random(75));
                Eigen::VectorXf acc_gt = Eigen::VectorXf::Random(75);
                Eigen::VectorXf force_pred = model.inverse_dynamics(acc_gt, external_force, external_torque);
                Eigen::VectorXf acc_pred = model.forward_dynamics(force_pred, external_force, external_torque);
                Assert::IsTrue(acc_gt.isApprox(acc_pred, 1e-3));
            }
            for (int i = 0; i < 10; i++) {
                model.set_state_q(pose, tran, Eigen::VectorXf::Random(75));
                Eigen::VectorXf force_gt = Eigen::VectorXf::Random(75);
                Eigen::VectorXf acc_pred = model.forward_dynamics(force_gt, external_force, external_torque);
                Eigen::VectorXf force_pred = model.inverse_dynamics(acc_pred, external_force, external_torque);
                Assert::IsTrue(force_gt.isApprox(force_pred, 1e-3));
            }
        }
        TEST_METHOD(test_external_forces)
        {
            Eigen::VectorXf ext = Eigen::VectorXf::Zero(75);
            std::vector<DynamicModel::ExternalForce> external_force;
            std::vector<DynamicModel::ExternalTorque> external_torque;
            model.set_state_q(pose, tran, Eigen::VectorXf::Random(75));
            for (int i = 0; i < 24; i++) {
                for (int j = 0; j < 2; j++) {
                    const Eigen::Vector3f local_position = Eigen::Vector3f::Random();
                    const Eigen::Vector3f ext_f = Eigen::Vector3f::Random();
                    const Eigen::Vector3f ext_tau = Eigen::Vector3f::Random();
                    external_force.emplace_back(i, ext_f, local_position);
                    external_torque.emplace_back(i, ext_tau);
                    ext += model.get_linear_Jacobian(i, local_position).transpose() * ext_f;
                    ext += model.get_angular_Jacobian(i).transpose() * ext_tau;
                }
            }
            for (int i = 0; i < 10; i++) {
                model.set_state_q(pose, tran, Eigen::VectorXf::Random(75));
                Eigen::VectorXf acc_gt = Eigen::VectorXf::Random(75);
                Eigen::VectorXf force_pred1 = model.inverse_dynamics(acc_gt, external_force, external_torque);
                Eigen::VectorXf force_pred2 = model.inverse_dynamics(acc_gt) - ext;
                Assert::IsTrue(force_pred1.isApprox(force_pred2, 1e-4));
            }
            for (int i = 0; i < 10; i++) {
                model.set_state_q(pose, tran, Eigen::VectorXf::Random(75));
                Eigen::VectorXf force_gt = Eigen::VectorXf::Random(75);
                Eigen::VectorXf acc_pred1 = model.forward_dynamics(force_gt, external_force, external_torque);
                Eigen::VectorXf acc_pred2 = model.forward_dynamics(force_gt + ext);
                Assert::IsTrue(acc_pred1.isApprox(acc_pred2, 1e-4));
            }
        }
    };
}
