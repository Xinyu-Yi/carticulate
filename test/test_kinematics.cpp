#include "pch.h"
#include "CppUnitTest.h"
#include "../kinematics/kinematic_model.h"
#include "../kinematics/kinematic_optimizer.h"
#include <sstream>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace test_kinematics
{
	TEST_CLASS(test_kinematic_armature)
	{
	public:
		TEST_METHOD(test_smpl_male)
		{
			KinematicArmature armature("C:/Users/Admin/Work/projects/carticulate/armatures/SMPL_male.armature");
            Assert::AreEqual(armature.name, std::string("SMPL_male"));
            Assert::AreEqual(armature.n_joints, 24);
            Assert::AreEqual(armature.bone.size(), (size_t)24);
            Assert::AreEqual(armature.parent.size(), (size_t)24);
            Assert::AreEqual(armature.bone[0].norm(), 0.0f);
		}
        TEST_METHOD(test_smpl_female)
        {
            KinematicArmature armature("C:/Users/Admin/Work/projects/carticulate/armatures/SMPL_female.armature");
            Assert::AreEqual(armature.name, std::string("SMPL_female"));
            Assert::AreEqual(armature.n_joints, 24);
            Assert::AreEqual(armature.bone.size(), (size_t)24);
            Assert::AreEqual(armature.parent.size(), (size_t)24);
            Assert::AreEqual(armature.bone[0].norm(), 0.0f);
        }
        TEST_METHOD(test_mano_left)
        {
            KinematicArmature armature("C:/Users/Admin/Work/projects/carticulate/armatures/MANO_left.armature");
            Assert::AreEqual(armature.name, std::string("MANO_left"));
            Assert::AreEqual(armature.n_joints, 16);
            Assert::AreEqual(armature.bone.size(), (size_t)16);
            Assert::AreEqual(armature.parent.size(), (size_t)16);
            Assert::AreEqual(armature.bone[0].norm(), 0.0f);
        }
        TEST_METHOD(test_mano_right)
        {
            KinematicArmature armature("C:/Users/Admin/Work/projects/carticulate/armatures/MANO_right.armature");
            Assert::AreEqual(armature.name, std::string("MANO_right"));
            Assert::AreEqual(armature.n_joints, 16);
            Assert::AreEqual(armature.bone.size(), (size_t)16);
            Assert::AreEqual(armature.parent.size(), (size_t)16);
            Assert::AreEqual(armature.bone[0].norm(), 0.0f);
        }
	};

    TEST_CLASS(test_kinematic_model)
    {
    private:
        std::vector<Eigen::Quaternionf> pose;
        Eigen::Vector3f tran;
        KinematicModel model;

    public:
        test_kinematic_model() : model("C:/Users/Admin/Work/projects/carticulate/armatures/SMPL_male.armature") {
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
        }
        TEST_METHOD(get_orientation_R)
        {
            std::vector<Eigen::Matrix3f> poseR;
            for (int i = 0; i < 24; i++) {
				poseR.push_back(pose[i].toRotationMatrix());
			}
            model.set_state_R(poseR, tran);
            Eigen::Matrix3f R = model.get_orientation_R(22);
            Eigen::Matrix3f R_gt;
            R_gt << 0.934936225414, 0.333263248205, 0.121781542897, 0.343607902527, -0.764812767506, -0.544973790646, -0.088479414582, 0.551360368729, -0.829562544823;
            Assert::IsTrue(R.isApprox(R_gt, 1e-6));
        }
        TEST_METHOD(get_orientation_q)
        {
            model.set_state_q(pose, tran);
            Eigen::Quaternionf q = model.get_orientation_q(11);
            Eigen::Quaternionf q_gt1(0.029300743714, -0.835265219212, -0.487387508154, -0.252837836742);
            Eigen::Quaternionf q_gt2(-0.029300743714, 0.835265219212, 0.487387508154, 0.252837836742);
            Assert::IsTrue(q.isApprox(q_gt1, 1e-6) || q.isApprox(q_gt2, 1e-6));
        }
        TEST_METHOD(get_position)
        {
            model.set_state_q(pose, tran);

            Eigen::Vector3f p = model.get_position(15);
            Eigen::Vector3f pgt(-1.284297764301, 2 - 0.241089671850, 3.065410442650);
            Assert::IsTrue(p.isApprox(pgt, 1e-6));

            Eigen::Vector3f p2 = model.get_position(23, Eigen::Vector3f::Ones());
            Eigen::Vector3f pgt2(-2.610363841057, 2.549575448036, 3 - 0.479912221432);
            Assert::IsTrue(p2.isApprox(pgt2, 1e-6));
        }
        TEST_METHOD(get_position_Jacobian)
        {
			model.set_state_q(pose, tran);
            for (int i = 0; i < 24; i++) {
                Eigen::Vector3f local_position = Eigen::Vector3f::Random();
                Eigen::VectorXf delta = Eigen::VectorXf::Random(75) / 1000;
                Eigen::MatrixXf J = model.get_position_Jacobian(i, local_position);
                Eigen::Vector3f p = model.get_position(i, local_position);
                Eigen::Vector3f pnew = p + J * delta;

                model.update_state(delta);
                Eigen::Vector3f pgt = model.get_position(i, local_position);

                Assert::IsTrue(pnew.isApprox(pgt, 1e-5));
                Assert::IsFalse(p.isApprox(pgt, 1e-5));
            }
        }
        TEST_METHOD(get_orientation_Jacobian_R)
        {
			model.set_state_q(pose, tran);
            for (int i = 0; i < 24; i++) {
				Eigen::Matrix3f local_orientation = Eigen::Quaternionf::UnitRandom().toRotationMatrix();
				Eigen::VectorXf delta = Eigen::VectorXf::Random(75) / 1000;
				Eigen::MatrixXf J = model.get_orientation_Jacobian_R(i, local_orientation);
				Eigen::Matrix3f R = model.get_orientation_R(i, local_orientation);
                Eigen::Matrix3f dR;
                Eigen::VectorXf dR_flatten = J * delta;
                dR << dR_flatten[0], dR_flatten[3], dR_flatten[6], dR_flatten[1], dR_flatten[4], dR_flatten[7], dR_flatten[2], dR_flatten[5], dR_flatten[8];
				Eigen::Matrix3f Rnew = R + dR;

				model.update_state(delta);
				Eigen::Matrix3f Rgt = model.get_orientation_R(i, local_orientation);

				Assert::IsTrue(Rnew.isApprox(Rgt, 1e-4));
				Assert::IsFalse(Rnew.isApprox(R, 1e-4));
			}
		}
        TEST_METHOD(get_orientation_Jacobian_q)
        {
            model.set_state_q(pose, tran);
            for (int i = 0; i < 24; i++) {
                Eigen::Quaternionf local_orientation = Eigen::Quaternionf::UnitRandom();
                Eigen::VectorXf delta = Eigen::VectorXf::Random(75) / 1000;
                Eigen::MatrixXf J = model.get_orientation_Jacobian_q(i, local_orientation);
                Eigen::Quaternionf q = model.get_orientation_q(i, local_orientation);
                Eigen::VectorXf dq = J * delta;
                Eigen::Quaternionf qnew(q.w() + dq[0], q.x() + dq[1], q.y() + dq[2], q.z() + dq[3]);
                
                model.update_state(delta);
                Eigen::Quaternionf qgt1 = model.get_orientation_q(i, local_orientation);
                Eigen::Quaternionf qgt2(-qgt1.w(), -qgt1.x(), -qgt1.y(), -qgt1.z());

                Assert::IsTrue(qnew.isApprox(qgt1, 1e-5) || qnew.isApprox(qgt2, 1e-5));
                Assert::IsFalse(qnew.isApprox(q, 1e-5));
            }
        }
    };

    TEST_CLASS(test_kinematic_optimizer)
    {
    private:
        std::vector<Eigen::Quaternionf> pose;
        Eigen::Vector3f tran;
        KinematicModel model;

    public:
        test_kinematic_optimizer() : model("C:/Users/Admin/Work/projects/carticulate/armatures/SMPL_male.armature") {
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
        }
        TEST_METHOD(test_get_model)
        {
            KinematicOptimizer optimizer(model);
            KinematicOptimizer optimizer2(optimizer);
            Assert::IsTrue(&optimizer.get_model() == &optimizer2.get_model());
            Assert::IsTrue(&optimizer.get_model() == &model);
        }
        TEST_METHOD(test_optimize_orientation_observation)
        {
            KinematicOptimizer optimizer(model, true, true);
            model.set_state_q(pose, tran);
            for (int i = 0; i < 24; i++) {
                Eigen::Quaternionf local_orientation = Eigen::Quaternionf::UnitRandom();
                Eigen::Quaternionf observation = model.get_orientation_q(i, local_orientation);
                optimizer.add_observation(new OrientationObservation(i, local_orientation, observation, RobustKernel::NONE, 1, 1));
            }

            //optimizer.print();
            model.set_state_q(std::vector<Eigen::Quaternionf>(24, Eigen::Quaternionf::Identity()), Eigen::Vector3f::Zero());
            optimizer.set_constraints(std::vector<bool>(24, true), true);
            optimizer.optimize(100);

            std::vector<Eigen::Quaternionf> pose_est;
            Eigen::Vector3f tran_est;
            model.get_state_q(pose_est, tran_est);
            for (int i = 0; i < 24; i++) {
                float err = pose_est[i].angularDistance(pose[i]);
                Logger::WriteMessage(("Joint " + std::to_string(i) + " error: " + std::to_string(err) + "\n").c_str());
                Assert::IsTrue(err < 1e-5);
            }
            Logger::WriteMessage(("GT Tran:   " + std::to_string(tran[0]) + ", " + std::to_string(tran[1]) + "," + std::to_string(tran[2]) + "\n").c_str());
            Logger::WriteMessage(("Pred Tran: " + std::to_string(tran_est[0]) + ", " + std::to_string(tran_est[1]) + "," + std::to_string(tran_est[2]) + "\n").c_str());
        }
        TEST_METHOD(test_optimize_position3D_observation)
        {
            KinematicOptimizer optimizer(model, true, true);
            model.set_state_q(pose, tran);
            for (int i = 0; i < 24; i++) {
                for (int j = 0; j < 6; j++) {
                    Eigen::Vector3f local_position = Eigen::Vector3f::Random();
                    Eigen::Vector3f observation = model.get_position(i, local_position);
                    optimizer.add_observation(new Position3DObservation(i, local_position, observation, RobustKernel::NONE, 1, 1));
                }
            }

            //optimizer.print();
            model.set_state_q(std::vector<Eigen::Quaternionf>(24, Eigen::Quaternionf::Identity()), Eigen::Vector3f::Zero());
            optimizer.set_constraints(std::vector<bool>(24, true), true);
            optimizer.optimize(100);

            std::vector<Eigen::Quaternionf> pose_est;
            Eigen::Vector3f tran_est;
            model.get_state_q(pose_est, tran_est);
            for (int i = 0; i < 24; i++) {
                float err = pose_est[i].angularDistance(pose[i]);
                Logger::WriteMessage(("Joint " + std::to_string(i) + " error: " + std::to_string(err) + "\n").c_str());
                Assert::IsTrue(err < 1e-5);
            }
            Assert::IsTrue((tran - tran_est).norm() < 1e-5);
            Logger::WriteMessage(("GT Tran:   " + std::to_string(tran[0]) + ", " + std::to_string(tran[1]) + "," + std::to_string(tran[2]) + "\n").c_str());
            Logger::WriteMessage(("Pred Tran: " + std::to_string(tran_est[0]) + ", " + std::to_string(tran_est[1]) + "," + std::to_string(tran_est[2]) + "\n").c_str());
        }
        TEST_METHOD(test_optimize_position2D_observation)
        {
            KinematicOptimizer optimizer(model, true, true);
            Eigen::Matrix<float, 3, 4> KT;
            KT << 100, 0, 100, -1000,
                  0, 100, 100, -1000,
                  0, 0, 1, -10;

            int try_ = 20;
            while (--try_ > 0) {
                model.set_state_q(pose, tran);
                optimizer.clear_observations();
                for (int i = 0; i < 24; i++) {
                    for (int j = 0; j < 10; j++) {
                        Eigen::Vector3f local_position = Eigen::Vector3f::Random();
                        Eigen::Vector3f p = model.get_position(i, local_position);
                        const Eigen::Vector3f x = KT * Eigen::Vector4f(p[0], p[1], p[2], 1);
                        const Eigen::Vector2f uv(x[0] / x[2], x[1] / x[2]);
                        optimizer.add_observation(new Position2DObservation(i, local_position, uv, KT, RobustKernel::NONE, 1, 1));
                    }
                }
                //optimizer.print();
                model.set_state_q(std::vector<Eigen::Quaternionf>(24, Eigen::Quaternionf::Identity()), Eigen::Vector3f::Zero());
                optimizer.set_constraints(std::vector<bool>(24, true), true);
                optimizer.optimize(100);

                std::vector<Eigen::Quaternionf> pose_est;
                Eigen::Vector3f tran_est;
                model.get_state_q(pose_est, tran_est);
                bool success = true;
                for (int i = 0; i < 24; i++) {
                    float err = pose_est[i].angularDistance(pose[i]);
                    Logger::WriteMessage(("Joint " + std::to_string(i) + " error: " + std::to_string(err) + "\n").c_str());
                    if (err > 1e-4) {
                        success = false;
						break;
                    }
                }
                Logger::WriteMessage(("GT Tran:   " + std::to_string(tran[0]) + ", " + std::to_string(tran[1]) + "," + std::to_string(tran[2]) + "\n").c_str());
                Logger::WriteMessage(("Pred Tran: " + std::to_string(tran_est[0]) + ", " + std::to_string(tran_est[1]) + "," + std::to_string(tran_est[2]) + "\n").c_str());
                if ((tran - tran_est).norm() > 1e-4) {
                    success = false;
                }
                if (success) {
					break;
				}
            }
            Assert::IsTrue(try_ > 0);
        }
    };
}
