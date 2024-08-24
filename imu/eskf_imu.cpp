#include "eskf_imu.h"
#include <iostream>

#define TRUNCATION_ORDER     3    // discrete-time integration truncation order of exponential map

#define WINDOW_SIZE          6
#define CONSTANT_G_NORM      9.8
#define CONSTANT_W_NORM      0
#define CONSTANT_N_NORM      1

#define MAGNORM_THRESHOLD    0.2   // MagNorm
#define ACCNORM_THRESHOLD    2     // AccNorm
#define GYRNORM_THRESHOLD    0.05  // GyrNorm

#define OBSERVATION_MAGNORTH    0x0001
#define OBSERVATION_GRAVITY     0x0002
#define OBSERVATION_STATIONARY  0x0004
#define OBSERVATION_POSITION    0x0008
#define OBSERVATION_VELOCITY    0x0010
#define OBSERVATION_NOTUSE      0x0020


using namespace std;


const Eigen::Vector3f ESKF_IMU::null_observation(2023, 6, 10);


ESKF_IMU::ESKF_IMU(float an, float wn, float aw, float ww, float mn) :
    acc_norm_detector(WINDOW_SIZE, CONSTANT_G_NORM),
    gyr_norm_detector(WINDOW_SIZE, CONSTANT_W_NORM),
    mag_norm_detector(WINDOW_SIZE, CONSTANT_N_NORM),
    an(an), wn(wn), aw(aw), ww(ww), mn(mn) {}


Eigen::VectorXf ESKF_IMU::h(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, const Eigen::Vector3f &mm, const Eigen::Vector3f &pm, const Eigen::Vector3f &vm, const Eigen::Vector3f &gI, const Eigen::Vector3f &nI, unsigned int observationFlag) const
{
    int r = 0, n = count_observations(observationFlag);
    Eigen::VectorXf y = Eigen::VectorXf::Zero(n * 3);
    if (observationFlag & OBSERVATION_MAGNORTH)   { y.segment<3>(r) = nominal_state.q.toRotationMatrix() * mm - nI; r += 3; }
    if (observationFlag & OBSERVATION_GRAVITY)    { y.segment<3>(r) = nominal_state.q.toRotationMatrix() * (am - nominal_state.ab) + gI; r += 3; }
    if (observationFlag & OBSERVATION_STATIONARY) { y.segment<3>(r) = nominal_state.v;       r += 3; }
    if (observationFlag & OBSERVATION_POSITION)   { y.segment<3>(r) = nominal_state.p - pm;  r += 3; }
    if (observationFlag & OBSERVATION_VELOCITY)   { y.segment<3>(r) = nominal_state.v - vm;  r += 3; }
    return y;
}


Eigen::VectorXf ESKF_IMU::y(unsigned int observationFlag) const
{
    int n = count_observations(observationFlag);
    Eigen::VectorXf y = Eigen::VectorXf::Zero(n * 3);
    return y;
}


Eigen::MatrixXf ESKF_IMU::Fdx(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, float dt) const
{
    Eigen::MatrixXf Fdx = Eigen::MatrixXf::Identity(ErrorState::DIM, ErrorState::DIM);
    Fdx.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity() * dt;
    Fdx.block<3, 3>(3, 6) = -nominal_state.q.toRotationMatrix() * SO3::hat(am - nominal_state.ab) * dt;
    Fdx.block<3, 3>(3, 9) = -nominal_state.q.toRotationMatrix() * dt;
    Fdx.block<3, 3>(6, 6) = SO3::Exp((wm - nominal_state.wb) * dt).transpose();
    Fdx.block<3, 3>(6, 12) = -Eigen::Matrix3f::Identity() * dt;

#if TRUNCATION_ORDER >= 2
    Fdx.block<3, 3>(0, 6) = -0.5 * nominal_state.q.toRotationMatrix() * SO3::hat(am - nominal_state.ab) * dt * dt;
    Fdx.block<3, 3>(0, 9) = -0.5 * nominal_state.q.toRotationMatrix() * dt * dt;
    Fdx.block<3, 3>(3, 12) = 0.5 * nominal_state.q.toRotationMatrix() * SO3::hat(am - nominal_state.ab) * dt * dt;
#endif

#if TRUNCATION_ORDER >= 3
    Fdx.block<3, 3>(0, 12) = 1.0 / 6.0 * nominal_state.q.toRotationMatrix() * SO3::hat(am - nominal_state.ab) * dt * dt * dt;
#endif

    return Fdx;
}


Eigen::MatrixXf ESKF_IMU::Fi_Qi_FiT(float dt) const
{
    Eigen::MatrixXf Fi_Qi_FiT = Eigen::MatrixXf::Zero(ErrorState::DIM, ErrorState::DIM);
    Fi_Qi_FiT.block<3, 3>(3, 3) = Eigen::Matrix3f::Identity() * pow(an, 2) * dt * dt;
    Fi_Qi_FiT.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() * pow(wn, 2) * dt * dt;
    Fi_Qi_FiT.block<3, 3>(9, 9) = Eigen::Matrix3f::Identity() * pow(aw, 2) * dt;
    Fi_Qi_FiT.block<3, 3>(12, 12) = Eigen::Matrix3f::Identity() * pow(ww, 2) * dt;
    return Fi_Qi_FiT;
}


Eigen::MatrixXf ESKF_IMU::Hdx(const Eigen::Vector3f &am, const Eigen::Vector3f &mm, unsigned int observationFlag) const
{
    int r = 0, n = count_observations(observationFlag);
    Eigen::MatrixXf Hdx = Eigen::MatrixXf::Zero(n * 3, ErrorState::DIM);

    if (observationFlag & OBSERVATION_MAGNORTH) {
        Hdx.block<3, 3>(r, 6) = SO3::dRaddtheta(nominal_state.q, mm);
        r += 3;
    }
    if (observationFlag & OBSERVATION_GRAVITY) {
        Hdx.block<3, 3>(r, 6) = SO3::dRaddtheta(nominal_state.q, am - nominal_state.ab);
        Hdx.block<3, 3>(r, 9) = -nominal_state.q.toRotationMatrix();
        r += 3;
    }
    if (observationFlag & OBSERVATION_STATIONARY) {
        Hdx.block<3, 3>(r, 3) = Eigen::Matrix3f::Identity();
        r += 3;
    }
    if (observationFlag & OBSERVATION_POSITION) {
        Hdx.block<3, 3>(r, 0) = Eigen::Matrix3f::Identity();
        r += 3;
    }
    if (observationFlag & OBSERVATION_VELOCITY) {
        Hdx.block<3, 3>(r, 3) = Eigen::Matrix3f::Identity();
        r += 3;
    }
    return Hdx;
}


Eigen::MatrixXf ESKF_IMU::G() const
{
    Eigen::MatrixXf G = Eigen::MatrixXf::Identity(ErrorState::DIM, ErrorState::DIM);
    G.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() - SO3::hat(error_state.theta / 2);
    return G;
}


bool ESKF_IMU::initialize_9dof(const Eigen::Matrix3f &RIS, const Eigen::Vector3f &gI, const Eigen::Vector3f &nI)
{
    // initialize nominal state and error state
    nominal_state.reset();
    error_state.reset();
    nominal_state.q = Eigen::Quaternionf(RIS);

    // initialize global gravity and magnetic field
    this->gI = gI;
    this->nI = nI;

    // initialize covariance matrix P
    P = Eigen::MatrixXf::Zero(ErrorState::DIM, ErrorState::DIM);
    P.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();                        // Var(p)
    P.block<3, 3>(3, 3) = Eigen::Matrix3f::Zero();                            // Var(v)
    P.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() * 1e-3;                 // Var(theta)
    P.block<3, 3>(9, 9) = Eigen::Matrix3f::Identity() * 600 * pow(aw, 2);     // Var(ab)
    P.block<3, 3>(12, 12) = Eigen::Matrix3f::Identity() * 600 * pow(ww, 2);   // Var(wb)

    return true;
}


bool ESKF_IMU::initialize_9dof(const Eigen::Vector3f &am, const Eigen::Vector3f &mm)
{
    if (acc_norm_detector.score(am) > ACCNORM_THRESHOLD || mag_norm_detector.score(mm) > MAGNORM_THRESHOLD) return false;
    const Eigen::Vector3f a = acc_norm_detector.get_mean();
    const Eigen::Vector3f m = mag_norm_detector.get_mean();
    
    // compute sensor orientation in NED global frame
    Eigen::Matrix3f R;
    R.col(2) = -a.normalized();
    R.col(1) = R.col(2).cross(m).normalized();
    R.col(0) = R.col(1).cross(R.col(2)).normalized();
    R.transposeInPlace();

    return initialize_9dof(R, -R * a, R * m);
}


bool ESKF_IMU::initialize_6dof(const Eigen::Matrix3f &RIS, const Eigen::Vector3f &gI)
{
    // initialize nominal state and error state
    nominal_state.reset();
    error_state.reset();
    nominal_state.q = Eigen::Quaternionf(RIS);

    // initialize global gravity and magnetic field
    this->gI = gI;
    this->nI = null_observation;

    // initialize covariance matrix P
    P = Eigen::MatrixXf::Zero(ErrorState::DIM, ErrorState::DIM);
    P.block<3, 3>(0, 0) = Eigen::Matrix3f::Identity();                        // Var(p)
    P.block<3, 3>(3, 3) = Eigen::Matrix3f::Zero();                            // Var(v)
    P.block<3, 3>(6, 6) = Eigen::Matrix3f::Identity() * 1e-3;                 // Var(theta)
    P.block<3, 3>(9, 9) = Eigen::Matrix3f::Identity() * 600 * pow(aw, 2);     // Var(ab)
    P.block<3, 3>(12, 12) = Eigen::Matrix3f::Identity() * 600 * pow(ww, 2);   // Var(wb)

    return true;
}


bool ESKF_IMU::initialize_6dof(const Eigen::Vector3f &am)
{
    if (acc_norm_detector.score(am) > ACCNORM_THRESHOLD) return false;
    const Eigen::Vector3f a = acc_norm_detector.get_mean();

    // compute sensor orientation in NED global frame
    Eigen::Matrix3f R;
    R.col(2) = -a.normalized();
    if (R.col(2).isApprox(Eigen::Vector3f::UnitX())) {
        R.col(1) = R.col(2).cross(Eigen::Vector3f::UnitY()).normalized();
    }
    else {
        R.col(1) = R.col(2).cross(Eigen::Vector3f::UnitX()).normalized();
    }
    R.col(0) = R.col(1).cross(R.col(2)).normalized();
    R.transposeInPlace();

    return initialize_6dof(R, -R * a);
}


void ESKF_IMU::predict(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, float dt)
{
    if (gI == null_observation) throw runtime_error("ESKF_IMU::predict: gravity is not initialized");
    const Eigen::MatrixXf _Fdx = Fdx(am, wm, dt);
    const Eigen::MatrixXf _Fi_Qi_FiT = Fi_Qi_FiT(dt);
    nominal_state.update(am, wm, gI, dt);
    // error_state = ErrorState(_Fdx * error_state.to_vector());  // always zero
    P = _Fdx * P * _Fdx.transpose() + _Fi_Qi_FiT;
}


Eigen::VectorXf ESKF_IMU::correct(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, const Eigen::Vector3f &mm, const Eigen::Vector3f &pm, const Eigen::Vector3f &vm, float pn, float vn)
{
    if (gI == null_observation && am != null_observation) throw runtime_error("ESKF_IMU::correct: gravity is not initialized");
    if (nI == null_observation && mm != null_observation) throw runtime_error("ESKF_IMU::correct: magnetic field is not initialized");

    // check observations
    unsigned int observation_flag = 0U;
    const float sm = mm != null_observation ? mag_norm_detector.score(mm) / MAGNORM_THRESHOLD : 1e8;                      // mag is good
    const float sa = am != null_observation ? acc_norm_detector.score(am - nominal_state.ab) / ACCNORM_THRESHOLD : 1e8;   // acc is gravity
    const float sw = wm != null_observation ? gyr_norm_detector.score(wm - nominal_state.wb) / GYRNORM_THRESHOLD : 1e8;   // sensor is stationary
    const float sp = pm != null_observation ? 0 : 1e8;       // has position observation
    const float sv = vm != null_observation ? 0 : 1e8;       // has velocity observation

    if (sm < 1) observation_flag |= OBSERVATION_MAGNORTH;
    if (sa < 1) observation_flag |= OBSERVATION_GRAVITY;
    if (sw < 1) observation_flag |= OBSERVATION_STATIONARY;
    if (sp < 1) observation_flag |= OBSERVATION_POSITION;
    if (sv < 1) observation_flag |= OBSERVATION_VELOCITY;

    if (observation_flag > 0) {
        // calculate observation noise R
        int r = 0, n = count_observations(observation_flag);
        Eigen::MatrixXf R = Eigen::MatrixXf::Zero(n * 3, n * 3);
        if (observation_flag & OBSERVATION_MAGNORTH)   { R.block<3, 3>(r, r) = score_to_sigma(sm) * pow(mn,   2) * Eigen::Matrix3f::Identity(); r += 3; }
        if (observation_flag & OBSERVATION_GRAVITY)    { R.block<3, 3>(r, r) = score_to_sigma(sa) * pow(an,   2) * Eigen::Matrix3f::Identity(); r += 3; }
        if (observation_flag & OBSERVATION_STATIONARY) { R.block<3, 3>(r, r) = score_to_sigma(sw) * pow(1e-2, 2) * Eigen::Matrix3f::Identity(); r += 3; }
        if (observation_flag & OBSERVATION_POSITION)   { R.block<3, 3>(r, r) = score_to_sigma(sp) * pow(pn,   2) * Eigen::Matrix3f::Identity(); r += 3; }
        if (observation_flag & OBSERVATION_VELOCITY)   { R.block<3, 3>(r, r) = score_to_sigma(sv) * pow(vn,   2) * Eigen::Matrix3f::Identity(); r += 3; }

        // update error state by observation
        const Eigen::VectorXf _y = y(observation_flag);
        const Eigen::MatrixXf _Hdx = Hdx(am, mm, observation_flag);
        const Eigen::VectorXf _h = h(am, wm, mm, pm, vm, gI, nI, observation_flag);
        const Eigen::MatrixXf _K = P * _Hdx.transpose() * (_Hdx * P * _Hdx.transpose() + R).inverse();
        const Eigen::MatrixXf _I = Eigen::MatrixXf::Identity(ErrorState::DIM, ErrorState::DIM);
        error_state = ErrorState(_K * (_y - _h));
        // P = (_I - _K * _Hdx) * P;   // poor numerical stability
        P = (_I - _K * _Hdx) * P * (_I - _K * _Hdx).transpose() + _K * R * _K.transpose();  // Joseph form of covariance update

        // correct nominal state and reset error state
        const Eigen::MatrixXf _G = G();
        nominal_state.correct(error_state);
        P = _G * P * _G.transpose();
        error_state.reset();
    }

    // for debug
    Eigen::VectorXf score01debug(5);
    score01debug << sm, sa, sw, sp, sv;
    return score01debug;
}
