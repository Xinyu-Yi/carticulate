#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "utils.h"
#include "detector.h"


#ifndef FULL_ESKF_WITH_POS_VEL

struct ErrorState {
    constexpr static int DIM = 6;
    Eigen::Vector3f theta;
    Eigen::Vector3f wb;

    ErrorState() = default;
    ErrorState(const Eigen::Vector3f &theta_,
               const Eigen::Vector3f &wb_) :
        theta(theta_), wb(wb_) {}
    ErrorState(const Eigen::VectorXf &x) {
        theta = x.segment<3>(0);
        wb = x.segment<3>(3);
    }
    Eigen::VectorXf to_vector() const {
        Eigen::VectorXf x(DIM);
        x << theta, wb;
        return x;
    }
    void reset() {
        theta.setZero();
        wb.setZero();
    }
};


struct NominalState {
    constexpr static int DIM = 7;
    Eigen::Quaternionf q;
    Eigen::Vector3f wb;

    NominalState() = default;
    NominalState(const Eigen::Quaternionf &q_,
                 const Eigen::Vector3f &wb_) :
        q(q_), wb(wb_) {}
    NominalState(const Eigen::VectorXf &x) {
        q = Eigen::Quaternionf(x[0], x[1], x[2], x[3]);
        wb = x.segment<3>(4);
    }
    Eigen::VectorXf to_vector() const {
        Eigen::VectorXf x(DIM);
        x << q.w(), q.x(), q.y(), q.z(), wb;
        return x;
    }
    void correct(const ErrorState &e) {
        q = q * SO3::Exp_q(e.theta);
        wb += e.wb;
    }
    void update(const Eigen::Vector3f &wm, float dt) {
        q = q * SO3::Exp_q((wm - wb) * dt);
        wb = wb;
    }
    void reset() {
        q.setIdentity();
        wb.setZero();
    }
};


class ESKF_IMU {
public:
    ESKF_IMU(float an = 1e-3, float wn = 1e-4, float mn = 1e-3, float ww = 1e-9);   // accelerometer[a]/gyroscope[w]/magnetometer[m]'s measurement noise[n]/random walk[w] standard deviation

    bool initialize(const Eigen::Matrix3f &RIS, const Eigen::Vector3f &gI, const Eigen::Vector3f &nI);
    bool initialize(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, const Eigen::Vector3f &mm);
    void predict(const Eigen::Vector3f &wm, float dt);
    void correct(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, const Eigen::Vector3f &mm);

    const StateDetector &get_detector() const { return state_detector; }
    const NominalState &get_state() const { return nominal_state; }
    Eigen::MatrixXf get_P() const { return P; }      // get state covariance matrix
    Eigen::Vector3f get_gI() const { return gI; }    // get gravity vector
    Eigen::Vector3f get_nI() const { return nI; }    // get magnetic field vector
    bool is_initialized() const { return is_init; }  // check if initialized
    float get_wb_confidence() const;                 // get wb estimation confience indicating whether its variance is sufficiently small

private:
    Eigen::MatrixXf Fdx(const Eigen::Vector3f &wm, float dt) const;
    Eigen::MatrixXf Fi_Qi_FiT(float dt) const;
    Eigen::MatrixXf Hdx(const Eigen::Vector3f &am, const Eigen::Vector3f &mm, unsigned int observationFlag) const;
    Eigen::MatrixXf R(float cgrav, float cmagn, float cbias, unsigned int observationFlag) const;
    Eigen::VectorXf h(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, const Eigen::Vector3f &mm, unsigned int observationFlag) const;
    Eigen::VectorXf y(unsigned int observationFlag) const;
    Eigen::MatrixXf G() const;

private:
    const float an, wn, mn, ww;
    bool is_init;
    Eigen::Matrix<float, ErrorState::DIM, ErrorState::DIM> P;
    Eigen::Vector3f gI;
    Eigen::Vector3f nI;

    StateDetector state_detector;
    NominalState nominal_state;
    ErrorState error_state;
};

#else

struct ErrorState {
    constexpr static int DIM = 15;
    Eigen::Vector3f p;
    Eigen::Vector3f v;
    Eigen::Vector3f theta;
    Eigen::Vector3f ab;
    Eigen::Vector3f wb;

    ErrorState() = default;
    ErrorState(const Eigen::Vector3f &p_,
               const Eigen::Vector3f &v_,
               const Eigen::Vector3f &theta_,
               const Eigen::Vector3f &ab_,
               const Eigen::Vector3f &wb_) :
        p(p_), v(v_), theta(theta_), ab(ab_), wb(wb_) {}
    ErrorState(const Eigen::VectorXf &x) {
        p = x.segment<3>(0);
        v = x.segment<3>(3);
        theta = x.segment<3>(6);
        ab = x.segment<3>(9);
        wb = x.segment<3>(12);
    }
    Eigen::VectorXf to_vector() const {
        Eigen::VectorXf x(DIM);
        x << p, v, theta, ab, wb;
        return x;
    }
    void reset() {
        p.setZero();
        v.setZero();
        theta.setZero();
        ab.setZero();
        wb.setZero();
    }
};


struct NominalState {
    constexpr static int DIM = 16;
    Eigen::Vector3f p;
    Eigen::Vector3f v;
    Eigen::Quaternionf q;
    Eigen::Vector3f ab;
    Eigen::Vector3f wb;

    NominalState() = default;
    NominalState(const Eigen::Vector3f &p_,
                 const Eigen::Vector3f &v_,
                 const Eigen::Quaternionf &q_,
                 const Eigen::Vector3f &ab_,
                 const Eigen::Vector3f &wb_) :
        p(p_), v(v_), q(q_), ab(ab_), wb(wb_) {}
    NominalState(const Eigen::VectorXf &x) {
        p = x.segment<3>(0);
        v = x.segment<3>(3);
        q = Eigen::Quaternionf(x[6], x[7], x[8], x[9]);
        ab = x.segment<3>(10);
        wb = x.segment<3>(13);
    }
    Eigen::VectorXf to_vector() const {
        Eigen::VectorXf x(DIM);
        x << p, v, q.w(), q.x(), q.y(), q.z(), ab, wb;
        return x;
    }
    void correct(const ErrorState &e) {
        p += e.p;
        v += e.v;
        q = q * SO3::Exp_q(e.theta);
        ab += e.ab;
        wb += e.wb;
    }
    void update(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, const Eigen::Vector3f &g, float dt) {
        const Eigen::Vector3f a = q.toRotationMatrix() * (am - ab) + g;
        p += v * dt + 0.5 * a * dt * dt;
        v += a * dt;
        q = q * SO3::Exp_q((wm - wb) * dt);
        ab = ab;
        wb = wb;
    }
    void reset() {
        p.setZero();
        v.setZero();
        q.setIdentity();
        ab.setZero();
        wb.setZero();
    }
};


class ESKF_IMU {
public:
    ESKF_IMU(float an, float wn, float aw, float ww, float mn);   // accelerometer[a]/gyroscope[w]/magnetometer[m]'s measurement noise[n]/random walk[w] standard deviation

    // sensor 6/9-dof initialization: return succeed or not
    bool initialize_9dof(const Eigen::Matrix3f &RIS, const Eigen::Vector3f &gI, const Eigen::Vector3f &nI);
    bool initialize_9dof(const Eigen::Vector3f &am, const Eigen::Vector3f &mm);
    bool initialize_6dof(const Eigen::Matrix3f &RIS, const Eigen::Vector3f &gI);
    bool initialize_6dof(const Eigen::Vector3f &am);

    // state prediction and correction
    void predict(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, float dt);   // predict state
    Eigen::VectorXf correct(const Eigen::Vector3f &am = null_observation,   // accelerometer measurement
                            const Eigen::Vector3f &wm = null_observation,   // gyroscope measurement
                            const Eigen::Vector3f &mm = null_observation,   // magnetometer measurement
                            const Eigen::Vector3f &pm = null_observation,   // position measurement (global)
                            const Eigen::Vector3f &vm = null_observation,   // velocity measurement (global)
                            float pn = 1e-2, float vn = 1e-2                // measurement noise standard deviation
                            );   // return observation score for debug

    // get state
    NominalState get_state() const { return nominal_state; }   // get state estimation
    Eigen::MatrixXf get_P() const { return P; }                // get state covariance matrix
    Eigen::Vector3f get_gI() const { if (gI == null_observation) throw std::runtime_error("gravity vector is not initialized"); return gI; }        // get gravity vector
    Eigen::Vector3f get_nI() const { if (nI == null_observation) throw std::runtime_error("magnetic field vector is not initialized"); return nI; } // get magnetic field vector

    static const Eigen::Vector3f null_observation;

private:
    Eigen::VectorXf h(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, const Eigen::Vector3f &mm, const Eigen::Vector3f &pm, const Eigen::Vector3f &vm, const Eigen::Vector3f &gI, const Eigen::Vector3f &nI, unsigned int observationFlag) const;
    Eigen::VectorXf y(unsigned int observationFlag) const;
    Eigen::MatrixXf Fdx(const Eigen::Vector3f &am, const Eigen::Vector3f &wm, float dt) const;
    Eigen::MatrixXf Fi_Qi_FiT(float dt) const;
    Eigen::MatrixXf Hdx(const Eigen::Vector3f &am, const Eigen::Vector3f &mm, unsigned int observationFlag) const;
    Eigen::MatrixXf G() const;
    float score_to_sigma(float score01) const { return exp(2 * score01); }

private:
    Eigen::Matrix<float, ErrorState::DIM, ErrorState::DIM> P;
    float an, wn, aw, ww, mn;
    Eigen::Vector3f gI = null_observation;
    Eigen::Vector3f nI = null_observation;

    NormDetector acc_norm_detector;
    NormDetector gyr_norm_detector;
    NormDetector mag_norm_detector;

    NominalState nominal_state;
    ErrorState error_state;
};

#endif
