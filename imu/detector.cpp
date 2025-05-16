#include "detector.h"

#ifndef FULL_ESKF_WITH_POS_VEL

#define WINDOW_SIZE             10
#define CONSTANT_G_NORM         9.8
#define CONSTANT_N_NORM         1

#define ENABLE_INITIALIZATION_CHECK
#define PRINT_INITIALIZATION_CHECK
#define INITIALIZATION_A_TH     0.2   // "|aS| - g" threshold
#define INITIALIZATION_M_TH     0.02  // "|mS| - n" threshold

#define ENABLE_GRAV_CORRECTION
//#define PRINT_GRAV_CORRECTION
#define GRAV_CORRECTION_A_TH    1     // "|aS| - g" threshold

#define ENABLE_MAGN_CORRECTION
//#define PRINT_MAGN_CORRECTION
#define MAGN_CORRECTION_A_TH    1     // "|aS| - g" threshold
#define MAGN_CORRECTION_M_TH    0.1   // "|mS| - n" threshold
#define MAGN_CORRECTION_D_TH    10    // "<aS, mS> - gnangle" threshold

#define ENABLE_BIAS_CORRECTION
//#define PRINT_BIAS_CORRECTION
#define BIAS_CORRECTION_A_TH    0.25  // "|aS - aS_recent|" threshold
#define BIAS_CORRECTION_W_TH    0.05  // "|wS|" threshold

#if defined(PRINT_INITIALIZATION_CHECK) || defined(PRINT_GRAV_CORRECTION) || defined(PRINT_MAGN_CORRECTION) || defined(PRINT_BIAS_CORRECTION)
#include <iostream>
#endif


void StateDetector::init(const Eigen::Vector3f &gI_, const Eigen::Vector3f &nI_)
{
    gnangle = angleBetween(gI_, nI_);
    am.clear();
    wm.clear();
    mm.clear();
}

void StateDetector::add(const Eigen::Vector3f &am_, const Eigen::Vector3f &wm_, const Eigen::Vector3f &mm_)
{
    am.push_back(am_);
    wm.push_back(wm_);
    mm.push_back(mm_);
    if (am.size() > WINDOW_SIZE) am.pop_front();
    if (wm.size() > WINDOW_SIZE) wm.pop_front();
    if (mm.size() > WINDOW_SIZE) mm.pop_front();
}

float StateDetector::initialization_confidence() const
{
#ifdef ENABLE_INITIALIZATION_CHECK
    if (am.size() < WINDOW_SIZE || mm.size() < WINDOW_SIZE) return 0;
    float aerror = 0, merror = 0;
    for (auto it_am = am.begin(), it_mm = mm.begin(); it_am != am.end() && it_mm != mm.end(); ++it_am, ++it_mm) {
        aerror += abs(it_am->norm() - CONSTANT_G_NORM);
        merror += abs(it_mm->norm() - CONSTANT_N_NORM);
    }
    aerror = aerror / WINDOW_SIZE;
    merror = merror / WINDOW_SIZE;
    bool succeed = aerror < INITIALIZATION_A_TH && merror < INITIALIZATION_M_TH;
    float confidence = 1 - (aerror / MAGN_CORRECTION_A_TH + merror / MAGN_CORRECTION_M_TH) / 2;
#ifdef PRINT_INITIALIZATION_CHECK
    std::cout << "[Initialization " << (succeed ? "Succeeded" : "Failed") << "] "
              << "Conf=" << (succeed ? confidence : 0) << " "
              << "|aS|-g=" << aerror << "(" << INITIALIZATION_A_TH << ") "
              << "|mS|-n=" << merror << "(" << INITIALIZATION_M_TH << ") " 
              << std::endl;
#endif
    return succeed ? confidence : 0;
#else
    return 1;
#endif
}

float StateDetector::gravity_correction_confidence() const
{
#ifdef ENABLE_GRAV_CORRECTION
    if (am.size() < WINDOW_SIZE) return 0;
    float aerror = 0;
    for (auto &am_ : am) {
        aerror += abs(am_.norm() - CONSTANT_G_NORM);
    }
    aerror = aerror / WINDOW_SIZE;
    bool succeed = aerror < GRAV_CORRECTION_A_TH;
    float confidence = 1 - aerror / GRAV_CORRECTION_A_TH;
#ifdef PRINT_GRAV_CORRECTION
    std::cout << "[Grav Correction " << (succeed ? "On" : "Off") << "] "
              << "Conf=" << (succeed ? confidence : 0) << " "
              << "|aS|-g=" << aerror << "(" << GRAV_CORRECTION_A_TH << ") "
              << std::endl;
#endif
    return succeed ? confidence : 0;
#else
    return 0;
#endif
}

float StateDetector::magnetic_correction_confidence() const
{
#ifdef ENABLE_MAGN_CORRECTION
    if (gnangle < 0 || am.size() < WINDOW_SIZE || mm.size() < WINDOW_SIZE) return 0;
    float aerror = 0, merror = 0, derror = 0;
    for (auto it_am = am.begin(), it_mm = mm.begin(); it_am != am.end() && it_mm != mm.end(); ++it_am, ++it_mm) {
        aerror += abs(it_am->norm() - CONSTANT_G_NORM);
        merror += abs(it_mm->norm() - CONSTANT_N_NORM);
        derror += abs(angleBetween(-*it_am, *it_mm) - gnangle) * 180.0f / 3.1416f;
    }
    aerror = aerror / WINDOW_SIZE;
    merror = merror / WINDOW_SIZE;
    derror = derror / WINDOW_SIZE;
    bool succeed = aerror < MAGN_CORRECTION_A_TH && merror < MAGN_CORRECTION_M_TH && derror < MAGN_CORRECTION_D_TH;
    float confidence = 1 - (aerror / MAGN_CORRECTION_A_TH + merror / MAGN_CORRECTION_M_TH + derror / MAGN_CORRECTION_D_TH) / 3;
#ifdef PRINT_MAGN_CORRECTION
    std::cout << "[Magn Correction " << (succeed ? "On" : "Off") << "] "
              << "Conf=" << (succeed ? confidence : 0) << " "
              << "|aS|-g=" << aerror << "(" << MAGN_CORRECTION_A_TH << ") "
              << "|mS|-n=" << merror << "(" << MAGN_CORRECTION_M_TH << ") "
              << "<-aS,mS>-<g,n>=" << derror << "(" << MAGN_CORRECTION_D_TH << ") "
              << std::endl;
#endif
    return succeed ? confidence : 0;
#else
    return 0;
#endif
}

float StateDetector::gyrobias_correction_confidence() const
{
#ifdef ENABLE_BIAS_CORRECTION
    if (am.size() < WINDOW_SIZE || wm.size() < WINDOW_SIZE) return 0;
    float aerror = 0, werror = 0;
    for (auto it_am = am.begin(), it_wm = wm.begin(); it_am != am.end() && it_wm != wm.end(); ++it_am, ++it_wm) {
        aerror += (*it_am - am.back()).norm();
        werror += it_wm->norm();
    }
    aerror = aerror / (WINDOW_SIZE - 1);
    werror = werror / WINDOW_SIZE;
    bool succeed = aerror < BIAS_CORRECTION_A_TH && werror < BIAS_CORRECTION_W_TH;
    float confidence = 1 - (aerror / BIAS_CORRECTION_A_TH + werror / BIAS_CORRECTION_W_TH) / 2;
#ifdef PRINT_BIAS_CORRECTION
    std::cout << "[Bias Correction " << (succeed ? "On" : "Off") << "] "
              << "Conf=" << (succeed ? confidence : 0) << " "
              << "|aS-aS_recent|=" << aerror << "(" << BIAS_CORRECTION_A_TH << ") "
              << "|wS|=" << werror << "(" << BIAS_CORRECTION_W_TH << ") "
              << std::endl;
#endif
    return succeed ? confidence : 0;
#else
    return 0;
#endif
}

#else

float SHOEDetector::score(const Eigen::Vector3f &am, const Eigen::Vector3f &wm)
{
    am_window.push_back(am);
    wm_window.push_back(wm);
    if (am_window.size() > window_size) {
        am_window.pop_front();
        wm_window.pop_front();
    }
    if (am_window.size() < window_size) {
        return 1e8;
    }
    const Eigen::Vector3f am_mean = get_mean_am().normalized();
    float error = 0;
    for (auto &am : am_window) {
        error += (am - g * am_mean).squaredNorm() / (an * an);
    }
    for (auto &wm : wm_window) {
        error += wm.squaredNorm() / (wn * wn);
    }
    error = error / window_size;

    float scale = 1e-3 / (wn * wn) + 1e-3 / (an * an);
    error /= scale;    // unify the scale
    return error;
}

Eigen::Vector3f SHOEDetector::get_mean_am() const
{
    Eigen::Vector3f am_mean(0, 0, 0);
    for (auto &am : am_window) {
        am_mean += am;
    }
    am_mean /= window_size;
    return am_mean;
}

Eigen::Vector3f SHOEDetector::get_mean_wm() const
{
    Eigen::Vector3f wm_mean(0, 0, 0);
    for (auto &wm : wm_window) {
        wm_mean += wm;
    }
    wm_mean /= window_size;
    return wm_mean;
}

float NormDetector::score(const Eigen::Vector3f &m)
{
    window.push_back(m);
    if (window.size() > window_size) {
        window.pop_front();
    }
    if (window.size() < window_size) {
        return 1e8;
    }
    float error = 0;
    for (auto &m : window) {
        error += abs(m.norm() - c);
    }
    error /= window_size;
    return error;
}

Eigen::Vector3f NormDetector::get_mean() const
{
    Eigen::Vector3f m_mean(0, 0, 0);
    for (auto &m : window) {
        m_mean += m;
    }
    m_mean /= window_size;
    return m_mean;
}

#endif
