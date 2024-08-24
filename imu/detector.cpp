#include "detector.h"


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
