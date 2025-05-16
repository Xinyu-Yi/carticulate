#pragma once

#include "Eigen/Core"
#include <list>

#ifndef FULL_ESKF_WITH_POS_VEL

class StateDetector {
public:
    StateDetector() = default;
    void init(const Eigen::Vector3f &gI_, const Eigen::Vector3f &nI_);
    void add(const Eigen::Vector3f &am_, const Eigen::Vector3f &wm_, const Eigen::Vector3f &mm_);
    float initialization_confidence() const;
    float gravity_correction_confidence() const;
    float magnetic_correction_confidence() const;
    float gyrobias_correction_confidence() const;

private:
    float angleBetween(const Eigen::Vector3f &a, const Eigen::Vector3f &b) const {
        return std::acos(std::max(-1.0f, std::min(1.0f, a.normalized().dot(b.normalized()))));
    }
    float gnangle = -1.0f;
    std::list<Eigen::Vector3f> am;
    std::list<Eigen::Vector3f> wm;
    std::list<Eigen::Vector3f> mm;
};

#else

class SHOEDetector {
public:
    SHOEDetector(int window_size_, float an_, float wn_, float g_ = 9.8)
        : an(an_), wn(wn_), window_size(window_size_), g(g_) {}
    float score(const Eigen::Vector3f &am, const Eigen::Vector3f &wm);
    Eigen::Vector3f get_mean_am() const;
    Eigen::Vector3f get_mean_wm() const;

private:
    int window_size;
    float an;
    float wn;
    float g;
    std::list<Eigen::Vector3f> am_window;
    std::list<Eigen::Vector3f> wm_window;
};


class NormDetector {
public:
    NormDetector(int window_size_, float c_ = 9.8) : window_size(window_size_), c(c_) {}
    float score(const Eigen::Vector3f &m);
    Eigen::Vector3f get_mean() const;

private:
    int window_size;
    float c;
    std::list<Eigen::Vector3f> window;
};

#endif
