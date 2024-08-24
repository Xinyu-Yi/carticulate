#pragma once

#include "Eigen/Core"
#include <list>


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
