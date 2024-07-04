#pragma once

#include "kinematic_model.h"
#include "observation.h"

class KinematicOptimizer {
public:
    // if manage_observations is true, the optimizer will manage the memory of observations (automatically delete each observation's pointer)
    explicit KinematicOptimizer(KinematicModel &model, bool manage_observations = false, bool verbose = true) : model(model), manage_observations(manage_observations), verbose(verbose)
    {
        optimize_pose.resize(model.get_armature().n_joints, true);
		optimize_tran = true;
    }
    ~KinematicOptimizer() { clear_observations(); }

    KinematicModel &get_model() { return model; }               // get the kinematic model
    const KinematicModel &get_model() const { return model; }   // get the kinematic model
    void print() const;                                         // print observation information
    void clear_observations();                                  // clear all observations
    void add_observation(const Observation *obs) { observations.push_back(obs); }       // add observation
    void set_constraints(const std::vector<bool> &optimize_pose, bool optimize_tran);   // set constraints
    void optimize(int iterations, float init_lambda = -1);      // optimize the state of the kinematic model

private:
    KinematicModel &model;
    std::vector<const Observation *> observations;
    std::vector<bool> optimize_pose;
    bool optimize_tran;
    bool manage_observations;
    bool verbose;
};