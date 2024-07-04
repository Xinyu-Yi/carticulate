#include "kinematic_optimizer.h"
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <iostream>

#define VERTEX_DIM 75

static class MotionVertex : public g2o::BaseVertex<VERTEX_DIM, KinematicModel *>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MotionVertex() : optimize_pose({}), optimize_tran(true) {}
    MotionVertex(KinematicModel &model, const std::vector<bool> &optimize_pose, bool optimize_tran) : optimize_pose(optimize_pose), optimize_tran(optimize_tran) { setEstimate(&model); }
    virtual void setToOriginImpl() {}
    virtual void oplusImpl(const number_t *update)
    {
        Eigen::VectorX<number_t>::ConstMapType delta(update, VERTEX_DIM);
        _estimate->update_state(delta.cast<float>());
    }
    virtual bool read(std::istream &in) { return false; }
    virtual bool write(std::ostream &out) const { return false; }
    virtual void push()
    {
        _backup_pose.emplace_back();
        _backup_tran.emplace_back();
        _backup_pose.back().resize(_estimate->get_armature().n_joints);
        _estimate->get_state_q(_backup_pose.back(), _backup_tran.back());
    }
    virtual void pop() 
    { 
        _estimate->set_state_q(_backup_pose.back(), _backup_tran.back());
        _backup_pose.pop_back();
        _backup_tran.pop_back();
        updateCache(); 
    }
    virtual void discardTop() 
    { 
        _backup_pose.pop_back();
        _backup_tran.pop_back();
    }
    virtual int stackSize() const 
    {
        return _backup_pose.size(); 
    }
    Eigen::MatrixXf mask_Jacobian(Eigen::MatrixXf J) const
    {
        if (!optimize_tran) J.block(0, 0, J.rows(), 3).setZero();
        for (int i = 0; i < _estimate->get_armature().n_joints; i++) {
            if (!optimize_pose[i]) J.block(0, 3 + i * 3, J.rows(), 3).setZero();
        }
        return J;
	}
protected:
    const std::vector<bool> &optimize_pose;
    const bool &optimize_tran;
    std::vector<std::vector<Eigen::Quaternionf>> _backup_pose;
    std::vector<Eigen::Vector3f> _backup_tran;
};

static class Position3dEdge : public g2o::BaseUnaryEdge<3, Eigen::Vector3f, MotionVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Position3dEdge(const Position3DObservation *observation) : joint_idx(observation->joint_idx), local_position(observation->local_position) 
    {
        setMeasurement(observation->observation);
        setInformation(Eigen::Matrix3d::Identity() * observation->weight);
    }
    virtual void computeError()
    {
        const MotionVertex *v = static_cast<const MotionVertex *>(_vertices[0]);
        _error = (v->estimate()->get_position(joint_idx, local_position) - _measurement).cast<double>();
    }
    virtual void linearizeOplus()
    {
        const MotionVertex *v = static_cast<const MotionVertex *>(_vertices[0]);
        _jacobianOplusXi = v->mask_Jacobian(v->estimate()->get_position_Jacobian(joint_idx, local_position)).cast<double>();
	}
    virtual bool read(std::istream &in) { return false; }
    virtual bool write(std::ostream &out) const { return false; }
protected:
    const int joint_idx;
    const Eigen::Vector3f local_position;
};

static class Position2dEdge : public g2o::BaseUnaryEdge<2, Eigen::Vector2f, MotionVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Position2dEdge(const Position2DObservation *observation) : joint_idx(observation->joint_idx), local_position(observation->local_position), KT(observation->KT) 
    {
        setMeasurement(observation->observation);
        setInformation(Eigen::Matrix2d::Identity() * observation->weight);
    }
    virtual void computeError()
    {
        const MotionVertex *v = static_cast<const MotionVertex *>(_vertices[0]);
        const Eigen::Vector3f p = v->estimate()->get_position(joint_idx, local_position);
        const Eigen::Vector3f x = KT * Eigen::Vector4f(p[0], p[1], p[2], 1);
        const Eigen::Vector2f uv(x[0] / x[2], x[1] / x[2]);
        _error = (uv - _measurement).cast<double>();
    }
    virtual void linearizeOplus()
    {
        const MotionVertex *v = static_cast<const MotionVertex *>(_vertices[0]);
        const Eigen::Vector3f p = v->estimate()->get_position(joint_idx, local_position);
        const Eigen::Vector3f x = KT * Eigen::Vector4f(p[0], p[1], p[2], 1);
        const Eigen::Vector2f uv(x[0] / x[2], x[1] / x[2]);
        const Eigen::MatrixXf J = v->estimate()->get_position_Jacobian(joint_idx, local_position);
        const Eigen::Matrix<float, 2, 3> P = (KT.block<2, 3>(0, 0) - uv * KT.block<1, 3>(2, 0)) / x[2];   // camera projection jacobian
        _jacobianOplusXi = v->mask_Jacobian(P * J).cast<double>();
	}
    virtual bool read(std::istream &in) { return false; }
    virtual bool write(std::ostream &out) const { return false; }
protected:
    const int joint_idx;
    const Eigen::Vector3f local_position;
    const Eigen::Matrix<float, 3, 4> KT;
};

static class OrientationEdge : public g2o::BaseUnaryEdge<3, Eigen::Quaternionf, MotionVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    OrientationEdge(const OrientationObservation *observation) : joint_idx(observation->joint_idx), local_orientation(observation->local_orientation) 
    {
        setMeasurement(observation->observation);
        setInformation(Eigen::Matrix3d::Identity() * observation->weight);
    }
    virtual void computeError()
    {
        const MotionVertex *v = static_cast<const MotionVertex *>(_vertices[0]);
        const Eigen::Quaternionf q = v->estimate()->get_orientation_q(joint_idx, local_orientation);
        _error = (_measurement.conjugate() * q).vec().cast<double>();
    }
    virtual void linearizeOplus()
    {
        const MotionVertex *v = static_cast<const MotionVertex *>(_vertices[0]);
        const Eigen::Quaternionf q = _measurement.conjugate();
        Eigen::Matrix<float, 3, 4> qL_bottom_row3; 
        qL_bottom_row3 << q.x(), q.w(), -q.z(), q.y(),
                          q.y(), q.z(), q.w(), -q.x(),
                          q.z(), -q.y(), q.x(), q.w();
        _jacobianOplusXi = v->mask_Jacobian(qL_bottom_row3 * v->estimate()->get_orientation_Jacobian_q(joint_idx, local_orientation)).cast<double>();
    }
    virtual bool read(std::istream &in) { return false; }
    virtual bool write(std::ostream &out) const { return false; }
protected:
    const int joint_idx;
    const Eigen::Quaternionf local_orientation;
};

static inline g2o::OptimizableGraph::Edge *get_edge(const Observation *observation) {
    g2o::OptimizableGraph::Edge *edge = nullptr;
    if (dynamic_cast<const Position3DObservation *>(observation)) edge = new Position3dEdge((const Position3DObservation *)observation);
    if (dynamic_cast<const Position2DObservation *>(observation)) edge = new Position2dEdge((const Position2DObservation *)observation);
    if (dynamic_cast<const OrientationObservation *>(observation)) edge = new OrientationEdge((const OrientationObservation *)observation);

    g2o::RobustKernel *kernel = nullptr;
    switch (observation->robust_kernel) {
        case RobustKernel::NONE:   kernel = nullptr; break;
        case RobustKernel::HUBER:  kernel = new g2o::RobustKernelHuber; break;
        case RobustKernel::CAUCHY: kernel = new g2o::RobustKernelCauchy; break;
        case RobustKernel::TUKEY:  kernel = new g2o::RobustKernelTukey; break;
    }
    if (kernel) {
        kernel->setDelta(observation->robust_kernel_delta);
        edge->setRobustKernel(kernel);
    }
    return edge;
}

void KinematicOptimizer::print() const
{
    std::cout << "========== Motion Optimizer Information ==========\n"
              << "KinematicArmature: " << model.get_armature().name << " (" << model.get_armature().n_joints << " joints)\n"
              << "Observations (" << observations.size() << "): \n";
    for (const Observation *obs : observations) {
        obs->print();
    }
    std::cout << "Constraints: \n";
    std::cout << "  translation   \t " << (optimize_tran ? "optimizable" : "fixed") << "\n";
    for (int i = 0; i < optimize_pose.size(); i++)
		std::cout << "  pose (joint " << i << ")\t " << (optimize_pose[i] ? "optimizable" : "fixed") << "\n";
}

void KinematicOptimizer::clear_observations()
{
    if (manage_observations)
        for (const Observation *obs : observations)
            delete obs;
    observations.clear();
}

void KinematicOptimizer::set_constraints(const std::vector<bool> &optimize_pose, bool optimize_tran)
{
    if (optimize_pose.size() != this->optimize_pose.size()) {
        throw std::runtime_error("Error: Pose mask size mismatch");
    }
    this->optimize_pose = optimize_pose;
	this->optimize_tran = optimize_tran;
}

void KinematicOptimizer::optimize(int iterations, float init_lambda)
{
    // setup optimizer
    g2o::SparseOptimizer optimizer;
    auto linearSolver = std::make_unique<g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>>();
    auto solver_ptr = std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
    if (init_lambda > 0) solver->setUserLambdaInit(init_lambda);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(verbose); 

    // set vertices and edges
    MotionVertex *vertex = new MotionVertex(model, optimize_pose, optimize_tran);
    vertex->setId(0);
    optimizer.addVertex(vertex);
    for (const Observation *observation : observations) {
        g2o::OptimizableGraph::Edge *edge = get_edge(observation);
        edge->setVertex(0, vertex);
        optimizer.addEdge(edge);
    }

    // start optimization
    optimizer.initializeOptimization();
    optimizer.optimize(iterations);
}

