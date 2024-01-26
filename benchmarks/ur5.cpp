#include <iostream>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/parsers/srdf.hpp>

#include "mim_solvers/csqp.hpp"
#include "mim_solvers/sqp.hpp"

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/multibody/fwd.hpp"

#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"

#include "crocoddyl/multibody/residuals/control-gravity.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"

#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "crocoddyl/core/integrator/euler.hpp"


#include "crocoddyl/core/constraints/residual.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"

#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"

#include "timings.hpp"



int main(){

    // LOADING THE ROBOT AND INIT VARIABLES

    auto urdf_path = EXAMPLE_ROBOT_DATA_MODEL_DIR "/ur_description/urdf/ur5_robot.urdf";

    boost::shared_ptr<pinocchio::Model> rmodel = boost::make_shared<pinocchio::Model>();
    pinocchio::urdf::buildModel(urdf_path, *rmodel.get());
    const int nq = rmodel->nq;
    const int nv = rmodel->nv;
    const int nu = nv;
    Eigen::VectorXd x0; x0.resize(nq + nu); x0.setZero();

    // STATE AND ACTUATION VARIABLES

    boost::shared_ptr<crocoddyl::StateMultibody> state = boost::make_shared<crocoddyl::StateMultibody>(rmodel);
    boost::shared_ptr<crocoddyl::ActuationModelFull> actuation = boost::make_shared<crocoddyl::ActuationModelFull>(state);
     
    boost::shared_ptr<crocoddyl::ResidualModelControlGrav> uResidual = boost::make_shared<crocoddyl::ResidualModelControlGrav>(state); 
    boost::shared_ptr<crocoddyl::CostModelResidual> uRegCost = boost::make_shared<crocoddyl::CostModelResidual>(state, uResidual); 
    
    boost::shared_ptr<crocoddyl::ResidualModelState> xResidual = boost::make_shared<crocoddyl::ResidualModelState>(state, x0);
    boost::shared_ptr<crocoddyl::CostModelResidual> xRegCost = boost::make_shared<crocoddyl::CostModelResidual>(state, xResidual); 
     
    // END EFFECTOR FRAME TRANSLATION COST

    const int endeff_frame_id = rmodel->getFrameId("tool0");
    Eigen::Vector3d endeff_translation = {0.4, 0.4, 0.4};
    boost::shared_ptr<crocoddyl::ResidualModelFrameTranslation> frameTranslationResidual = boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                                                                                                    state, 
                                                                                                    endeff_frame_id, 
                                                                                                    endeff_translation    
                                                                                                );
    boost::shared_ptr<crocoddyl::CostModelResidual> frameTranslationCost = boost::make_shared<crocoddyl::CostModelResidual>(state, frameTranslationResidual); 

    // DEFINE CONSTRAINTS
    boost::shared_ptr<crocoddyl::ResidualModelFrameTranslation> frameTranslationConstraintResidual = boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(
                                                                                                    state, 
                                                                                                    endeff_frame_id, 
                                                                                                    Eigen::Vector3d::Zero()    
                                                                                                );
    
    Eigen::Vector3d lb = {-1.0, -1.0, -1.0};
    Eigen::Vector3d ub = {1.0, 0.4, 0.4};
    
    boost::shared_ptr<crocoddyl::ConstraintModelResidual> ee_constraint = boost::make_shared<crocoddyl::ConstraintModelResidual>(
                                                                                                    state, 
                                                                                                    frameTranslationResidual,
                                                                                                    lb,
                                                                                                    ub
                                                                                                );

    // CREATING RUNNING MODELS
    std::vector< boost::shared_ptr<crocoddyl::ActionModelAbstract> > runningModels;
    boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> terminal_model;
    const double dt = 5e-2;
    const int T = 40;

    for (unsigned t = 0; t < T + 1; ++t){
        boost::shared_ptr<crocoddyl::CostModelSum> runningCostModel = boost::make_shared<crocoddyl::CostModelSum>(state);
        runningCostModel->addCost("stateReg", xRegCost, 1e-1);
        runningCostModel->addCost("ctrlRegGrav", uRegCost, 1e-4);
        if (t != T){
            runningCostModel->addCost("translation", frameTranslationCost, 4);
        }
        else{
            runningCostModel->addCost("translation", frameTranslationCost, 40);
        }
        boost::shared_ptr<crocoddyl::ConstraintModelManager> constraints = boost::make_shared<crocoddyl::ConstraintModelManager>(state, nu);    
        if(t != 0){
            constraints->addConstraint("ee_bound", ee_constraint);
        }

        // CREATING DAM MODEL
        boost::shared_ptr<crocoddyl::DifferentialActionModelFreeFwdDynamics> running_DAM = boost::make_shared<crocoddyl::DifferentialActionModelFreeFwdDynamics>(
                                                                                                        state, 
                                                                                                        actuation,
                                                                                                        runningCostModel, 
                                                                                                        constraints
                                                                                                    );
        if (t != T){
            boost::shared_ptr<crocoddyl::IntegratedActionModelEuler> running_model = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(
                                                                                                        running_DAM,
                                                                                                        dt
                                                                                                        );
            runningModels.push_back(running_model);
        }
        else{
            terminal_model = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(running_DAM, dt);
        }
    }

    boost::shared_ptr<crocoddyl::ShootingProblem> problem = boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, terminal_model); 



    // Header
    
    std::cout << std::left << std::setw(42) << "      "
            << "  " << std::left << std::setw(15) << "AVG (ms)" << std::left
            << std::setw(15) << "STDDEV (ms)" << std::left << std::setw(15)
            << "MAX (ms)" << std::left << std::setw(15) << "MIN (ms)"
                << std::endl;


    mim_solvers::Timer timer;

    // SETTING UP WARM START
    std::vector<Eigen::VectorXd> xs(T + 1, x0);
    std::vector<Eigen::VectorXd> us(T, Eigen::VectorXd::Zero(nu));
    
    // Unonstrained case
    // DEFINE SOLVER
    mim_solvers::SolverSQP solver_SQP = mim_solvers::SolverSQP(problem);
    solver_SQP.set_termination_tolerance(0);
    solver_SQP.setCallbacks(false);
    const int max_iter_SQP = 1;

    // SETTING UP STATISTICS
    const int nb_SQP = 1000;
    Eigen::VectorXd duration_SQP(nb_SQP);
    for (unsigned i = 0; i < nb_SQP; ++i){
        timer.start();
        solver_SQP.solve(xs, us, max_iter_SQP);
        timer.stop();
        duration_SQP[i] = timer.elapsed().user;
    }
    double const std_dev_SQP = std::sqrt((duration_SQP.array() - duration_SQP.mean()).square().sum() / (nb_SQP - 1));

    std::cout << "  " << std::left << std::setw(42) << "UR5 SQP" << std::left
              << std::setw(15) << duration_SQP.mean() << std::left << std::setw(15)
              << std_dev_SQP << std::left << std::setw(15)
              << duration_SQP.maxCoeff() << std::left << std::setw(15)
              << duration_SQP.minCoeff() << std::endl;


    // Constrained case 50
    // DEFINE SOLVER
    mim_solvers::SolverCSQP solver_CSQP = mim_solvers::SolverCSQP(problem);
    solver_CSQP.set_termination_tolerance(1e-4);
    solver_CSQP.setCallbacks(false);
    solver_CSQP.set_eps_abs(0.0);
    solver_CSQP.set_eps_rel(0.0);
    solver_CSQP.set_max_qp_iters(50);
    const int max_iter_CSQP = 1;

    // SETTING UP STATISTICS
    const int nb_CSQP = 1000;
    Eigen::VectorXd duration_CSQP(nb_CSQP);
    for (unsigned i = 0; i < nb_CSQP; ++i){
        timer.start();
        solver_CSQP.solve(xs, us, max_iter_CSQP);
        timer.stop();
        duration_CSQP[i] = timer.elapsed().user;
    }
    double const std_dev_CSQP = std::sqrt((duration_CSQP.array() -  duration_CSQP.mean()).square().sum() / (nb_CSQP - 1));

    std::cout << "  " << std::left << std::setw(42) << "UR5 CSQP (50 QP iters)" << std::left
              << std::setw(15) <<  duration_CSQP.mean() << std::left << std::setw(15)
              << std_dev_CSQP << std::left << std::setw(15)
              << duration_CSQP.maxCoeff() << std::left << std::setw(15)
              << duration_CSQP.minCoeff() << std::endl;

    // Constrained case 200
    solver_CSQP.set_max_qp_iters(200);

    // SETTING UP STATISTICS
    for (unsigned i = 0; i < nb_CSQP; ++i){
        timer.start();
        solver_CSQP.solve(xs, us, max_iter_CSQP);
        timer.stop();
        duration_CSQP[i] = timer.elapsed().user;
    }
    double const std_dev_CSQP1000 = std::sqrt((duration_CSQP.array() -  duration_CSQP.mean()).square().sum() / (nb_CSQP - 1));

    std::cout << "  " << std::left << std::setw(42) << "UR5 CSQP (200 QP iters)" << std::left
              << std::setw(15) <<  duration_CSQP.mean() << std::left << std::setw(15)
              << std_dev_CSQP1000 << std::left << std::setw(15)
              << duration_CSQP.maxCoeff() << std::left << std::setw(15)
              << duration_CSQP.minCoeff() << std::endl;


    return 0;
}

