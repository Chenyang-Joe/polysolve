#include "Newton.hpp"

#include <polysolve/Utils.hpp>

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

#include <polysolve/save_problem.hpp>

namespace polysolve::nonlinear
{

    std::vector<std::shared_ptr<DescentStrategy>> Newton::create_solver(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
    {
        // Copies stuff from main newton
        json proj_solver_params = R"({"ProjectedNewton": {}})"_json;
        proj_solver_params["ProjectedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];

        json reg_solver_params = R"({"RegularizedNewton": {}})"_json;
        reg_solver_params["RegularizedNewton"]["residual_tolerance"] = solver_params["Newton"]["residual_tolerance"];
        reg_solver_params["RegularizedNewton"]["reg_weight_min"] = solver_params["Newton"]["reg_weight_min"];
        reg_solver_params["RegularizedNewton"]["reg_weight_max"] = solver_params["Newton"]["reg_weight_max"];
        reg_solver_params["RegularizedNewton"]["reg_weight_inc"] = solver_params["Newton"]["reg_weight_inc"];

        std::vector<std::shared_ptr<DescentStrategy>> res;
        const bool force_psd_projection = solver_params["Newton"]["force_psd_projection"];
        if (!force_psd_projection)
            res.push_back(std::make_unique<Newton>(
                sparse,
                solver_params, linear_solver_params,
                characteristic_length, logger));

        const bool use_psd_projection = solver_params["Newton"]["use_psd_projection"];
        if (use_psd_projection)
            res.push_back(std::make_unique<ProjectedNewton>(
                sparse,
                proj_solver_params, linear_solver_params,
                characteristic_length, logger));

        const double reg_weight_min = solver_params["Newton"]["reg_weight_min"];
        if (reg_weight_min > 0)
            res.push_back(std::make_unique<RegularizedNewton>(
                sparse, solver_params["Newton"]["use_psd_projection_in_regularized"],
                reg_solver_params, linear_solver_params,
                characteristic_length, logger));

        if (res.empty())
            log_and_throw_error(logger, "Newton needs to have at least one of force_psd_projection=false, reg_weight_min>0, or use_psd_projection=true");

        return res;
    }

    Newton::Newton(const bool sparse,
                   const double residual_tolerance,
                   const json &solver_params,
                   const json &linear_solver_params,
                   const double characteristic_length,
                   spdlog::logger &logger)
        : Superclass(solver_params, characteristic_length, logger),
          is_sparse(sparse), characteristic_length(characteristic_length), residual_tolerance(residual_tolerance)
    {
        linear_solver = polysolve::linear::Solver::create(linear_solver_params, logger);
        if (linear_solver->is_dense() == sparse)
            log_and_throw_error(logger, "Newton linear solver must be {}, instead got {}", sparse ? "sparse" : "dense", linear_solver->name());

        if (residual_tolerance <= 0)
            log_and_throw_error(logger, "Newton residual_tolerance must be > 0, instead got {}", residual_tolerance);
    }

    Newton::Newton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Newton(sparse, extract_param("Newton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, logger)
    {
    }

    ProjectedNewton::ProjectedNewton(
        const bool sparse,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(sparse, extract_param("ProjectedNewton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, logger)
    {
    }

    RegularizedNewton::RegularizedNewton(
        const bool sparse,
        const bool project_to_psd,
        const json &solver_params,
        const json &linear_solver_params,
        const double characteristic_length,
        spdlog::logger &logger)
        : Superclass(sparse, extract_param("RegularizedNewton", "residual_tolerance", solver_params), solver_params, linear_solver_params, characteristic_length, logger),
          project_to_psd(project_to_psd)
    {
        reg_weight_min = extract_param("RegularizedNewton", "reg_weight_min", solver_params);
        reg_weight_max = extract_param("RegularizedNewton", "reg_weight_max", solver_params);
        reg_weight_inc = extract_param("RegularizedNewton", "reg_weight_inc", solver_params);

        reg_weight = reg_weight_min;

        if (reg_weight_min <= 0)
            log_and_throw_error(logger, "Newton reg_weight_min must be  > 0, instead got {}", reg_weight_min);

        if (reg_weight_inc <= 1)
            log_and_throw_error(logger, "Newton reg_weight_inc must be  > 1, instead got {}", reg_weight_inc);

        if (reg_weight_max <= reg_weight_min)
            log_and_throw_error(logger, "Newton reg_weight_max must be  > {}, instead got {}", reg_weight_min, reg_weight_max);
    }

    // =======================================================================

    void Newton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        internal_solver_info = json::array();
    }

    void RegularizedNewton::reset(const int ndof)
    {
        Superclass::reset(ndof);
        reg_weight = reg_weight_min;
    }

    // =======================================================================

    bool Newton::compute_update_direction(
        Problem &objFunc,
        const TVector &x,
        const TVector &grad,
        TVector &direction)
    {
        const double residual =
            is_sparse ? solve_sparse_linear_system(objFunc, x, grad, direction)
                      : solve_dense_linear_system(objFunc, x, grad, direction);

        if (std::isnan(residual) || residual > residual_tolerance * characteristic_length)
        {
            m_logger.debug("[{}] large (or nan) linear solve residual {}>{} (‖∇f‖={})",
                           name(), residual, residual_tolerance * characteristic_length, grad.norm());

            return false;
        }
        else
        {
            m_logger.trace("linear solve residual {}", residual);
        }

        return true;
    }

    // =======================================================================

    double Newton::solve_sparse_linear_system(Problem &objFunc,
                                              const TVector &x,
                                              const TVector &grad,
                                              TVector &direction)
    {        
        polysolve::StiffnessMatrix hessian;

        // Eigen::SparseMatrix<double> hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        // Addon 1: manually save the hessian and grad for analysis
		benchy::io::Problem<double> probleminfo;
		probleminfo.A = hessian;
		probleminfo.b = -grad;
		probleminfo.is_projected = is_psd_projected() ? 1 : 0;
		benchy::io::iter_global++;  // starts from 1
		std::cout << "TIME STEP: " << benchy::io::ts_global << " ITER: " << benchy::io::iter_global << " SOLVER: " << name() << std::endl;
		// probleminfo.nullspace = remove_boundary_vertices(test_vertices, test_boundary_nodes);
		benchy::io::save_problem(probleminfo);

        
        // Addon 2: load the matrix and vector that can be solved by Eigen::CholmodSupernodalLLT solely
        // and test if it can be solved by the same solver in polysolve integreated in the polyfem framework
        // std::cout << "Testing loading matrix and vector using save_problem.hpp functions" << std::endl;
        // std::string matrix_file = "/u/1/chenyang/mat_analysis/data/A_sonic.bin";
        // std::string vector_file = "/u/1/chenyang/mat_analysis/data/1_1_b.bin";
        // SparseMatrix<double> A;
        // int dim_local, is_spd, is_sequence;
        // benchy::io::Deserialize<double, ColMajor, int>(A, dim_local, is_spd, is_sequence, matrix_file);
        // MatrixXd b_matrix;
        // benchy::io::ReadMat(b_matrix, vector_file);
        // VectorXd b = b_matrix.col(0);
        // json temp_solver_params_0 = R"({"solver": "Eigen::CholmodSupernodalLLT"})"_json;
        // std::shared_ptr<polysolve::linear::Solver> temp_solver_0 = polysolve::linear::Solver::create(temp_solver_params_0, m_logger, false);
        // try{
        //     temp_solver_0->analyze_pattern(A, A.rows());
        //     temp_solver_0->factorize(A);
        //     VectorXd x0(b.size());
        //     temp_solver_0->solve(b, x0); // A x = b
        //     double temp_residual_0 = (A * x0 - b).norm(); // A x - b = 0
        //     m_logger.info("TestPolysolve solve successful, residual: {}", temp_residual_0);
        // }
        // catch (const std::runtime_error &err) {
        //     m_logger.warn("TestPolysolve factorization failed: {}", err.what());
        // }




        // Addon 3: log the sum of hessian and -grad for debugging
        // double hessian_sum = hessian.sum();
        // double grad_sum = -grad.sum();
        // m_logger.info("Hessian sum: {}, -grad sum: {}", hessian_sum, grad_sum);

        // Addon 4: test if the matrix passed to polyfem (not reloaded version, but the vanilla version)
        // can be solved by the same solver in polysolve integreated in the polyfem framework
        // double verification_time = 0;
        // {
        //     POLYSOLVE_SCOPED_STOPWATCH("test Eigen::CholmodSupernodalLLT", verification_time, m_logger);

        //     m_logger.info("Testing Eigen::CholmodSupernodalLLT solver compatibility");
        //     json temp_solver_params = R"({"solver": "Eigen::CholmodSupernodalLLT"})"_json;
        //     std::shared_ptr<polysolve::linear::Solver> temp_solver = polysolve::linear::Solver::create(temp_solver_params, m_logger, false);
            
        //     try {
        //         temp_solver->analyze_pattern(hessian, hessian.rows());
        //         temp_solver->factorize(hessian);
        //         VectorXd x(grad.size());
        //         temp_solver->solve(-grad, x); // H Δx = -g
        //         double temp_residual = (hessian * x + grad).norm(); // H Δx + g = 0
        //         m_logger.info("Eigen::CholmodSupernodalLLT solve successful, residual: {}", temp_residual);
                
        //     }
        //     catch (const std::runtime_error &err) {
        //         m_logger.warn("Eigen::CholmodSupernodalLLT factorization failed: {}", err.what());
        //     }
        // }
        // m_logger.info("verification time: {}", verification_time);


        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);
            // TODO: get the correct size
            linear_solver->analyze_pattern(hessian, hessian.rows());

            try
            {
                linear_solver->factorize(hessian);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"", err.what());

                // Eigen::saveMarket(hessian, "problematic_hessian.mtx");
                return std::nan("");
            }


            
            linear_solver->solve(-grad, direction); // H Δx = -g
        }

        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);

        return residual;
    }

    double Newton::solve_dense_linear_system(Problem &objFunc,
                                             const TVector &x,
                                             const TVector &grad,
                                             TVector &direction)
    {
        Eigen::MatrixXd hessian;

        {
            POLYSOLVE_SCOPED_STOPWATCH("assembly time", this->assembly_time, m_logger);
            compute_hessian(objFunc, x, hessian);
        }

        {
            POLYSOLVE_SCOPED_STOPWATCH("linear solve", this->inverting_time, m_logger);

            try
            {
                linear_solver->analyze_pattern_dense(hessian, hessian.rows());
                linear_solver->factorize_dense(hessian);
                linear_solver->solve(-grad, direction);
            }
            catch (const std::runtime_error &err)
            {
                // warn if using gradient descent
                m_logger.debug("Unable to factorize Hessian: \"{}\"",
                               err.what());

                return std::nan("");
            }
        }

        const double residual = (hessian * direction + grad).norm(); // H Δx + g = 0

        json info;
        linear_solver->get_info(info);
        internal_solver_info.push_back(info);

        return residual;
    }
    // =======================================================================

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          polysolve::StiffnessMatrix &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            polysolve::StiffnessMatrix &hessian)

    {
        if (x.size() != x_cache.size() || x != x_cache)
        {
            objFunc.set_project_to_psd(project_to_psd);
            objFunc.hessian(x, hessian_cache);
            x_cache = x;
        }
        hessian = hessian_cache;
        if (reg_weight > 0)
        {
            hessian += reg_weight * sparse_identity(hessian.rows(), hessian.cols());
        }
    }

    void Newton::compute_hessian(Problem &objFunc,
                                 const TVector &x,
                                 Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(false);
        objFunc.hessian(x, hessian);
    }

    void ProjectedNewton::compute_hessian(Problem &objFunc,
                                          const TVector &x,
                                          Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(true);
        objFunc.hessian(x, hessian);
    }

    void RegularizedNewton::compute_hessian(Problem &objFunc,
                                            const TVector &x,
                                            Eigen::MatrixXd &hessian)

    {
        objFunc.set_project_to_psd(project_to_psd);
        objFunc.hessian(x, hessian);
        if (reg_weight > 0)
        {
            for (int k = 0; k < x.size(); k++)
                hessian(k, k) += reg_weight;
        }
    }
    // =======================================================================

    bool RegularizedNewton::handle_error()
    {
        reg_weight *= reg_weight_inc;
        return reg_weight < reg_weight_max;
    }
    // =======================================================================

    void Newton::update_solver_info(json &solver_info, const double per_iteration)
    {
        Superclass::update_solver_info(solver_info, per_iteration);

        solver_info["internal_solver"] = internal_solver_info;
        solver_info["time_assembly"] = assembly_time / per_iteration;
        solver_info["time_inverting"] = inverting_time / per_iteration;
    }

    void Newton::reset_times()
    {
        assembly_time = 0;
        inverting_time = 0;
    }

    void Newton::log_times() const
    {
        if (assembly_time <= 0 && inverting_time <= 0)
            return; // nothing to log
        m_logger.debug(
            "[{}][{}] assembly: {:.2e}s; linear_solve: {:.2e}s",
            fmt::format(fmt::fg(fmt::terminal_color::magenta), "timing"),
            name(), assembly_time, inverting_time);
    }

    // =======================================================================

} // namespace polysolve::nonlinear
