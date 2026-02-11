#include "TrilinosSolver.hpp"
#include <string>
#include <vector>
#include <unsupported/Eigen/SparseExtra>

// POSIX headers for file descriptor manipulation
#include <unistd.h>
#include <fcntl.h>

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

// Teuchos
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

// Tpetra
#include <Tpetra_Core.hpp>

#ifdef HAVE_MPI
#include <mpi.h>
#endif

namespace polysolve::linear
{
    TrilinosSolver::TrilinosSolver()
    {
        // Initialize MPI if necessary
#ifdef HAVE_MPI
        int done_already;
        MPI_Initialized(&done_already);
        if (!done_already)
        {
            int argc = 1;
            char name[] = "polysolve";
            char *argv[] = {name};
            char **argvv = &argv[0];
            MPI_Init(&argc, &argvv);
        }
#endif
        // Get Default Communicator (wraps MPI_COMM_WORLD or Serial)
        comm_ = Tpetra::getDefaultComm();
    }
    
    TrilinosSolver::~TrilinosSolver()
    {
        // Let Kokkos/Trilinos handle cleanup automatically
        // Note: We do not call MPI_Finalize() to avoid interfering with outside MPI context.
    }

    void TrilinosSolver::set_parameters(const json &params)
    {
        if (params.contains("Trilinos"))
        {
            if (params["Trilinos"].contains("block_size"))
            {
                int bs = params["Trilinos"]["block_size"];
                if (bs == 2 || bs == 3)
                {
                    numPDEs = bs;
                }
            }
            if (params["Trilinos"].contains("max_iter"))
            {
                max_iter_ = params["Trilinos"]["max_iter"];
            }
            if (params["Trilinos"].contains("tolerance"))
            {
                conv_tol_ = params["Trilinos"]["tolerance"];
            }
            if (params["Trilinos"].contains("is_nullspace"))
            {
                is_nullspace_ = params["Trilinos"]["is_nullspace"];
            }
        }
    }

    void TrilinosSolver::get_info(json &params) const
    {
        params["num_iterations"] = iterations_;
        params["final_res_norm"] = residual_error_;
        params["solver_tol"] = conv_tol_;
        params["solver_maxiter"] = max_iter_;
    }

    void TrilinosSolver::factorize(const StiffnessMatrix &Ain)
    {
        POLYSOLVE_SCOPED_STOPWATCH("factorize", total_time, *logger);
        
        // Convert Eigen::SparseMatrix to Tpetra::CrsMatrix
        Eigen::SparseMatrix<double, Eigen::RowMajor> Arow(Ain);
        
        const GlobalOrdinal numGlobalRows = static_cast<GlobalOrdinal>(Arow.rows());
        const GlobalOrdinal indexBase = 0;
        
        // Verify divisibility
        if ((numGlobalRows % numPDEs) != 0) {
             throw std::runtime_error("Number of matrix rows is not divisible by #dofs");
        }

        // Create Map
        // We let Tpetra decide the distribution/map unless specific load balancing is needed.
        // Tpetra default constructor creates a uniform distribution.
        rowMap_ = Teuchos::rcp(new Map(numGlobalRows, indexBase, comm_));

        // Create Matrix
        // We estimate non-zeros. For correct parallel allocation, we should count local nnz.
        // For simplicity, we use dynamic profile (0) or strict count if easy.
        
        // Count NNZ for local rows to optimize allocation
        size_t localNumRows = rowMap_->getLocalNumElements();
        Teuchos::ArrayRCP<size_t> nnzPerRow(localNumRows);
        
        for(size_t i=0; i<localNumRows; ++i) {
            GlobalOrdinal gid = rowMap_->getGlobalElement(i);
            // Assuming Arow has global indexing.
            if(gid < Arow.outerSize()) {
                 nnzPerRow[i] = Arow.outerIndexPtr()[gid+1] - Arow.outerIndexPtr()[gid];
            } else {
                 nnzPerRow[i] = 0;
            }
        }

        // Create CrsMatrix with static profile graph (if possible) or dynamic
        A_ = Teuchos::rcp(new CrsMatrix(rowMap_, nnzPerRow()));

        // Fill Matrix
        // Iterate over LOCAL rows to avoid Tpetra errors about non-owned rows.
        for (size_t i = 0; i < localNumRows; ++i)
        {
            GlobalOrdinal globalRow = rowMap_->getGlobalElement(i);
            
            if (globalRow >= Arow.outerSize()) continue;
            
            int start = Arow.outerIndexPtr()[globalRow];
            int end = Arow.outerIndexPtr()[globalRow+1];
            int numEntries = end - start;
            
            const double* values_ptr = Arow.valuePtr() + start;
            const int* indices_ptr = Arow.innerIndexPtr() + start;
            
            std::vector<GlobalOrdinal> col_indices(numEntries);
            for(int k=0; k<numEntries; ++k) col_indices[k] = static_cast<GlobalOrdinal>(indices_ptr[k]);
            
            Teuchos::ArrayView<const double> valView(values_ptr, numEntries);
            Teuchos::ArrayView<const GlobalOrdinal> idxView(col_indices.data(), numEntries);
            
            A_->insertGlobalValues(globalRow, idxView, valView);
        }
        
        A_->fillComplete();
        
        // Preconditioner Setup (MueLu)
        Teuchos::ParameterList mueLuParams;
        mueLuParams.set("verbosity", "none");
        mueLuParams.set("coarse: max size", 1000);
        mueLuParams.set("multigrid algorithm", "sa");

        // Aggregation
        mueLuParams.set("aggregation: type", "uncoupled");
        mueLuParams.set("aggregation: drop tol", 0.08);

        // Smoother
        mueLuParams.set("smoother: type", "CHEBYSHEV");
        Teuchos::ParameterList& smootherList = mueLuParams.sublist("smoother: params");
        smootherList.set("chebyshev: degree", 5);
        smootherList.set("chebyshev: ratio eigenvalue", 30.0);
        
        // Nullspace Coordinates
        Teuchos::RCP<MultiVector> coords = Teuchos::null;
        if (numPDEs > 1 && is_nullspace_ && reduced_vertices.rows() > 0)
        {
             // TODO: Implement coordinate transfer if necessary.
             // Usually mapping Eigen Matrix to MultiVector.
             // For now, we skip explicit coordinates unless crucial for convergence on this problem.
             // If needed:
             // 1. Create Map for Nodes (numGlobalRows / numPDEs)
             // 2. Create MultiVector(nodeMap, 3)
             // 3. Fill.
             // 4. Pass to CreateTpetraPreconditioner as 3rd arg or via "Coordinates" param.
        }

        try {
            preconditioner_ = MueLu::CreateTpetraPreconditioner((Teuchos::RCP<Operator>)A_, mueLuParams);
        } catch (const std::exception& e) {
            std::cerr << "MueLu setup failed: " << e.what() << std::endl;
            // Fallback or rethrow
            throw;
        }
    }

    void TrilinosSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        POLYSOLVE_SCOPED_STOPWATCH("solve", total_time, *logger);
        
        if (A_ == Teuchos::null) throw std::runtime_error("Matrix not factorized");

        // Create Vectors
        Teuchos::RCP<MultiVector> X = Teuchos::rcp(new MultiVector(rowMap_, 1));
        Teuchos::RCP<MultiVector> B = Teuchos::rcp(new MultiVector(rowMap_, 1));
        
        // Fill B and Initial Guess X
        // Assuming we can access local data directly.
        // Note: rhs/result are global Eigen vectors?
        
        // Copy data logic matching original distribution assumptions
        auto x_data = X->getDataNonConst(0);
        auto b_data = B->getDataNonConst(0);
        
        size_t localLen = X->getLocalLength();
        for(size_t i=0; i<localLen; ++i)
        {
            GlobalOrdinal gid = rowMap_->getGlobalElement(i);
            if (gid < rhs.size()) {
                b_data[i] = rhs[gid];
                x_data[i] = result[gid]; // Initial guess
            }
        }
        
        // Linear Problem
        Teuchos::RCP<BelosProblem> problem = Teuchos::rcp(new BelosProblem(A_, X, B));
        
        if (preconditioner_ != Teuchos::null) {
            problem->setLeftPrec(preconditioner_);
        }
        
        bool set = problem->setProblem();
        if (!set) {
            throw std::runtime_error("Belos::LinearProblem::setProblem() failed");
        }
        
        // Solver Parameter List
        Teuchos::ParameterList belosList;
        belosList.set("Maximum Iterations", max_iter_);
        belosList.set("Convergence Tolerance", conv_tol_);
        belosList.set("Verbosity", Belos::Errors + Belos::Warnings);
        belosList.set("Output Frequency", 50);  // Print every 50 iterations
        
        // Use GMRES instead of CG for general (non-SPD) matrices
        // GMRES works for any matrix, while CG requires symmetric positive-definite
        Belos::BlockGmresSolMgr<Scalar, MultiVector, Operator> solver(problem, Teuchos::rcp(&belosList, false));
        
        // Solve
        Belos::ReturnType ret;
        try {
            ret = solver.solve();
        } catch (const std::exception& e) {
            std::cerr << "Belos solver threw exception: " << e.what() << std::endl;
            throw;
        }
        
        iterations_ = solver.getNumIters();
        residual_error_ = solver.achievedTol();

        if (ret != Belos::Converged) {
            // Log warning but don't throw - some applications may accept non-converged solutions
            std::cerr << "Warning: Belos did not converge after " << iterations_ 
                      << " iterations. Final residual: " << residual_error_ << std::endl;
            // Uncomment to make non-convergence fatal:
            // throw std::runtime_error("Belos did not converge");
        }
        
        // Copy result back
        for(size_t i=0; i<localLen; ++i)
        {
            GlobalOrdinal gid = rowMap_->getGlobalElement(i);
            if (gid < result.size()) {
                result[gid] = x_data[i];
            }
        }
    }
}
