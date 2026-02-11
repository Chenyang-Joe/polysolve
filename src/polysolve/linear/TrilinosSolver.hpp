#pragma once

// #ifdef POLYSOLVE_WITH_TRILINOS

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

// Tpetra
#include <Tpetra_Core.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_MultiVector.hpp>

// Kokkos
#include <Kokkos_Core.hpp>

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

#include <Teuchos_CommandLineProcessor.hpp>

#include "../Utils.hpp"

namespace polysolve::linear
{

    class TrilinosSolver : public Solver
    {
    public:
        // Use default Tpetra types for now, or match PolySolve's index types if possible.
        // Assuming double scalar.
        using Scalar = double;
        using LocalOrdinal = int;
#ifdef POLYSOLVE_LARGE_INDEX
        using GlobalOrdinal = long long;
#else
        using GlobalOrdinal = int;
#endif
        using Node = Tpetra::Map<>::node_type;

        using Map = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
        using CrsMatrix = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
        using MultiVector = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
        using Operator = Tpetra::Operator<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
        using BelosProblem = Belos::LinearProblem<Scalar, MultiVector, Operator>;
        using BelosSolverMan = Belos::SolverManager<Scalar, MultiVector, Operator>;

        TrilinosSolver();
        ~TrilinosSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(TrilinosSolver)

    public:
        //////////////////////
        // Public interface //
        //////////////////////

        // Set solver parameters
        virtual void set_parameters(const json &params) override;

        // Retrieve memory information from Pardiso
        virtual void get_info(json &params) const override;

        // Analyze sparsity pattern
        virtual void analyze_pattern(const StiffnessMatrix &A, const int precond_num) override { precond_num_ = precond_num; }

        // Factorize system matrix
        virtual void factorize(const StiffnessMatrix &A) override;

        // Solve the linear system Ax = b
        virtual void solve(const Ref<const VectorXd> b, Ref<VectorXd> x) override;
        
        // Name of the solver type (for debugging purposes)
        virtual std::string name() const override { return "Trilinos Belos and MueLu"; }

    protected:
        int numPDEs = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        double conv_tol_ = 1e-8;
        size_t iterations_ = 0;
        double residual_error_ = 0.0;
        bool is_nullspace_ = true;
        Eigen::MatrixXd reduced_vertices;
        
        Teuchos::RCP<Operator> preconditioner_;
        Teuchos::RCP<const Map> rowMap_;

    private:
        int precond_num_;
        Teuchos::RCP<CrsMatrix> A_;
        double total_time;
        Teuchos::RCP<const Teuchos::Comm<int>> comm_;
    };

} // namespace polysolve

// #endif