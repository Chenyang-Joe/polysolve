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
#include <Tpetra_Export.hpp>
#include <Tpetra_Import.hpp>

// Kokkos
#include <Kokkos_Core.hpp>

// Belos
#include <BelosConfigDefs.hpp>
#include <BelosLinearProblem.hpp>
#include <BelosBlockCGSolMgr.hpp>
#include <BelosPseudoBlockCGSolMgr.hpp>
#include <BelosBlockGmresSolMgr.hpp>
#include <BelosTpetraAdapter.hpp>

// MueLu
#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>

// Teuchos
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_DefaultMpiComm.hpp>  // For MpiComm
#include <Teuchos_DefaultSerialComm.hpp>  // For SerialComm

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
        virtual std::string name() const override { return "Trilinos GMRES + SA-AMG"; }

        // Outer Krylov method. Switchable via:
        //   JSON:  params["Trilinos"]["krylov"] = "gmres" | "cg"
        //   ENV :  POLYSOLVE_TRILINOS_KRYLOV = gmres | cg
        //
        //   GMRES — default. Belos::BlockGmresSolMgr. Tolerates non-SPD
        //           preconditioners and indefinite operators; uses more
        //           memory (restart_dim × N).
        //
        //   CG    — Belos::PseudoBlockCGSolMgr. Cheaper per iter for SPD
        //           matrices (which PolyFEM stiffness matrices are after
        //           the Hessian PSD projection), often converges faster.
        //           Requires both A and M (MueLu SA-AMG) to be SPD.
        enum class KrylovType {
            GMRES,
            CG,
        };

    protected:
        int numPDEs = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        double conv_tol_ = 1e-10;
        KrylovType krylov_type_ = KrylovType::GMRES; // default unchanged
        size_t iterations_ = 0;
        double residual_error_ = 0.0;
        bool is_nullspace_ = true;
        bool enable_repartition_ = false;  // Graph-based repartitioning (default: off, matching Hypre)
        Eigen::MatrixXd reduced_vertices;
        
        Teuchos::RCP<Operator> preconditioner_;
        Teuchos::RCP<const Map> rowMap_;

    private:
        int precond_num_;
        Teuchos::RCP<CrsMatrix> A_;
        double total_time;
        Teuchos::RCP<const Teuchos::Comm<int>> comm_;
        bool mpi_initialized_ = false;
        bool tpetra_initialized_ = false;
    };

} // namespace polysolve

// #endif