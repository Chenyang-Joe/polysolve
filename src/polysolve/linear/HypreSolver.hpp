#pragma once

////////////////////////////////////////////////////////////////////////////////
#include "Solver.hpp"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

#include <HYPRE_utilities.h>
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
#include <HYPRE_parcsr_mv.h>

#include "../Utils.hpp"
////////////////////////////////////////////////////////////////////////////////
//
// https://computation.llnl.gov/sites/default/files/public/hypre-2.11.2_usr_manual.pdf
// https://github.com/LLNL/hypre/blob/v2.14.0/docs/HYPRE_usr_manual.pdf
//

namespace polysolve::linear
{

    class HypreSolver : public Solver
    {

    public:
        HypreSolver();
        ~HypreSolver();

    private:
        POLYSOLVE_DELETE_MOVE_COPY(HypreSolver)

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
        virtual std::string name() const override { return "Hypre"; }

    public:
        // How matrix rows are split across MPI ranks. Selected at runtime via
        // the env var POLYSOLVE_HYPRE_PARTITION (case-insensitive). See
        // HypreSolver.cpp::read_partition_mode_from_env for the parser.
        //
        //   RowBlock  — default. Rank k owns rows [k*N/P, (k+1)*N/P). Cheap,
        //               but cuts the matrix graph at arbitrary index
        //               boundaries; for ill-conditioned matrices (mat_twist
        //               etc.) this weakens BoomerAMG's coarsening enough that
        //               PCG breaks down with "Subnormal gamma" at np >= 3.
        //
        //   RankZero  — debug/fallback. Rank 0 owns all rows, others own the
        //               empty range. Effectively serial. Useful to confirm
        //               "the matrix itself is fine, only the partition is
        //               the problem".
        //
        //   Metis     — call METIS_PartGraphKway on the matrix's adjacency
        //               graph, then permute rows so each rank's vertices form
        //               a contiguous block. Minimizes cut edges across ranks,
        //               which preserves BoomerAMG's preconditioner quality.
        //               Requires POLYSOLVE_WITH_METIS at compile time.
        enum class PartitionMode {
            RowBlock,
            RankZero,
            Metis,
        };

        // Which preconditioner PCG should use. Switchable via either
        //   JSON:  params["Hypre"]["preconditioner"] = "boomeramg" | "euclid"
        //   ENV :  POLYSOLVE_HYPRE_PRECOND = boomeramg | euclid
        // JSON wins if both are set, since set_parameters() is called after
        // the constructor has already populated precond_type_ from env.
        //
        //   BoomerAMG — default. Classical algebraic multigrid. Best on
        //               well-conditioned elliptic PDEs; can break down on
        //               near-singular contact matrices (mat_twist).
        //
        //   Euclid    — parallel ILU(k). More robust on ill-conditioned
        //               matrices because each application of M⁻¹ is an
        //               explicit triangular solve rather than an AMG cycle,
        //               so the preconditioned residual ‖M⁻¹r‖ doesn't
        //               collapse the way it does with a weak AMG.
        enum class PreconditionerType {
            BoomerAMG,
            Euclid,
        };

        // Which outer Krylov method drives the iteration. Switchable via:
        //   JSON:  params["Hypre"]["krylov"] = "pcg" | "gmres"
        //   ENV :  POLYSOLVE_HYPRE_KRYLOV = pcg | gmres
        //
        //   PCG   — default. 3-term Lanczos recurrence, minimal memory,
        //           requires SPD matrix and SPD preconditioner. Hypre's
        //           implementation has a "subnormal gamma" bail-out
        //           (pcg.c:709) that fires when (r, M⁻¹r) underflows;
        //           this is triggered by ill-conditioned matrices.
        //
        //   GMRES — Arnoldi-based, minimizes 2-norm residual. Does NOT
        //           compute (r, M⁻¹r), so it has no subnormal bail-out
        //           and tolerates non-SPD preconditioners. Uses more
        //           memory (kdim × N vectors) but is more robust on
        //           ill-conditioned systems. This is what Trilinos/Belos
        //           wraps by default.
        enum class KrylovType {
            PCG,
            GMRES,
        };

    protected:
        int dimension_ = 1; // 1 = scalar (Laplace), 2 or 3 = vector (Elasticity)
        int max_iter_ = 1000;
        int pre_max_iter_ = 1;
        double conv_tol_ = 1e-10;

        HYPRE_Int num_iterations;
        HYPRE_Complex final_res_norm;

    private:
        bool has_matrix_ = false;
        int precond_num_;

        HYPRE_IJMatrix A;
        HYPRE_ParCSRMatrix parcsr_A;

        double total_time;

        int mpi_rank_ = 0;
        int mpi_size_ = 1;
        HYPRE_Int ilower_ = 0;
        HYPRE_Int iupper_ = 0;
        HYPRE_Int num_rows_ = 0;

        // --- Partition control (added 2026-05-20) ---
        PartitionMode partition_mode_ = PartitionMode::RowBlock;

        // --- Preconditioner choice (added 2026-05-20) ---
        PreconditionerType precond_type_ = PreconditionerType::BoomerAMG;

        // Euclid ILU(k) level (default 1; only used when precond_type_ == Euclid).
        // Higher k → better preconditioner but more memory + setup time.
        int euclid_level_ = 1;

        // --- Krylov choice (added 2026-05-20) ---
        KrylovType krylov_type_ = KrylovType::PCG;

        // GMRES restart dimension (only used when krylov_type_ == GMRES).
        // Hypre default is 5; 30 is a more typical robustness sweet spot.
        // Larger k → less restart overhead per outer iter but k × N memory.
        int gmres_kdim_ = 30;

        // Per-rank ownership: rank r owns rows [rank_ilower_[r], rank_iupper_[r]].
        // For RowBlock this is just the trivial block split; for Metis it is
        // computed after permutation so still contiguous per rank. Lengths =
        // mpi_size_. Cached once per factorize() call and reused in solve()
        // for MPI_Allgatherv recvcounts/displs.
        std::vector<HYPRE_Int> rank_ilower_;
        std::vector<HYPRE_Int> rank_iupper_;

        // METIS permutation. Empty when partition_mode_ != Metis (identity).
        //   perm_[i]     = new row index of original row i  (used: A' = P A P^T)
        //   inv_perm_[j] = original row index of new row j  (used: x = P^T x')
        // Both rebuilt every factorize() so that solve() can apply b' = P b
        // and undo it after Hypre returns.
        std::vector<int> perm_;
        std::vector<int> inv_perm_;
    };

} // namespace polysolve::linear
