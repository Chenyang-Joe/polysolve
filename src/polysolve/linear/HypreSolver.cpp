#ifdef POLYSOLVE_WITH_HYPRE

////////////////////////////////////////////////////////////////////////////////
#include "HypreSolver.hpp"

#include <HYPRE_krylov.h>
#include <HYPRE_utilities.h>
#include <algorithm>
#include <cstdlib>   // getenv
#include <cstring>   // strcasecmp
#include <numeric>   // iota

#if defined(SPDLOG_FMT_EXTERNAL)
#include <fmt/color.h>
#else
#include <spdlog/fmt/bundled/color.h>
#endif

#ifdef POLYSOLVE_WITH_METIS
#  include <metis.h>
#endif
////////////////////////////////////////////////////////////////////////////////

namespace polysolve::linear
{

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {
        // Parse the env var POLYSOLVE_HYPRE_PARTITION. Accepts:
        //   "row_block"  / "rowblock" / "block" / ""   → RowBlock (default)
        //   "rank_zero"  / "rankzero" / "zero"          → RankZero
        //   "metis"                                     → Metis
        // Any unrecognized value falls back to RowBlock with a stderr warning
        // so a typo doesn't silently activate the wrong path.
        HypreSolver::PartitionMode read_partition_mode_from_env()
        {
            const char *env = std::getenv("POLYSOLVE_HYPRE_PARTITION");
            if (env == nullptr || env[0] == '\0') {
                return HypreSolver::PartitionMode::RowBlock;
            }
            if (strcasecmp(env, "row_block") == 0
             || strcasecmp(env, "rowblock") == 0
             || strcasecmp(env, "block")    == 0) {
                return HypreSolver::PartitionMode::RowBlock;
            }
            if (strcasecmp(env, "rank_zero") == 0
             || strcasecmp(env, "rankzero")  == 0
             || strcasecmp(env, "zero")      == 0) {
                return HypreSolver::PartitionMode::RankZero;
            }
            if (strcasecmp(env, "metis") == 0) {
#ifdef POLYSOLVE_WITH_METIS
                return HypreSolver::PartitionMode::Metis;
#else
                std::fprintf(stderr,
                    "polysolve: POLYSOLVE_HYPRE_PARTITION=metis requested but "
                    "this build does not have METIS (POLYSOLVE_WITH_METIS=OFF). "
                    "Falling back to row_block.\n");
                return HypreSolver::PartitionMode::RowBlock;
#endif
            }
            std::fprintf(stderr,
                "polysolve: unknown POLYSOLVE_HYPRE_PARTITION='%s'; "
                "falling back to row_block.\n", env);
            return HypreSolver::PartitionMode::RowBlock;
        }

        const char *partition_mode_name(HypreSolver::PartitionMode m) {
            switch (m) {
                case HypreSolver::PartitionMode::RowBlock: return "row_block";
                case HypreSolver::PartitionMode::RankZero: return "rank_zero";
                case HypreSolver::PartitionMode::Metis:    return "metis";
            }
            return "?";
        }

        // Parse POLYSOLVE_HYPRE_PRECOND. Accepts:
        //   ""  / "boomeramg" / "amg"  → BoomerAMG (default)
        //   "euclid" / "ilu"           → Euclid
        // Unknown values warn + fall back to BoomerAMG.
        HypreSolver::PreconditionerType read_precond_type_from_env()
        {
            const char *env = std::getenv("POLYSOLVE_HYPRE_PRECOND");
            if (env == nullptr || env[0] == '\0') {
                return HypreSolver::PreconditionerType::BoomerAMG;
            }
            if (strcasecmp(env, "boomeramg") == 0
             || strcasecmp(env, "amg")       == 0) {
                return HypreSolver::PreconditionerType::BoomerAMG;
            }
            if (strcasecmp(env, "euclid") == 0
             || strcasecmp(env, "ilu")    == 0) {
                return HypreSolver::PreconditionerType::Euclid;
            }
            std::fprintf(stderr,
                "polysolve: unknown POLYSOLVE_HYPRE_PRECOND='%s'; "
                "falling back to boomeramg.\n", env);
            return HypreSolver::PreconditionerType::BoomerAMG;
        }

        const char *precond_type_name(HypreSolver::PreconditionerType p) {
            switch (p) {
                case HypreSolver::PreconditionerType::BoomerAMG: return "boomeramg";
                case HypreSolver::PreconditionerType::Euclid:    return "euclid";
            }
            return "?";
        }

        // Parse POLYSOLVE_HYPRE_KRYLOV. Accepts:
        //   "" / "pcg" / "cg"   → PCG (default)
        //   "gmres"             → GMRES
        // Unknown values warn + fall back to PCG.
        HypreSolver::KrylovType read_krylov_type_from_env()
        {
            const char *env = std::getenv("POLYSOLVE_HYPRE_KRYLOV");
            if (env == nullptr || env[0] == '\0') {
                return HypreSolver::KrylovType::PCG;
            }
            if (strcasecmp(env, "pcg") == 0 || strcasecmp(env, "cg") == 0) {
                return HypreSolver::KrylovType::PCG;
            }
            if (strcasecmp(env, "gmres") == 0) {
                return HypreSolver::KrylovType::GMRES;
            }
            std::fprintf(stderr,
                "polysolve: unknown POLYSOLVE_HYPRE_KRYLOV='%s'; "
                "falling back to pcg.\n", env);
            return HypreSolver::KrylovType::PCG;
        }

        const char *krylov_type_name(HypreSolver::KrylovType k) {
            switch (k) {
                case HypreSolver::KrylovType::PCG:   return "pcg";
                case HypreSolver::KrylovType::GMRES: return "gmres";
            }
            return "?";
        }
    } // anonymous namespace

    HypreSolver::HypreSolver()
    {
        precond_num_ = 0;
#ifdef HYPRE_WITH_MPI
        int done_already;

        MPI_Initialized(&done_already);
        if (!done_already)
        {
            /* Initialize MPI */
            int argc = 1;
            char name[] = "";
            char *argv[] = {name};
            char **argvv = &argv[0];
            MPI_Init(&argc, &argvv);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size_);

        // Pick partition mode + preconditioner + krylov from env (defaults:
        // RowBlock + BoomerAMG + PCG). set_parameters() can later override
        // precond and krylov via JSON; partition is env-only. Only rank 0
        // announces to keep mpirun -np N stdout readable.
        partition_mode_ = read_partition_mode_from_env();
        precond_type_   = read_precond_type_from_env();
        krylov_type_    = read_krylov_type_from_env();
        // Note: max_iter is *not* env-controlled. To change it, edit the
        // "default" value of /Hypre/max_iter in linear-solver-spec.json
        // (Solver::create() injects that default and calls set_parameters()
        // with it, which lands in max_iter_ before solve() runs).
        if (mpi_rank_ == 0) {
            std::fprintf(stderr,
                "[polysolve::HypreSolver] mpi_size=%d  partition=%s  precond=%s  krylov=%s\n",
                mpi_size_,
                partition_mode_name(partition_mode_),
                precond_type_name(precond_type_),
                krylov_type_name(krylov_type_));
        }
#endif
    }

    // Set solver parameters
    void HypreSolver::set_parameters(const json &params)
    {
        if (params.contains("Hypre"))
        {
            if (params["Hypre"].contains("max_iter"))
            {
                max_iter_ = params["Hypre"]["max_iter"];
            }
            if (params["Hypre"].contains("pre_max_iter"))
            {
                pre_max_iter_ = params["Hypre"]["pre_max_iter"];
            }
            if (params["Hypre"].contains("tolerance"))
            {
                conv_tol_ = params["Hypre"]["tolerance"];
            }
            // Preconditioner: "boomeramg" (default) | "euclid".
            // Overrides whatever was picked from POLYSOLVE_HYPRE_PRECOND
            // env var at construction time.
            if (params["Hypre"].contains("preconditioner"))
            {
                const std::string s = params["Hypre"]["preconditioner"].get<std::string>();
                if (strcasecmp(s.c_str(), "boomeramg") == 0
                 || strcasecmp(s.c_str(), "amg")       == 0) {
                    precond_type_ = PreconditionerType::BoomerAMG;
                } else if (strcasecmp(s.c_str(), "euclid") == 0
                        || strcasecmp(s.c_str(), "ilu")    == 0) {
                    precond_type_ = PreconditionerType::Euclid;
                } else {
                    std::fprintf(stderr,
                        "polysolve: unknown Hypre.preconditioner='%s'; "
                        "keeping %s.\n", s.c_str(), precond_type_name(precond_type_));
                }
            }
            // Euclid ILU(k) level. Only meaningful when preconditioner=euclid.
            if (params["Hypre"].contains("euclid_level"))
            {
                euclid_level_ = params["Hypre"]["euclid_level"];
            }
            // Krylov method: "pcg" (default) | "gmres".
            // Overrides POLYSOLVE_HYPRE_KRYLOV from construction.
            if (params["Hypre"].contains("krylov"))
            {
                const std::string s = params["Hypre"]["krylov"].get<std::string>();
                if (strcasecmp(s.c_str(), "pcg") == 0
                 || strcasecmp(s.c_str(), "cg")  == 0) {
                    krylov_type_ = KrylovType::PCG;
                } else if (strcasecmp(s.c_str(), "gmres") == 0) {
                    krylov_type_ = KrylovType::GMRES;
                } else {
                    std::fprintf(stderr,
                        "polysolve: unknown Hypre.krylov='%s'; keeping %s.\n",
                        s.c_str(), krylov_type_name(krylov_type_));
                }
            }
            // GMRES restart dimension. Only meaningful when krylov=gmres.
            if (params["Hypre"].contains("gmres_kdim"))
            {
                gmres_kdim_ = params["Hypre"]["gmres_kdim"];
            }
        }
    }

    void HypreSolver::get_info(json &params) const
    {
        params["num_iterations"] = num_iterations;
        params["final_res_norm"] = final_res_norm;
        params["solver_maxiter"] = max_iter_;
        params["solver_tol"] = conv_tol_;
    }

    ////////////////////////////////////////////////////////////////////////////////

    void HypreSolver::factorize(const StiffnessMatrix &Ain)
    {
        // TODO: distributed loading. Currently every MPI process receives the
        // FULL Eigen matrix `Ain` and only inserts its local rows via the
        // `if (it.row() < ilower_ || it.row() > iupper_) continue;` filter
        // below. This means total memory = P * sizeof(full matrix), so memory
        // does NOT scale down with more processes. For matrices approaching
        // per-node RAM limits, the caller should instead load only its local
        // row range (e.g. via parallel I/O or rank-0 scatter) and pass a
        // pre-partitioned matrix here.
        POLYSOLVE_SCOPED_STOPWATCH("factorize", total_time, *logger);
        assert(precond_num_ > 0);

        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }

        has_matrix_ = true;
        const HYPRE_Int rows = Ain.rows();
        const HYPRE_Int cols = Ain.cols();
        (void)cols; // only used in non-MPI branch below; silence unused warning

#ifdef HYPRE_WITH_MPI
        num_rows_ = rows;

        // -----------------------------------------------------------------
        // Decide partition. Each branch sets:
        //   - ilower_, iupper_           (this rank's contiguous row range)
        //   - rank_ilower_[r], rank_iupper_[r]  (for all r; used in solve())
        //   - perm_, inv_perm_           (METIS only; empty otherwise = identity)
        // After this block, the rest of factorize() inserts matrix entries
        // assuming the (possibly permuted) row numbering encoded in perm_.
        // -----------------------------------------------------------------
        rank_ilower_.assign(mpi_size_, 0);
        rank_iupper_.assign(mpi_size_, -1);
        perm_.clear();
        inv_perm_.clear();

        if (partition_mode_ == PartitionMode::RowBlock)
        {
            // Classical block split: rank k gets rows [k*base + ..., ...].
            // No permutation — perm_ stays empty, treated as identity.
            const HYPRE_Int base = rows / mpi_size_;
            const HYPRE_Int rem  = rows % mpi_size_;
            for (int r = 0; r < mpi_size_; ++r) {
                rank_ilower_[r] = r * base + std::min<HYPRE_Int>(r, rem);
                rank_iupper_[r] = rank_ilower_[r] + base - 1 + (r < rem ? 1 : 0);
            }
        }
        else if (partition_mode_ == PartitionMode::RankZero)
        {
            // Debug mode: all rows on rank 0; other ranks own the empty range
            // [rows, rows-1]. Hypre/BoomerAMG treat empty-range ranks as
            // "participate in collectives but own no data". No permutation.
            rank_ilower_[0] = 0;
            rank_iupper_[0] = rows - 1;
            for (int r = 1; r < mpi_size_; ++r) {
                rank_ilower_[r] = rows;       // ilower > iupper → empty
                rank_iupper_[r] = rows - 1;
            }
        }
        else // PartitionMode::Metis
        {
#ifndef POLYSOLVE_WITH_METIS
            // Should never reach here: read_partition_mode_from_env downgrades
            // to RowBlock when METIS isn't compiled in.
            std::fprintf(stderr, "polysolve: METIS path requested but not compiled.\n");
            std::abort();
#else
            // ============================================================
            //  METIS partition: minimize cut edges across MPI ranks.
            //
            //  Step 1. Build the matrix's adjacency graph in METIS CSR form
            //          (xadj[N+1], adjncy[NNZ]). METIS expects an undirected
            //          graph with no self-loops, so we skip diagonal entries.
            //          The stiffness matrix is symmetric so we don't need to
            //          symmetrize it explicitly.
            //
            //  Step 2. On rank 0, call METIS_PartGraphKway(nparts = mpi_size_)
            //          to get part[i] = rank that should own vertex i.
            //          Broadcast part[] to all ranks so they agree.
            //
            //  Step 3. From part[], construct perm_ / inv_perm_ such that
            //          rank r owns the contiguous range
            //              [sum_{r'<r} count[r'],
            //               sum_{r'<=r} count[r'] - 1].
            //          Within a rank, sort vertices by original index for
            //          determinism. perm_[i] = new index of original i.
            //
            //  Step 4. Populate rank_ilower_/rank_iupper_.
            //
            //  All cross-rank communication is one MPI_Bcast of an int array
            //  of length N; everything else is local-only.
            // ============================================================

            // --- Step 1: build CSR adjacency (rank 0 only is sufficient,
            // but each rank has the full Ain anyway; we do it everywhere
            // to keep it simple and avoid having to ship the graph).
            const HYPRE_Int N = rows;
            std::vector<idx_t> xadj(N + 1, 0);
            // count non-diagonal nonzeros per row first
            for (HYPRE_Int k = 0; k < Ain.outerSize(); ++k) {
                for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it) {
                    if (it.row() != it.col())
                        xadj[it.row() + 1]++;
                }
            }
            // prefix sum → xadj
            for (HYPRE_Int i = 1; i <= N; ++i)
                xadj[i] += xadj[i - 1];

            std::vector<idx_t> adjncy(xadj[N]);
            std::vector<idx_t> cursor(N, 0); // write head per row
            for (HYPRE_Int k = 0; k < Ain.outerSize(); ++k) {
                for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it) {
                    if (it.row() == it.col()) continue;
                    const idx_t r = static_cast<idx_t>(it.row());
                    adjncy[xadj[r] + cursor[r]++] = static_cast<idx_t>(it.col());
                }
            }

            // --- Step 2: METIS on rank 0, broadcast result
            std::vector<idx_t> part(N, 0);

            if (mpi_rank_ == 0 && mpi_size_ >= 2) {
                idx_t nvtxs   = N;
                idx_t ncon    = 1;             // 1 balance constraint (vertex count)
                idx_t nparts  = mpi_size_;
                idx_t edgecut = 0;             // METIS will fill
                idx_t options[METIS_NOPTIONS];
                METIS_SetDefaultOptions(options);
                options[METIS_OPTION_SEED] = 0; // deterministic across runs

                const int ret = METIS_PartGraphKway(
                    &nvtxs, &ncon, xadj.data(), adjncy.data(),
                    /*vwgt=*/nullptr, /*vsize=*/nullptr, /*adjwgt=*/nullptr,
                    &nparts,
                    /*tpwgts=*/nullptr, /*ubvec=*/nullptr,
                    options, &edgecut, part.data());

                if (ret != METIS_OK) {
                    std::fprintf(stderr,
                        "polysolve: METIS_PartGraphKway failed (ret=%d). "
                        "Falling back to row_block partition for this matrix.\n", ret);
                    // Mark as "all on rank 0" so we at least produce a correct
                    // partition (degraded but not wrong).
                    for (HYPRE_Int i = 0; i < N; ++i) part[i] = 0;
                } else {
                    std::fprintf(stderr,
                        "[polysolve::HypreSolver] METIS edgecut=%lld for "
                        "nvtxs=%lld nparts=%lld\n",
                        (long long)edgecut, (long long)N, (long long)mpi_size_);
                }
            }
            // mpi_size_ == 1 → part stays all-zero, trivial partition.
            // Broadcast (MPI_INT or MPI_LONG_LONG depending on idx_t width).
            static_assert(sizeof(idx_t) == 4 || sizeof(idx_t) == 8,
                          "Unexpected METIS idx_t size");
            const MPI_Datatype idx_mpi_type =
                (sizeof(idx_t) == 4) ? MPI_INT32_T : MPI_INT64_T;
            MPI_Bcast(part.data(), N, idx_mpi_type, 0, MPI_COMM_WORLD);

            // --- Step 3: build perm_ / inv_perm_ from part[]
            // Group vertices by rank, preserving original index order within
            // each group. Result: inv_perm_[j] = original index sitting at
            // new position j; perm_[i] = new position of original index i.
            std::vector<int> count_per_rank(mpi_size_, 0);
            for (HYPRE_Int i = 0; i < N; ++i)
                count_per_rank[part[i]]++;

            // Per-rank starting offset in the new global index space.
            std::vector<int> rank_offset(mpi_size_, 0);
            for (int r = 1; r < mpi_size_; ++r)
                rank_offset[r] = rank_offset[r - 1] + count_per_rank[r - 1];

            perm_.assign(N, 0);
            inv_perm_.assign(N, 0);
            std::vector<int> write_cursor = rank_offset;
            for (HYPRE_Int i = 0; i < N; ++i) {
                const int r = part[i];
                const int new_index = write_cursor[r]++;
                perm_[i] = new_index;
                inv_perm_[new_index] = i;
            }

            // --- Step 4: fill rank_ilower_/rank_iupper_
            for (int r = 0; r < mpi_size_; ++r) {
                rank_ilower_[r] = rank_offset[r];
                rank_iupper_[r] = rank_offset[r] + count_per_rank[r] - 1;
                // If a rank has 0 vertices (extreme edge case), encode empty.
                if (count_per_rank[r] == 0) {
                    rank_ilower_[r] = N;
                    rank_iupper_[r] = N - 1;
                }
            }
#endif
        }

        ilower_ = rank_ilower_[mpi_rank_];
        iupper_ = rank_iupper_[mpi_rank_];

        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower_, iupper_, ilower_, iupper_, &A);
#else
        HYPRE_IJMatrixCreate(0, 0, rows - 1, 0, cols - 1, &A);
#endif
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);

        // Insert matrix entries. When METIS is active (perm_ is non-empty),
        // every Eigen index (it.row(), it.col()) is rewritten to its permuted
        // index (perm_[it.row()], perm_[it.col()]) — this is the
        // P · A · P^T operation. When perm_ is empty (RowBlock / RankZero),
        // the lookups become the identity and we insert entries unchanged.
        const bool use_perm = !perm_.empty();
        for (HYPRE_Int k = 0; k < Ain.outerSize(); ++k)
        {
            for (StiffnessMatrix::InnerIterator it(Ain, k); it; ++it)
            {
                const HYPRE_Int row_new = use_perm ? perm_[it.row()] : it.row();
                const HYPRE_Int col_new = use_perm ? perm_[it.col()] : it.col();
#ifdef HYPRE_WITH_MPI
                // Skip rows this rank doesn't own (other ranks will insert them).
                if (row_new < ilower_ || row_new > iupper_)
                    continue;
#endif
                const HYPRE_Int i[1] = {row_new};
                const HYPRE_Int j[1] = {col_new};
                const HYPRE_Complex v[1] = {it.value()};
                HYPRE_Int n_cols[1] = {1};

                HYPRE_IJMatrixSetValues(A, 1, n_cols, i, j, v);
            }
        }

        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
    }

    ////////////////////////////////////////////////////////////////////////////////

    namespace
    {

        void HypreBoomerAMG_SetDefaultOptions(HYPRE_Solver &amg_precond)
        {
            // AMG coarsening options:
            int coarsen_type = 10; // 10 = HMIS, 8 = PMIS, 6 = Falgout, 0 = CLJP
            int agg_levels = 1;    // number of aggressive coarsening levels
            double theta = 0.25;   // strength threshold: 0.25, 0.5, 0.8

            // AMG interpolation options:
            int interp_type = 6; // 6 = extended+i, 0 = classical
            int Pmax = 4;        // max number of elements per row in P

            // AMG relaxation options:
            int relax_type = 8;   // 8 = l1-GS, 6 = symm. GS, 3 = GS, 18 = l1-Jacobi
            int relax_sweeps = 1; // relaxation sweeps on each level

            // Additional options:
            int print_level = 0; // print AMG iterations? 1 = no, 2 = yes
            int max_levels = 25; // max number of levels in AMG hierarchy

            HYPRE_BoomerAMGSetCoarsenType(amg_precond, coarsen_type);
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, agg_levels);
            HYPRE_BoomerAMGSetRelaxType(amg_precond, relax_type);
            HYPRE_BoomerAMGSetNumSweeps(amg_precond, relax_sweeps);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, theta);
            HYPRE_BoomerAMGSetInterpType(amg_precond, interp_type);
            HYPRE_BoomerAMGSetPMaxElmts(amg_precond, Pmax);
            HYPRE_BoomerAMGSetPrintLevel(amg_precond, print_level);
            HYPRE_BoomerAMGSetMaxLevels(amg_precond, max_levels);

            // Use as a preconditioner (one V-cycle, zero tolerance)
            HYPRE_BoomerAMGSetMaxIter(amg_precond, 1);
            HYPRE_BoomerAMGSetTol(amg_precond, 0.0);
        }

        void HypreBoomerAMG_SetElasticityOptions(HYPRE_Solver &amg_precond, int dim)
        {
            // Make sure the systems AMG options are set
            HYPRE_BoomerAMGSetNumFunctions(amg_precond, dim);

            // More robust options with respect to convergence
            HYPRE_BoomerAMGSetAggNumLevels(amg_precond, 0);
            HYPRE_BoomerAMGSetStrongThreshold(amg_precond, 0.5);

            // Nodal coarsening options (nodal coarsening is required for this solver)
            // See hypre's new_ij driver and the paper for descriptions.
            int nodal = 4;        // strength reduction norm: 1, 3 or 4
            int nodal_diag = 1;   // diagonal in strength matrix: 0, 1 or 2
            int relax_coarse = 8; // smoother on the coarsest grid: 8, 99 or 29

            // Elasticity interpolation options
            int interp_vec_variant = 2;    // 1 = GM-1, 2 = GM-2, 3 = LN
            int q_max = 4;                 // max elements per row for each Q
            int smooth_interp_vectors = 1; // smooth the rigid-body modes?

            // Optionally pre-process the interpolation matrix through iterative weight
            // refinement (this is generally applicable for any system)
            int interp_refine = 1;

            HYPRE_BoomerAMGSetNodal(amg_precond, nodal);
            HYPRE_BoomerAMGSetNodalDiag(amg_precond, nodal_diag);
            HYPRE_BoomerAMGSetCycleRelaxType(amg_precond, relax_coarse, 3);
            HYPRE_BoomerAMGSetInterpVecVariant(amg_precond, interp_vec_variant);
            HYPRE_BoomerAMGSetInterpVecQMax(amg_precond, q_max);
            // HYPRE_BoomerAMGSetSmoothInterpVectors(amg_precond, smooth_interp_vectors);
            // HYPRE_BoomerAMGSetInterpRefine(amg_precond, interp_refine);

            // RecomputeRBMs();
            // HYPRE_BoomerAMGSetInterpVectors(amg_precond, rbms.Size(), rbms.GetData());
        }

    } // anonymous namespace

    ////////////////////////////////////////////////////////////////////////////////

    void HypreSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        POLYSOLVE_SCOPED_STOPWATCH("solve", total_time, *logger);
        HYPRE_IJVector b;
        HYPRE_ParVector par_b;
        HYPRE_IJVector x;
        HYPRE_ParVector par_x;

#ifdef HYPRE_WITH_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower_, iupper_, &b);
#else
        HYPRE_IJVectorCreate(0, 0, rhs.size() - 1, &b);
#endif
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b);
#ifdef HYPRE_WITH_MPI
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower_, iupper_, &x);
#else
        HYPRE_IJVectorCreate(0, 0, rhs.size() - 1, &x);
#endif
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x);

        assert(result.size() == rhs.size());

#ifdef HYPRE_WITH_MPI
        // local_n = number of rows owned by this rank in the (possibly permuted)
        // numbering. Empty range encoded as iupper_ < ilower_ → local_n = 0.
        const HYPRE_Int local_n = (iupper_ >= ilower_) ? (iupper_ - ilower_ + 1) : 0;

        // When METIS is active, we have to ship Hypre the *permuted* b and the
        // *permuted* initial guess. perm_/inv_perm_ are empty for RowBlock and
        // RankZero, so the branch below collapses to the original code path.
        std::vector<HYPRE_Complex> b_local;   // permuted slice [ilower_, iupper_]
        std::vector<HYPRE_Complex> x_local;
        if (local_n > 0) {
            if (perm_.empty()) {
                // No permutation: rhs/result already in matching index space.
                HYPRE_IJVectorSetValues(b, local_n, nullptr, rhs.data()    + ilower_);
                HYPRE_IJVectorSetValues(x, local_n, nullptr, result.data() + ilower_);
            } else {
                // b'[j] = rhs[inv_perm_[j]], x'[j] = result[inv_perm_[j]],
                // for j in [ilower_, iupper_].
                b_local.resize(local_n);
                x_local.resize(local_n);
                for (HYPRE_Int k = 0; k < local_n; ++k) {
                    const int orig = inv_perm_[ilower_ + k];
                    b_local[k] = rhs(orig);
                    x_local[k] = result(orig);
                }
                HYPRE_IJVectorSetValues(b, local_n, nullptr, b_local.data());
                HYPRE_IJVectorSetValues(x, local_n, nullptr, x_local.data());
            }
        }
#else
        HYPRE_IJVectorSetValues(b, rhs.size(), nullptr, rhs.data());
        HYPRE_IJVectorSetValues(x, rhs.size(), nullptr, result.data());
#endif

        HYPRE_IJVectorAssemble(b);
        HYPRE_IJVectorGetObject(b, (void **)&par_b);

        HYPRE_IJVectorAssemble(x);
        HYPRE_IJVectorGetObject(x, (void **)&par_x);

        /* Krylov solver + preconditioner.
         * Two Krylov paths, picked by krylov_type_:
         *   PCG   — 3-term Lanczos, requires SPD M; has Hypre's
         *           "subnormal gamma" bail-out (pcg.c:709) on near-singular A.
         *   GMRES — Arnoldi, no subnormal trap, accepts non-SPD M. Uses more
         *           memory (gmres_kdim_ × N vectors).
         */
        HYPRE_Solver solver, precond;
        const MPI_Comm krylov_comm =
#ifdef HYPRE_WITH_MPI
            MPI_COMM_WORLD;
#else
            0;
#endif

        if (krylov_type_ == KrylovType::PCG)
        {
            HYPRE_ParCSRPCGCreate(krylov_comm, &solver);
            HYPRE_PCGSetMaxIter(solver, max_iter_);
            HYPRE_PCGSetTol(solver, conv_tol_);
            HYPRE_PCGSetTwoNorm(solver, 1);     // PCG-specific: use 2-norm for stop test
            HYPRE_PCGSetLogging(solver, 1);
        }
        else // GMRES
        {
            HYPRE_ParCSRGMRESCreate(krylov_comm, &solver);
            HYPRE_GMRESSetMaxIter(solver, max_iter_);
            HYPRE_GMRESSetTol(solver, conv_tol_);
            HYPRE_GMRESSetKDim(solver, gmres_kdim_); // GMRES-specific: restart dim
            HYPRE_GMRESSetLogging(solver, 1);
        }

        // ============================================================
        // Set up the preconditioner. Two paths:
        //   BoomerAMG (default) — classical AMG, good on well-conditioned
        //                         elliptic PDEs, can break down on near-
        //                         singular contact matrices.
        //   Euclid              — parallel ILU(k); more robust on ill-
        //                         conditioned matrices because M⁻¹ is a
        //                         triangular solve rather than a multigrid
        //                         cycle, so (r, M⁻¹r) doesn't collapse.
        // ============================================================
        // Build the preconditioner. Same precond object plugs into both PCG
        // and GMRES; only the SetPrecond function name differs.
        HYPRE_PtrToSolverFcn precond_solve_fn = nullptr;
        HYPRE_PtrToSolverFcn precond_setup_fn = nullptr;
        if (precond_type_ == PreconditionerType::BoomerAMG)
        {
            HYPRE_BoomerAMGCreate(&precond);
            HypreBoomerAMG_SetDefaultOptions(precond);
            if (dimension_ > 1)
            {
                HypreBoomerAMG_SetElasticityOptions(precond, dimension_);
            }
            precond_solve_fn = (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve;
            precond_setup_fn = (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup;
        }
        else // Euclid
        {
            HYPRE_EuclidCreate(krylov_comm, &precond);
            // ILU(k): k = euclid_level_. k=0 is cheapest, k>=1 fills in
            // more entries (better preconditioner, more memory & flops).
            HYPRE_EuclidSetLevel(precond, euclid_level_);
            // Block-Jacobi mode (BJ=1): each rank does its own ILU on its
            // local block, no cross-rank coupling. This is what Hypre's own
            // example ex5big.c uses and what their docs recommend as the
            // robust default for SPD problems. Parallel ILU (BJ=0, the
            // raw Hypre default) couples ranks via off-process pivots and
            // can produce a non-SPD M, which makes PCG misbehave.
            HYPRE_EuclidSetBJ(precond, 1);
            HYPRE_EuclidSetStats(precond, 0);
            precond_solve_fn = (HYPRE_PtrToSolverFcn)HYPRE_EuclidSolve;
            precond_setup_fn = (HYPRE_PtrToSolverFcn)HYPRE_EuclidSetup;
        }

        // Attach precond to whichever Krylov method we picked.
        if (krylov_type_ == KrylovType::PCG) {
            HYPRE_PCGSetPrecond(solver, precond_solve_fn, precond_setup_fn, precond);
        } else {
            HYPRE_GMRESSetPrecond(solver, precond_solve_fn, precond_setup_fn, precond);
        }

        // Drain Hypre's internal error flag from preconditioner setup before
        // running the Krylov method, so the post-solve HYPRE_GetError() below
        // reflects only solve-level errors. Hypre is sticky on errors otherwise.
        (void)HYPRE_GetError();
        HYPRE_ClearAllErrors();

        // Setup + Solve via the chosen Krylov method.
        if (krylov_type_ == KrylovType::PCG)
        {
            HYPRE_ParCSRPCGSetup(solver, parcsr_A, par_b, par_x);
            HYPRE_ParCSRPCGSolve(solver, parcsr_A, par_b, par_x);
            HYPRE_PCGGetNumIterations(solver, &num_iterations);
            HYPRE_PCGGetFinalRelativeResidualNorm(solver, &final_res_norm);
        }
        else // GMRES
        {
            HYPRE_ParCSRGMRESSetup(solver, parcsr_A, par_b, par_x);
            HYPRE_ParCSRGMRESSolve(solver, parcsr_A, par_b, par_x);
            HYPRE_GMRESGetNumIterations(solver, &num_iterations);
            HYPRE_GMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);
        }

        // Surface any Hypre error from this solve (subnormal gamma, max iter,
        // etc.) so it shows up in stderr. Only rank 0 logs, to avoid N copies.
        {
            const HYPRE_Int err = HYPRE_GetError();
            if (err && mpi_rank_ == 0) {
                char msg[256] = {0};
                HYPRE_DescribeError(err, msg);
                std::fprintf(stderr,
                    "[polysolve::HypreSolver] Hypre %s error %lld (%s): "
                    "final_res_norm=%.3e num_iterations=%lld\n",
                    krylov_type_name(krylov_type_),
                    (long long)err, msg,
                    (double)final_res_norm, (long long)num_iterations);
            }
            HYPRE_ClearAllErrors();
        }

        /* Destroy solver and preconditioner */
        if (precond_type_ == PreconditionerType::BoomerAMG) {
            HYPRE_BoomerAMGDestroy(precond);
        } else {
            HYPRE_EuclidDestroy(precond);
        }
        if (krylov_type_ == KrylovType::PCG) {
            HYPRE_ParCSRPCGDestroy(solver);
        } else {
            HYPRE_ParCSRGMRESDestroy(solver);
        }

        assert(result.size() == rhs.size());

#ifdef HYPRE_WITH_MPI
        // Read this rank's local segment of x' (the solution in the permuted
        // numbering). For RowBlock/RankZero this is identity; for METIS we
        // un-permute it after the Allgatherv below.
        const HYPRE_Int local_size = (iupper_ >= ilower_) ? (iupper_ - ilower_ + 1) : 0;

        // Staging buffer for x' (size = total rows). We always materialize the
        // full permuted vector so MPI_Allgatherv can fill it; then either copy
        // straight to result (no perm) or scatter via inv_perm_ (METIS).
        std::vector<HYPRE_Complex> x_full(num_rows_, 0.0);

        if (local_size > 0) {
            // Write this rank's slice of x' into the corresponding slot of
            // x_full so we can MPI_Allgatherv in place.
            HYPRE_IJVectorGetValues(x, local_size, nullptr, x_full.data() + ilower_);
        }

        // Build recvcounts/displs from rank_ilower_/rank_iupper_ — these were
        // populated in factorize() and match exactly the partition Hypre saw.
        std::vector<int> recvcounts(mpi_size_, 0);
        std::vector<int> displs(mpi_size_, 0);
        for (int r = 0; r < mpi_size_; ++r) {
            if (rank_iupper_[r] >= rank_ilower_[r]) {
                recvcounts[r] = static_cast<int>(rank_iupper_[r] - rank_ilower_[r] + 1);
                displs[r]     = static_cast<int>(rank_ilower_[r]);
            } else {
                // Empty rank (RankZero mode): zero count; displs value is unused
                // but must be in-bounds for MPI implementations that range-check.
                recvcounts[r] = 0;
                displs[r]     = 0;
            }
        }

        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       x_full.data(), recvcounts.data(), displs.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);

        // Now every rank holds the full x'. Copy back to result, undoing the
        // METIS permutation if needed: result[i] = x'[perm_[i]].
        if (perm_.empty()) {
            std::copy(x_full.begin(), x_full.end(), result.data());
        } else {
            for (HYPRE_Int i = 0; i < num_rows_; ++i) {
                result(i) = x_full[perm_[i]];
            }
        }
#else
        HYPRE_IJVectorGetValues(x, rhs.size(), nullptr, result.data());
#endif

        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
    }

    ////////////////////////////////////////////////////////////////////////////////

    HypreSolver::~HypreSolver()
    {
        if (has_matrix_)
        {
            HYPRE_IJMatrixDestroy(A);
            has_matrix_ = false;
        }
    }

} // namespace polysolve::linear

#endif
