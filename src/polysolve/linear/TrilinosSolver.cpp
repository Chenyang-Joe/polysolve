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
#include <Teuchos_DefaultMpiComm.hpp>
#else
#include <Teuchos_DefaultSerialComm.hpp>
#endif

#include <Teuchos_CommHelpers.hpp>
#include <Tpetra_Export.hpp>
#include <Tpetra_Import.hpp>

namespace polysolve::linear
{
    TrilinosSolver::TrilinosSolver()
    {
#ifdef HAVE_MPI
        // Use polysolve's MPI environment
        int done_already;
        MPI_Initialized(&done_already);
        if (!done_already)
        {
            int argc = 1;
            char name[] = "polysolve";
            char *argv[] = {name};
            char **argvv = &argv[0];
            MPI_Init(&argc, &argvv);
            mpi_initialized_ = true;
        }
        comm_ = Teuchos::rcp(new Teuchos::MpiComm<int>(MPI_COMM_WORLD));
#else
        // No MPI: single-process serial mode
        comm_ = Teuchos::rcp(new Teuchos::SerialComm<int>());
#endif
        
        if (!Tpetra::isInitialized()) {
            int argc = 0;
            char **argv = nullptr;
            Tpetra::initialize(&argc, &argv);
            tpetra_initialized_ = true;
        }
    }
    
    TrilinosSolver::~TrilinosSolver()
    {
        // Explicitly release Tpetra objects before finalizing Kokkos
        A_ = Teuchos::null;
        rowMap_ = Teuchos::null;
        preconditioner_ = Teuchos::null;
        comm_ = Teuchos::null;

        if (tpetra_initialized_) {
            Tpetra::finalize();
        }
        
#ifdef HAVE_MPI
        if (mpi_initialized_) {
            MPI_Finalize();
        }
#endif
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
            // if (comm_->getRank() == 0) {
            //     std::cerr << "[TrilinosSolver] numPDEs=" << numPDEs
            //               << " max_iter=" << max_iter_ << " conv_tol=" << conv_tol_
            //               << " is_nullspace=" << is_nullspace_ << std::endl;
            // }
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
        // TODO: distributed loading. Currently every MPI process receives the
        // FULL Eigen matrix `Ain` and uses Tpetra::Export/Import below to push
        // entries into its locally-owned rows. This means total memory =
        // P * sizeof(full matrix), so memory does NOT scale down with more
        // processes. For matrices approaching per-node RAM limits, the caller
        // should instead load only its local row range (e.g. via parallel I/O
        // or rank-0 scatter) and pass a pre-partitioned matrix here.
        POLYSOLVE_SCOPED_STOPWATCH("factorize", total_time, *logger);

        // 1. Setup distributed row map (DOF-aligned for MueLu)
        GlobalOrdinal numGlobalRows = Ain.rows();
        const GlobalOrdinal indexBase = 0;

        if (numPDEs > 1 && numGlobalRows % numPDEs == 0) {
            // Create a DOF-aligned map: partition by nodes so each rank's
            // local row count is a multiple of numPDEs. This prevents MueLu's
            // aggregation from splitting a node's DOFs across ranks.
            GlobalOrdinal numNodes = numGlobalRows / numPDEs;
            Teuchos::RCP<const Map> nodeMap = Teuchos::rcp(new Map(numNodes, indexBase, comm_));
            size_t numLocalNodes = nodeMap->getLocalNumElements();
            size_t numLocalDofs = numLocalNodes * numPDEs;
            std::vector<GlobalOrdinal> myDofs(numLocalDofs);
            for (size_t n = 0; n < numLocalNodes; ++n) {
                GlobalOrdinal gNode = nodeMap->getGlobalElement(n);
                for (int d = 0; d < numPDEs; ++d) {
                    myDofs[n * numPDEs + d] = gNode * numPDEs + d;
                }
            }
            rowMap_ = Teuchos::rcp(new Map(numGlobalRows,
                Teuchos::ArrayView<const GlobalOrdinal>(myDofs.data(), numLocalDofs),
                indexBase, comm_));
        } else if (numPDEs > 1) {
            throw std::runtime_error(
                "TrilinosSolver: numGlobalRows=" + std::to_string(numGlobalRows)
                + " is not divisible by numPDEs=" + std::to_string(numPDEs)
                + ". Check block_size setting.");
        } else {
            // Scalar problem (numPDEs == 1): use default map
            rowMap_ = Teuchos::rcp(new Map(numGlobalRows, indexBase, comm_));
        }

        // 2. Convert Eigen matrix to RowMajor for efficient row access
        //    Every rank has the full Ain (read from file), so each rank
        //    fills only its own local rows — no Export needed.
        using EigenStorageIndex = typename StiffnessMatrix::StorageIndex;
        Eigen::SparseMatrix<double, Eigen::RowMajor, EigenStorageIndex> Arow(Ain);

        // 3. Each rank fills its local rows directly
        size_t numLocalRows = rowMap_->getLocalNumElements();
        Teuchos::ArrayRCP<size_t> nnzPerRow(numLocalRows);
        for (size_t lr = 0; lr < numLocalRows; ++lr) {
            GlobalOrdinal gr = rowMap_->getGlobalElement(lr);
            if (gr < Arow.outerSize()) {
                nnzPerRow[lr] = Arow.outerIndexPtr()[gr + 1] - Arow.outerIndexPtr()[gr];
            } else {
                nnzPerRow[lr] = 0;
            }
        }

        A_ = Teuchos::rcp(new CrsMatrix(rowMap_, nnzPerRow()));

        for (size_t lr = 0; lr < numLocalRows; ++lr) {
            GlobalOrdinal gr = rowMap_->getGlobalElement(lr);
            if (gr >= Arow.outerSize()) continue;

            auto start = Arow.outerIndexPtr()[gr];
            auto end = Arow.outerIndexPtr()[gr + 1];
            int numEntries = static_cast<int>(end - start);
            if (numEntries == 0) continue;

            const double* values_ptr = Arow.valuePtr() + start;
            const auto* indices_ptr = Arow.innerIndexPtr() + start;

            std::vector<GlobalOrdinal> col_indices(numEntries);
            for (int k = 0; k < numEntries; ++k)
                col_indices[k] = static_cast<GlobalOrdinal>(indices_ptr[k]);

            Teuchos::ArrayView<const double> valView(values_ptr, numEntries);
            Teuchos::ArrayView<const GlobalOrdinal> idxView(col_indices.data(), numEntries);

            A_->insertGlobalValues(gr, idxView, valView);
        }

        A_->fillComplete();

        // Preconditioner Setup (MueLu)
        Teuchos::ParameterList mueLuParams;
        mueLuParams.set("verbosity", "none");
        mueLuParams.set("smoother: type", "CHEBYSHEV");

        // Set numPDEs if applicable (elasticity)
        if (numPDEs > 1) {
            mueLuParams.set("number of equations", numPDEs);
        }
        // if (comm_->getRank() == 0) {
        //     std::cerr << "[TrilinosSolver::factorize] numPDEs=" << numPDEs
        //               << " numGlobalRows=" << numGlobalRows
        //               << " localRows=" << rowMap_->getLocalNumElements()
        //               << " aligned=" << (rowMap_->getLocalNumElements() % numPDEs == 0 ? "yes" : "no")
        //               << std::endl;
        // }

        try {
            auto Aop = Teuchos::rcp_static_cast<Operator>(A_);
            preconditioner_ = MueLu::CreateTpetraPreconditioner(Aop, mueLuParams);
        } catch (const std::exception& e) {
            std::cerr << "MueLu setup failed: " << e.what() << std::endl;
            preconditioner_ = Teuchos::null;
        }
    }

    void TrilinosSolver::solve(const Eigen::Ref<const VectorXd> rhs, Eigen::Ref<VectorXd> result)
    {
        POLYSOLVE_SCOPED_STOPWATCH("solve", total_time, *logger);
        
        if (A_ == Teuchos::null) throw std::runtime_error("Matrix not factorized");

        GlobalOrdinal numGlobalRows = rowMap_->getGlobalNumElements();

        // 1. Prepare Source Map (Rank 0) for B and X
        size_t numLocalSource = (comm_->getRank() == 0) ? numGlobalRows : 0;
        Teuchos::RCP<Map> sourceMap = Teuchos::rcp(new Map(numGlobalRows, numLocalSource, 0, comm_));

        // 2. Create Source Vector B (on Rank 0)
        Teuchos::RCP<MultiVector> B_source = Teuchos::rcp(new MultiVector(sourceMap, 1));
        
        if (comm_->getRank() == 0) {
            auto b_data = B_source->getDataNonConst(0);
            for (size_t i = 0; i < (size_t)rhs.size(); ++i) {
                b_data[i] = rhs(i);
            }
        }

        // 3. Distribute B to Target Vector (on rowMap_)
        Teuchos::RCP<MultiVector> B_target = Teuchos::rcp(new MultiVector(rowMap_, 1));
        Tpetra::Export<LocalOrdinal, GlobalOrdinal, Node> exporter(sourceMap, rowMap_);
        B_target->doExport(*B_source, exporter, Tpetra::INSERT);

        // 4. Create Target X (initial guess 0)
        Teuchos::RCP<MultiVector> X_target = Teuchos::rcp(new MultiVector(rowMap_, 1));
        X_target->putScalar(0.0);

        // 5. Setup Belos Solver
        Teuchos::RCP<Belos::LinearProblem<Scalar, MultiVector, Operator>> problem =
            Teuchos::rcp(new Belos::LinearProblem<Scalar, MultiVector, Operator>(A_, X_target, B_target));

        if (preconditioner_ != Teuchos::null) {
            problem->setRightPrec(preconditioner_);
        }
        
        bool set = problem->setProblem();
        if (!set) {
             throw std::runtime_error("Belos::LinearProblem failed to set up");
        }

        // Use ParameterList for Belos
        Teuchos::ParameterList belosList;
        belosList.set("Maximum Iterations", max_iter_);
        belosList.set("Convergence Tolerance", conv_tol_);
        belosList.set("Verbosity", Belos::Errors + Belos::Warnings);
        // belosList.set("Output Frequency", 50);

        Teuchos::RCP<Belos::SolverManager<Scalar, MultiVector, Operator>> solver =
            Teuchos::rcp(new Belos::BlockGmresSolMgr<Scalar, MultiVector, Operator>(problem, Teuchos::rcp(&belosList, false)));

        Belos::ReturnType ret = solver->solve();
        
        iterations_ = solver->getNumIters();
        residual_error_ = solver->achievedTol();

        // 6. Gather Solution X Back to Rank 0
        Teuchos::RCP<MultiVector> X_source_final = Teuchos::rcp(new MultiVector(sourceMap, 1));
        Tpetra::Import<LocalOrdinal, GlobalOrdinal, Node> importer(rowMap_, sourceMap);
        X_source_final->doImport(*X_target, importer, Tpetra::INSERT);

        // 7. Copy to result (on Rank 0) and Broadcast to all ranks
        if (result.size() != numGlobalRows) {
            result.resize(numGlobalRows);
        }

        if (comm_->getRank() == 0) {
            auto x_data = X_source_final->getData(0);
            for (size_t i = 0; i < (size_t)numGlobalRows; ++i) {
               result(i) = x_data[i];
            }
        }
        
        // Broadcast solution to all processes
        Teuchos::broadcast(*comm_, 0, static_cast<int>(result.size()), result.data());
    }
}
