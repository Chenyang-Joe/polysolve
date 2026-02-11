# Trilinos Epetra to Tpetra Migration

This document details the migration of `TrilinosSolver` from the deprecated **Epetra** stack to the newer **Tpetra** stack within the `polysolve` library.

## Changes Overview

The migration replaces the following components:

| Feature | Old Component (Epetra) | New Component (Tpetra) |
| :--- | :--- | :--- |
| **Linear Algebra** | `Epetra_Map`, `Epetra_CrsMatrix`, `Epetra_Vector` | `Tpetra::Map`, `Tpetra::CrsMatrix`, `Tpetra::MultiVector` |
| **Linear Solver** | `AztecOO` | `Belos` |
| **Preconditioner** | `ML` (Multi-Level) | `MueLu` |
| **Communicator** | `Epetra_Comm` (`Epetra_MpiComm` / `Epetra_SerialComm`) | `Teuchos::Comm` (via `Tpetra::getDefaultComm()`) |

## Build System Updates

- **Dependencies**: The `CMakeLists.txt` now searches for `Trilinos` components: `Tpetra`, `Belos`, and `MueLu` instead of `Epetra` and `ML`.
- **Path Configuration**: The `Trilinos_DIR` and `CMAKE_PREFIX_PATH` have been updated to point to the correct Trilinos installation (currently set to a local user path, which may need adjustment for other environments).
- **PolySolveOptions**: Optional solvers (Cholmod, UmfPack, SuperLU) have been disabled in `PolySolveOptions.cmake` to resolve build conflicts in the current environment, but can be re-enabled if dependencies are met.

## Implementation Details

### `TrilinosSolver` Class

- **Initialization**: The solver now initializes the `Teuchos::Comm` using `Tpetra::getDefaultComm()`, which automatically handles MPI or Serial environments.
- **Factorization**: 
    - Converts the input `Eigen::SparseMatrix` to a `Tpetra::CrsMatrix`.
    - Sets up the `MueLu` preconditioner hierarchy using a parameter list similar to the previous ML configuration (Smoother: Chebyshev, Aggregation: Uncoupled).
- **Solve**: 
    - Uses `Belos::BlockCGSolMgr` (Conjugate Gradient) or `Belos::BlockGmresSolMgr` (GMRES) based on configuration (currently hardcoded to BlockCG in the adapter, but expandable).
    - Wraps data in `Tpetra::MultiVector` for solution `X` and right-hand side `B`.

## Usage

The usage of `TrilinosSolver` remains unchanged from the client perspective. Parameters passed via JSON are mapped to the new solver settings where applicable.

```cpp
auto solver = polysolve::linear::Solver::create("Trilinos", "");
solver->set_parameters(params);
solver->analyze_pattern(A, precond_num);
solver->factorize(A);
solver->solve(b, x);
```

## Troubleshooting

- **Build Errors**: 
    - Ensure a compatible Trilinos version (with Tpetra, Belos, MueLu enabled) is installed and `Trilinos_DIR` is correctly set in `CMakeLists.txt`.
    - If `mpi.h` is missing, ensure `find_package(MPI)` is called and `MPI::MPI_CXX` is linked, or configure CMake with `-DCMAKE_CXX_COMPILER=mpicxx` and `-DCMAKE_C_COMPILER=mpicc`.
    - If `SuiteSparse` fails with missing `OpenMP::OpenMP_C`, ensure C language is enabled in `project(PolySolve ... LANGUAGES CXX C)`.
- **Runtime Errors**: If MPI is used, ensure `MPI_Init` is called before the solver is used (the class handles basic init if check fails, but external control is preferred).
