#pragma once

#include <Eigen/Sparse>

#include <fstream>
#include <iostream>
#include <string>
#include <stdexcept>
#include <limits>
#include <algorithm>

#include <Eigen/Core>
#include <vector>

// Include polysolve types for StiffnessMatrix
#include "polysolve/Types.hpp"

using namespace Eigen;

namespace benchy {
namespace io {

extern std::string mat_save_global;
extern int ts_global;
extern int iter_global;
extern int dim_global;

typedef Triplet<int> Trip;

// template <typename T, int whatever, typename IND>
// void Serialize(const SparseMatrix<T, whatever, IND>& mat, const int dim, const int is_symmetric_positive_definite, const int is_sequence_of_problems, std::string filename) {
//     std::vector<Trip> res;
//     int sz = mat.nonZeros();
//     SparseMatrix<T, whatever, IND> m=mat;
//     m.makeCompressed();

//     std::fstream writeFile;
//     writeFile.open(filename, std::ios::binary | std::ios::out);

//     if(writeFile.is_open())
//     {
//         writeFile.write((const char *)&(dim), sizeof(int));
//         writeFile.write((const char *)&(is_symmetric_positive_definite), sizeof(int));
//         writeFile.write((const char *)&(is_sequence_of_problems), sizeof(int));

//         IND rows, cols, nnzs, outS, innS;
//         rows = m.rows()     ;
//         cols = m.cols()     ;
//         nnzs = m.nonZeros() ;
//         outS = m.outerSize();
//         innS = m.innerSize();

//         writeFile.write((const char *)&(rows), sizeof(IND));
//         writeFile.write((const char *)&(cols), sizeof(IND));
//         writeFile.write((const char *)&(nnzs), sizeof(IND));
//         writeFile.write((const char *)&(innS), sizeof(IND));
//         writeFile.write((const char *)&(outS), sizeof(IND));

//         writeFile.write((const char *)(m.valuePtr()),       sizeof(T  ) * m.nonZeros());
//         writeFile.write((const char *)(m.outerIndexPtr()),  sizeof(IND) * m.outerSize());
//         writeFile.write((const char *)(m.innerIndexPtr()),  sizeof(IND) * m.nonZeros());

//         writeFile.close();
//     }
// }

// template <typename T, int whatever, typename IND>
// void Deserialize(SparseMatrix<T, whatever, IND>& m, int& dim, int& is_symmetric_positive_definite, int& is_sequence_of_problems, std::string filename) {
//     std::fstream readFile;
//     readFile.open(filename, std::ios::binary | std::ios::in);
//     if(readFile.is_open())
//     {
//         readFile.read((char*)&dim, sizeof(int));
//         readFile.read((char*)&is_symmetric_positive_definite, sizeof(int));
//         readFile.read((char*)&is_sequence_of_problems, sizeof(int));

//         IND rows, cols, nnz, inSz, outSz;
//         readFile.read((char*)&rows , sizeof(IND));
//         readFile.read((char*)&cols , sizeof(IND));
//         readFile.read((char*)&nnz  , sizeof(IND));
//         readFile.read((char*)&inSz , sizeof(IND));
//         readFile.read((char*)&outSz, sizeof(IND));

//         m.resize(rows, cols);
//         m.makeCompressed();
//         m.resizeNonZeros(nnz);

//         readFile.read((char*)(m.valuePtr())     , sizeof(T  ) * nnz  );
//         readFile.read((char*)(m.outerIndexPtr()), sizeof(IND) * outSz);
//         readFile.read((char*)(m.innerIndexPtr()), sizeof(IND) * nnz );

//         m.finalize();
//         readFile.close();

//     } // file is open
// }

// StiffnessMatrix versions - automatically handles large index based on POLYSOLVE_LARGE_INDEX
inline void SerializeStiffnessMatrix(const polysolve::StiffnessMatrix& mat, const int dim, const int is_symmetric_positive_definite, const int is_sequence_of_problems, std::string filename) {
    polysolve::StiffnessMatrix m = mat;
    m.makeCompressed();

    std::fstream writeFile;
    writeFile.open(filename, std::ios::binary | std::ios::out);

    if(writeFile.is_open())
    {
        writeFile.write((const char *)&(dim), sizeof(int));
        writeFile.write((const char *)&(is_symmetric_positive_definite), sizeof(int));
        writeFile.write((const char *)&(is_sequence_of_problems), sizeof(int));

#ifdef POLYSOLVE_LARGE_INDEX
        // Add format marker for ptrdiff_t format
        int32_t format_marker = -1;
        writeFile.write((const char *)&format_marker, sizeof(int32_t));
        
        typedef std::ptrdiff_t IndexType;
#else
        typedef int IndexType;
#endif

        IndexType rows = static_cast<IndexType>(m.rows());
        IndexType cols = static_cast<IndexType>(m.cols());
        IndexType nnzs = static_cast<IndexType>(m.nonZeros());
        IndexType outS = static_cast<IndexType>(m.outerSize());
        IndexType innS = static_cast<IndexType>(m.innerSize());

        writeFile.write((const char *)&(rows), sizeof(IndexType));
        writeFile.write((const char *)&(cols), sizeof(IndexType));
        writeFile.write((const char *)&(nnzs), sizeof(IndexType));
        writeFile.write((const char *)&(innS), sizeof(IndexType));
        writeFile.write((const char *)&(outS), sizeof(IndexType));

        writeFile.write((const char *)(m.valuePtr()),       sizeof(double) * m.nonZeros());
        writeFile.write((const char *)(m.outerIndexPtr()),  sizeof(IndexType) * m.outerSize());
        writeFile.write((const char *)(m.innerIndexPtr()),  sizeof(IndexType) * m.nonZeros());

        writeFile.close();
    }
}

inline void DeserializeStiffnessMatrix(polysolve::StiffnessMatrix& m, int& dim, int& is_symmetric_positive_definite, int& is_sequence_of_problems, std::string filename) {
    std::fstream readFile;
    readFile.open(filename, std::ios::binary | std::ios::in);
    if(readFile.is_open())
    {
        readFile.read((char*)&dim, sizeof(int));
        readFile.read((char*)&is_symmetric_positive_definite, sizeof(int));
        readFile.read((char*)&is_sequence_of_problems, sizeof(int));

#ifdef POLYSOLVE_LARGE_INDEX
        // Check if this is a new format file with ptrdiff_t indices
        int32_t format_check;
        readFile.read((char*)&format_check, sizeof(int32_t));
        
        if (format_check == -1) {
            // This is a ptrdiff_t format file (new format)
            typedef std::ptrdiff_t IndexType;
            IndexType rows, cols, nnz, inSz, outSz;
            readFile.read((char*)&rows , sizeof(IndexType));
            readFile.read((char*)&cols , sizeof(IndexType));
            readFile.read((char*)&nnz  , sizeof(IndexType));
            readFile.read((char*)&inSz , sizeof(IndexType));
            readFile.read((char*)&outSz, sizeof(IndexType));

            m.resize(rows, cols);
            m.makeCompressed();
            m.resizeNonZeros(nnz);

            readFile.read((char*)(m.valuePtr()),      sizeof(double) * nnz);
            readFile.read((char*)(m.outerIndexPtr()), sizeof(IndexType) * outSz);
            readFile.read((char*)(m.innerIndexPtr()), sizeof(IndexType) * nnz);

            m.finalize();
        } else {
            // This is an int32_t format file (legacy format)
            int32_t rows32 = format_check;
            int32_t cols32, nnz32, inSz32, outSz32;
            readFile.read((char*)&cols32 , sizeof(int32_t));
            readFile.read((char*)&nnz32  , sizeof(int32_t));
            readFile.read((char*)&inSz32 , sizeof(int32_t));
            readFile.read((char*)&outSz32, sizeof(int32_t));

            // Convert to ptrdiff_t
            std::ptrdiff_t rows = static_cast<std::ptrdiff_t>(rows32);
            std::ptrdiff_t cols = static_cast<std::ptrdiff_t>(cols32);
            std::ptrdiff_t nnz = static_cast<std::ptrdiff_t>(nnz32);
            std::ptrdiff_t inSz = static_cast<std::ptrdiff_t>(inSz32);
            std::ptrdiff_t outSz = static_cast<std::ptrdiff_t>(outSz32);

            m.resize(rows, cols);
            m.makeCompressed();
            m.resizeNonZeros(nnz);

            readFile.read((char*)(m.valuePtr()), sizeof(double) * nnz);

            // Read index arrays as int32_t and convert
            std::vector<int32_t> outerIndex32(outSz);
            std::vector<int32_t> innerIndex32(nnz);
            
            readFile.read((char*)outerIndex32.data(), sizeof(int32_t) * outSz);
            readFile.read((char*)innerIndex32.data(), sizeof(int32_t) * nnz);

            // Convert and copy to ptrdiff_t arrays
            std::ptrdiff_t* outerPtr = m.outerIndexPtr();
            std::ptrdiff_t* innerPtr = m.innerIndexPtr();
            
            for (std::ptrdiff_t i = 0; i < outSz; ++i) {
                outerPtr[i] = static_cast<std::ptrdiff_t>(outerIndex32[i]);
            }
            
            for (std::ptrdiff_t i = 0; i < nnz; ++i) {
                innerPtr[i] = static_cast<std::ptrdiff_t>(innerIndex32[i]);
            }

            m.finalize();
        }
#else
        // For regular build, read as int directly
        typedef int IndexType;
        IndexType rows, cols, nnz, inSz, outSz;
        
        // In non-POLYSOLVE_LARGE_INDEX mode, no format marker is written
        // Read dimensions directly
        readFile.read((char*)&rows , sizeof(IndexType));
        readFile.read((char*)&cols , sizeof(IndexType));
        readFile.read((char*)&nnz  , sizeof(IndexType));
        readFile.read((char*)&inSz , sizeof(IndexType));
        readFile.read((char*)&outSz, sizeof(IndexType));

        m.resize(rows, cols);
        m.makeCompressed();
        m.resizeNonZeros(nnz);

        readFile.read((char*)(m.valuePtr())     , sizeof(double) * nnz  );
        readFile.read((char*)(m.outerIndexPtr()), sizeof(IndexType) * outSz);
        readFile.read((char*)(m.innerIndexPtr()), sizeof(IndexType) * nnz );

        m.finalize();
#endif
        
        readFile.close();
    } // file is open
}

template<class Matrix>
void WriteMat(const Matrix& matrix, std::string filename){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename Matrix::Index));
    out.write((char*) (&cols), sizeof(typename Matrix::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
    out.close();
}
template<class Matrix>
void ReadMat(Matrix& matrix, std::string filename){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename Matrix::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename Matrix::Index));
    in.read((char*) (&cols),sizeof(typename Matrix::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
    in.close();
}

///
/// Lightweight class representing a linear system and some metadata.
///
/// @tparam     Scalar    Problem scalar type (float or double).
///
template <typename Scalar>
struct Problem
{
    /// Left-hand side sparse matrix.
    // Eigen::SparseMatrix<Scalar> A;
    polysolve::StiffnessMatrix A;

    /// Right-hand side dense matrix. To save multiple rhs, use separate columns.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> b;

    /// Whether the Hessian was assembled with PSD projection (ProjectedNewton=1, Newton=0).
    int is_projected = 0;
};

///
/// Saves a linear system and associated metadata.
///
/// To save a specific linear system, you can use the following example code:
/// @code
/// benchy::io::Problem<double> problem;
/// problem.A = A;
/// problem.b = b;
/// benchy::io::save_problem(problem);
/// @endcode
///
/// @param[in]  filename  Filename to save the problem to.
/// @param[in]  problem   Container describing the linear system to save.
///
/// @tparam     Scalar    Problem scalar type (float or double).
///
/// @return     True if the problem was successfully saved, False otherwise.
///

template <typename Scalar>
bool save_problem(const Problem<Scalar>& problem)
{
    // Check that all metadata is set
    if (problem.A.size() == 0) {
        std::cerr << "Matrix A is empty" << std::endl;
        return false;
    }
    if (problem.b.size() == 0) {
        std::cerr << "Matrix b is empty" << std::endl;
        return false;
    }
    
    static_assert(
        std::is_same<Scalar, float>::value || std::is_same<Scalar, double>::value,
        "Scalar must be float or double");

    // write binary
    std::string filename1=mat_save_global+"/"+std::to_string(ts_global)+"_"+std::to_string(iter_global)+"_A.bin";
    std::string filename2=mat_save_global+"/"+std::to_string(ts_global)+"_"+std::to_string(iter_global)+"_b.bin";
    std::string filename3=mat_save_global+"/"+std::to_string(ts_global)+"_"+std::to_string(iter_global)+"_meta.bin";

    // write A
    // Serialize(problem.A, dim_global, 1, 1, filename1);
    SerializeStiffnessMatrix(problem.A, dim_global, 1, 1, filename1);

    // write meta (is_projected flag)
    {
        std::ofstream metaFile(filename3, std::ios::binary);
        metaFile.write((const char *)&problem.is_projected, sizeof(int));
    }

    // printf("SERIALIZING A\n");
    // std::cout << "DIM GLOBAL: " << dim_global << std::endl;
    // std::cout << problem.A << std::endl;
    
    // // load A
    // Eigen::SparseMatrix<Scalar> A_;
    // int dim_local = 0;
    // int is_symmetric_positive_definite = 0;
    // int is_sequence_of_problems = 0;
    // Deserialize(A_, dim_local, is_symmetric_positive_definite, is_sequence_of_problems, filename1);

    // printf("DESERIALIZING A\n");
    // std::cout << "DIM LOCAL: " << dim_local <<  " IS_SPD: " << is_symmetric_positive_definite << " IS_SEQ: " << is_sequence_of_problems << std::endl;
    // std::cout << A_ << "\n" << std::endl;

    // write b
    WriteMat(problem.b, filename2);

    // printf("SERIALIZING b\n");
    // std::cout << problem.b << "\n" << std::endl;

    // // load b
    // Eigen::MatrixXd b_;
    // ReadMat(b_, filename2);

    // printf("DESERIALIZING b\n");
    // std::cout << b_ << "\n" << std::endl;

    return true;
}

} // namespace io
} // namespace benchy


