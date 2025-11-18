#pragma once

#include <Eigen/Sparse>

#include <fstream>
#include <iostream>
#include <string>

#include <Eigen/Core>
#include <vector>

#include <Eigen/Dense>
// #include <tinyxml2.h>

#include <filesystem>

using namespace Eigen;

namespace benchy {
namespace io {

extern std::string mat_save_global;
extern int ts_global;
extern int iter_global;
extern int dim_global;

typedef Triplet<int> Trip;

template <typename T, int _Options, typename IND>
void Serialize(const SparseMatrix<T, _Options, IND>& mat, const int dim, const int is_symmetric_positive_definite, const int is_sequence_of_problems, std::string filename) {
    std::filesystem::path out_path(filename);
    std::filesystem::create_directories(out_path.parent_path());
    
    std::cout << "write to "<<filename<<std::endl;
    
    // std::vector<Trip> res;
    // int sz = mat.nonZeros();
    SparseMatrix<T, _Options, IND> m=mat;
    m.makeCompressed();

    std::fstream writeFile;
    writeFile.open(filename, std::ios::binary | std::ios::out);

    if(writeFile.is_open())
    {
        writeFile.write((const char *)&(dim), sizeof(int));
        writeFile.write((const char *)&(is_symmetric_positive_definite), sizeof(int));
        writeFile.write((const char *)&(is_sequence_of_problems), sizeof(int));

        IND rows, cols, nnzs, outS, innS;
        rows = m.rows()     ;
        cols = m.cols()     ;
        nnzs = m.nonZeros() ;
        outS = m.outerSize();
        innS = m.innerSize();

        writeFile.write((const char *)&(rows), sizeof(IND));
        writeFile.write((const char *)&(cols), sizeof(IND));
        writeFile.write((const char *)&(nnzs), sizeof(IND));
        writeFile.write((const char *)&(innS), sizeof(IND));
        writeFile.write((const char *)&(outS), sizeof(IND));

        writeFile.write((const char *)(m.valuePtr()),       sizeof(T  ) * m.nonZeros());
        writeFile.write((const char *)(m.outerIndexPtr()),  sizeof(IND) * m.outerSize());
        writeFile.write((const char *)(m.innerIndexPtr()),  sizeof(IND) * m.nonZeros());

        writeFile.close();
    }
}

template <typename T, int _Options, typename IND>
void Deserialize(SparseMatrix<T, _Options, IND>& m, int& dim, int& is_symmetric_positive_definite, int& is_sequence_of_problems, std::string filename) {
    std::fstream readFile;
    readFile.open(filename, std::ios::binary | std::ios::in);
    if(readFile.is_open())
    {
        readFile.read((char*)&dim, sizeof(int));
        readFile.read((char*)&is_symmetric_positive_definite, sizeof(int));
        readFile.read((char*)&is_sequence_of_problems, sizeof(int));

        IND rows, cols, nnz, inSz, outSz;
        readFile.read((char*)&rows , sizeof(IND));
        readFile.read((char*)&cols , sizeof(IND));
        readFile.read((char*)&nnz  , sizeof(IND));
        readFile.read((char*)&inSz , sizeof(IND));
        readFile.read((char*)&outSz, sizeof(IND));

        m.resize(rows, cols);
        m.makeCompressed();
        m.resizeNonZeros(nnz);

        readFile.read((char*)(m.valuePtr())     , sizeof(T  ) * nnz  );
        readFile.read((char*)(m.outerIndexPtr()), sizeof(IND) * outSz);
        readFile.read((char*)(m.innerIndexPtr()), sizeof(IND) * nnz );

        m.finalize();
        readFile.close();

    } // file is open
}


template<class Matrix>
void WriteMat(const Matrix& matrix, std::string filename){
    std::filesystem::path out_path(filename);
    std::filesystem::create_directories(out_path.parent_path());

    std::cout << "write to "<<filename<<std::endl;
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
    Eigen::SparseMatrix<Scalar> A;

    /// Right-hand side dense matrix. To save multiple rhs, use separate columns.
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> b;

    /// Nullspace
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> nullspace;
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
    std::string filename3=mat_save_global+"/"+std::to_string(ts_global)+"_"+std::to_string(iter_global)+"_nullspace.bin";

    // write A
    Serialize(problem.A, dim_global, 1, 1, filename1);

    // cout dimenssion of A
    std::cout << "Matrix A saved to " << filename1 << std::endl;
    std::cout << "Serialized matrix A with dimensions: " << problem.A.rows() << " x " << problem.A.cols() << " and nnz: " << problem.A.nonZeros() << std::endl;




// ************************************************************************************************************
    // reload the matrix to check the difference
    // Eigen::SparseMatrix<Scalar> A_;
    // int dim_local = 0;
    // int is_symmetric_positive_definite = 0;
    // int is_sequence_of_problems = 0;
    // Deserialize(A_, dim_local, is_symmetric_positive_definite, is_sequence_of_problems, filename1);
    // // traverse the entries and print out the difference entry
    // bool difference_found = false;
    // for (int k=0; k<problem.A.outerSize(); ++k)
    //     for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(problem.A,k); it; ++it)
    //     {
    //         Scalar val1 = it.value();
    //         Scalar val2 = A_.coeff(it.row(), it.col());
    //         if (std::abs(val1 - val2) > 1e-10)
    //         {
    //             std::cout << "Difference at (" << it.row() << ", " << it.col() << "): " << val1 << " vs " << val2 << std::endl;
    //             difference_found = true;
    //         }
    //     }
    // if (!difference_found)
    //     std::cout << "No differences found between original and deserialized matrix A." << std::endl;

// ************************************************************************************************************

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

    // write nullspace
    // WriteMat(problem.nullspace, filename3);

    // printf("SERIALIZING nullspace\n");
    // std::cout << problem.nullspace << "\n" << std::endl;

    // // load b
    // Eigen::MatrixXd nullspace_;
    // ReadMat(nullspace_, filename3);

    // printf("DESERIALIZING nullspace\n");
    // std::cout << nullspace_ << "\n" << std::endl;

    return true;
}

// template<class Matrix>
// void save_vertices(Matrix &pointsMatrix, std::string fname)  // Matrix is Eigen::MatrixXd
// {
//     // std::string fname=mat_save_global+"/"+std::to_string(ts_global)+"_"+std::to_string(iter_global)+"_all_nodes.vtu";

//     // Eigen::MatrixXd pointsMatrix(5, 3);
//     // pointsMatrix << 0.0, 0.0, 0.0,
//     //                 1.0, 0.0, 0.0,
//     //                 0.0, 1.0, 0.0,
//     //                 0.0, 0.0, 1.0,
//     //                 1.0, 1.0, 1.0;

//     tinyxml2::XMLDocument doc;

//     tinyxml2::XMLElement* vtkFile = doc.NewElement("VTKFile");
//     vtkFile->SetAttribute("type", "UnstructuredGrid");
//     vtkFile->SetAttribute("version", "0.1");
//     vtkFile->SetAttribute("byte_order", "LittleEndian");
//     doc.InsertFirstChild(vtkFile);

//     tinyxml2::XMLElement* unstructuredGrid = doc.NewElement("UnstructuredGrid");
//     vtkFile->InsertEndChild(unstructuredGrid);

//     tinyxml2::XMLElement* piece = doc.NewElement("Piece");
//     piece->SetAttribute("NumberOfPoints", static_cast<int>(pointsMatrix.rows()));
//     piece->SetAttribute("NumberOfCells", 0);  // no cell
//     unstructuredGrid->InsertEndChild(piece);

//     tinyxml2::XMLElement* points = doc.NewElement("Points");
//     piece->InsertEndChild(points);

//     tinyxml2::XMLElement* pointDataArray = doc.NewElement("DataArray");
//     pointDataArray->SetAttribute("type", "Float32");
//     pointDataArray->SetAttribute("NumberOfComponents", 3); // 3D coordinates
//     pointDataArray->SetAttribute("format", "ascii");

//     std::string pointData;
//     for (int i = 0; i < pointsMatrix.rows(); ++i) {
//         pointData += std::to_string(pointsMatrix(i, 0)) + " " +
//                      std::to_string(pointsMatrix(i, 1)) + " " +
//                      std::to_string(pointsMatrix(i, 2)) + "\n";
//     }
//     pointDataArray->SetText(pointData.c_str());
//     points->InsertEndChild(pointDataArray);

//     doc.SaveFile(fname.c_str());

//     std::cout << "VTU file with points from Eigen::MatrixXd written to eigen_points.vtu" << std::endl;
// }
} // namespace io
} // namespace benchy

