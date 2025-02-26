// =======================
// Author: ChatGPT
// =======================

#include "MatrixFileHandler.h"
#include <sstream>
#include <iomanip>

MatrixFileHandler::MatrixFileHandler(const std::string& filename)
    : filename(filename), precision(6) {
    // Clear the file content initially
    std::ofstream outFile(filename, std::ios::trunc);
    outFile.close();
}

void MatrixFileHandler::save() {
    if (outFile.is_open()) {
        outFile.close();  // Close the file if it's open
    }
}

void MatrixFileHandler::startNewMatrixSetInFile(const std::string& setName, int numMatrices) {
    std::ofstream outFile(filename, std::ios::app);
    outFile << setName << " " << numMatrices << std::endl;
    this->outFile = std::move(outFile);
}

void MatrixFileHandler::addMatrixToSet(const Eigen::MatrixXd& matrix) {
    if (outFile.is_open()) {
        int rows = matrix.rows();
        int cols = matrix.cols();
        outFile << rows << " " << cols << std::endl;
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                outFile << std::fixed << std::setprecision(precision) << matrix(r, c);
                if (c < cols - 1) outFile << ",";
            }
            outFile << std::endl;
        }
    }
}

Eigen::MatrixXd MatrixFileHandler::loadMatrixFromSet(const std::string& filename, const std::string& setName, int matrixIndex) {
    std::ifstream inFile(filename);
    Eigen::MatrixXd matrix;

    if (!inFile.is_open()) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return matrix;
    }

    std::string currentSetName;
    int numMatrices;
    while (inFile >> currentSetName >> numMatrices) {
        if (currentSetName == setName) {
            // Read matrices for the current set
            for (int i = 0; i < numMatrices; ++i) {
                int rows, cols;
                inFile >> rows >> cols;  // Read matrix dimensions
                if (i == matrixIndex) {
                    matrix.resize(rows, cols);
                    std::string line;
                    std::getline(inFile, line);  // Skip the newline after dimensions
                    for (int r = 0; r < rows; ++r) {
                        std::getline(inFile, line);  // Read the matrix row
                        std::stringstream ss(line);
                        std::string value;
                        for (int c = 0; c < cols; ++c) {
                            std::getline(ss, value, ',');  // Read values separated by commas
                            matrix(r, c) = std::stod(value);  // Convert to double
                        }
                    }
                    inFile.close();
                    return matrix;
                }
                else {
                    // Skip the current matrix if it's not the one we want
                    std::string line;
                    std::getline(inFile, line);  // Skip the newline after dimensions
                    for (int r = 0; r < rows; ++r) {
                        std::getline(inFile, line);  // Skip matrix data
                    }
                }
            }
        }
        else {
            // Skip over matrices in a different set
            for (int i = 0; i < numMatrices; ++i) {
                int rows, cols;
                inFile >> rows >> cols;  // Read matrix dimensions to skip it
                std::string line;
                std::getline(inFile, line);  // Skip the newline after dimensions
                for (int r = 0; r < rows; ++r) {
                    std::getline(inFile, line);  // Skip matrix data
                }
            }
        }
    }
    inFile.close();
    return matrix;
}

std::vector<Eigen::MatrixXd> MatrixFileHandler::loadSetMatrices(const std::string& filename, const std::string& setName) {
    std::ifstream inFile(filename);
    std::vector<Eigen::MatrixXd> matrices;

    if (!inFile.is_open()) {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        return matrices;
    }

    std::string currentSetName;
    int numMatrices;
    while (inFile >> currentSetName >> numMatrices) {
        if (currentSetName == setName) {
            // Read matrices for the current set
            for (int i = 0; i < numMatrices; ++i) {
                int rows, cols;
                inFile >> rows >> cols;  // Read matrix dimensions
                Eigen::MatrixXd matrix(rows, cols);
                std::string line;
                std::getline(inFile, line);  // Skip the newline after dimensions
                for (int r = 0; r < rows; ++r) {
                    std::getline(inFile, line);  // Read the matrix row
                    std::stringstream ss(line);
                    std::string value;
                    for (int c = 0; c < cols; ++c) {
                        std::getline(ss, value, ',');  // Read values separated by commas
                        matrix(r, c) = std::stod(value);  // Convert to double
                    }
                }
                matrices.push_back(matrix);
            }
            inFile.close();
            return matrices;
        }
        else {
            // Skip over matrices in a different set
            for (int i = 0; i < numMatrices; ++i) {
                int rows, cols;
                inFile >> rows >> cols;  // Read matrix dimensions to skip it
                std::string line;
                std::getline(inFile, line);  // Skip the newline after dimensions
                for (int r = 0; r < rows; ++r) {
                    std::getline(inFile, line);  // Skip matrix data
                }
            }
        }
    }
    inFile.close();
    return matrices;
}
