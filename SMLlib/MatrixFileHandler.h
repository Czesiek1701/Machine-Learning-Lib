// =======================
// Author: ChatGPT
// =======================

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Dense>

class MatrixFileHandler {
private:
    std::string filename;  // File to which matrices are written
    int precision;         // Precision for saving matrices
    std::ofstream outFile; // Output file stream for writing to the file
public:
    // Constructor to initialize the filename and clear the file initially
    MatrixFileHandler(const std::string& filename);

    // Method to save and close the file
    void save();

    // Method to start a new matrix set in the file
    void startNewMatrixSetInFile(const std::string& setName, int numMatrices);

    // Method to add a matrix to the current set
    void addMatrixToSet(const Eigen::MatrixXd& matrix);

    // Static method to load a single matrix from a set (file)
    static Eigen::MatrixXd loadMatrixFromSet(const std::string& filename, const std::string& setName, int matrixIndex);

    // Static method to load all matrices from a set (file)
    static std::vector<Eigen::MatrixXd> loadSetMatrices(const std::string& filename, const std::string& setName);

};

