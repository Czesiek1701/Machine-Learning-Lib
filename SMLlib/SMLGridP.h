#ifndef H_SMLGRID
#define H_SMLGRID

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <map> 
#include <algorithm>

namespace afP
{
    typedef double (*afType)(const double&);

    afType getFunDer(afType);

    double linear(const double& input_sum);
    double linear_der(const double& input_sum);

    double sigmoid(const double& input_sum);
    double sigmoid_der(const double& input_sum);

    double sigmoid3(const double& input_sum);
    double sigmoid3_der(const double& input_sum);

    double tanh(const double& input_sum);
    double tanh_der(const double& input_sum);

    double stepBipolar(const double& input_sum);
    double stepBipolar_der(const double& input_sum);

    double bilinear(const double& input_sum);
    double bilinear_der(const double& input_sum);

    double sign(const double& input_sum);
    double sign_der(const double& input_sum);

};


double getRanddoubleRRR();

class MLGrid;

class MLNode
{
private:
    //double const_weight = -1;
    double output=0;
    double e_input=0;
    double bias = 0;
    //double (*afp)(const double&)= af::linear;
    std::vector< MLNode* > prevs;
    std::vector< MLNode* > nexts;
    std::vector< double > weights;
    double sigma = 0;
    friend MLGrid;

    afP::afType afp = afP::linear;

    void teach(const MLGrid* g);
};

class MLGrid
{
private:
    std::vector< MLNode > input;
    std::vector< std::vector<MLNode> > grid;
    std::vector<afP::afType> activate_functions;
    std::vector<afP::afType> der_act_funs;
    std::vector<double> out;
    double eta=10000;
    MLNode constNode;
public:
    void createGrid(
        int input_size, 
        std::vector<int> nodes_nums,
        std::vector<double(*)(const double&)> act_funs
    );
    void showGrid(); 
    void initializeGrid();
    void setInput(const std::vector<double>&);
    void calcOutput();
    void calcErrors(const std::vector<double>&);
    void correctWeights(const std::vector<double>&);
    void correctWeightsCoin(const std::vector<double>&);

    void correctWeightsOneByOne(const std::vector<double>&);

    std::vector<double>* getOutput();

    void setEta(double);

    friend MLNode;
};




#endif 