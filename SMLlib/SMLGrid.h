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

namespace af
{
    typedef float (*afType)(const float&);

    afType getFunDer(afType);

    float linear(const float& input_sum);
    float linear_der(const float& input_sum);

    float sigmoid(const float& input_sum);
    float sigmoid_der(const float& input_sum);

    float tanh(const float& input_sum);
    float tanh_der(const float& input_sum);

    float stepBipolar(const float& input_sum);
    float stepBipolar_der(const float& input_sum);

    float bilinear(const float& input_sum);
    float bilinear_der(const float& input_sum);

};

class MLNode;

class MLGrid
{
private:
    std::vector< MLNode > input;
    std::vector< std::vector<MLNode> > grid;
    std::vector<af::afType> activate_functions;
    std::vector<af::afType> der_act_funs;
    std::vector<float> out;
    float eta = 0.05;
public:
    void createGrid(
        int input_size, 
        std::vector<int> nodes_nums,
        std::vector<float(*)(const float&)> act_funs
    );
    void showGrid(); 
    void initializeGrid();
    void setInput(const std::vector<float>&);
    void calcOutput();
    void correctWeights(const std::vector<float>&);
    float getRandFloat();
    std::vector<float>* getOutput();
};

class MLNode
{
private:
    float output;
    float bef_af;
    float bias = 0;
    float (*afp)(const float&);
    std::vector< MLNode* > prevs;
    std::vector< MLNode* > nexts;
    std::vector< float > weights;
    float sigma;
    friend MLGrid;
};


#endif 