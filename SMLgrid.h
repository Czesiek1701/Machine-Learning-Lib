#include <iostream>
#include <vector>
#include <cmath>

class MLConn;
class MLNode;

namespace af
{
    float linear(const float& input_sum)
    {
        return input_sum; 
    }

    float sigmoid(const float& input_sum)
    {
        return 1.0/(1+std::exp(-1*input_sum));
    }

    float tanh(const float& input_sum)
    {
        return af::sigmoid(input_sum)*2-1;
    }
}

class MLGrid
{
private:
    std::vector< float >* input;
    std::vector< std::vector<MLNode> > grid;
public:
    void createGrid(
        /*std::vector< float >* input_ref,
        std::vector<int> nodes_nums,
        std::vector<float(*)(const float&)> activation_functions*/
        );

};

class MLNode
{
public:
    float bias=0;
    float (*afp)(const float&); 
    float output; 

    std::vector< MLConn > prevs;
};

class MLConn
{
public:
    MLNode* node;
    float weight;
};
