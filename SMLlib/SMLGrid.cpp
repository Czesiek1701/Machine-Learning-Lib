#include "SMLGrid.h"

af::afType af::getFunDer(af::afType fun)
{
    if (fun == &af::linear) return &af::linear_der;
    if (fun == &af::sigmoid) return &af::sigmoid_der;
    if (fun == &af::tanh) return &af::tanh_der;
    if (fun == &af::stepBipolar) return &af::stepBipolar_der;
    if (fun == &af::bilinear) return &af::bilinear_der;
}

float af::linear(const float& input_sum)
{
    return input_sum;
}
float af::linear_der(const float& input_sum)
{
    return 1;
}
float af::sigmoid(const float& input_sum)
{
    return 1.0 / (1 + std::exp(-1 * input_sum));
}
float af::sigmoid_der(const float& input_sum)
{
    float s = sigmoid(input_sum);
    return s = s * (1 - s);
}
float af::tanh(const float& input_sum)
{
    return 2.0 / (1 + std::exp(-2 * input_sum))-1;
}
float af::tanh_der(const float& input_sum)
{
    return 1.0 - std::pow(af::tanh(input_sum),2);
}
float af::stepBipolar(const float& input_sum)
{
    return (input_sum >0)?1:(-1);
}
float af::stepBipolar_der(const float& input_sum)
{
    return 1;
}
float af::bilinear(const float& input_sum)
{
    return (input_sum > 0) ? input_sum : (0);
}
float af::bilinear_der(const float& input_sum)
{
    return (input_sum > 0) ? 1 : (1);;
}

void MLGrid::createGrid(
    int input_size,
    std::vector<int> nodes_nums,
    std::vector<float(*)(const float&)> act_funs
    )
{
    if (nodes_nums.size() > act_funs.size())
    {
        std::cout << "Too less act funs" << nodes_nums.size() << ">" << act_funs.size() << std::endl;
        return;
    }
    
    input.clear();
    for (int i = 0; i < input_size; i++)
    {
        MLNode inode;
        inode.bias = 0;
        //inode.afp = nullptr;
        inode.prevs.clear();
        input.push_back(inode);
        std::cout << "input added" << std::endl;
    }

    for (int ln=0 ; ln<nodes_nums.size() ; ln++)
    {

        MLNode node;
        //node.afp = act_funs[ln]; 
        if (ln == 0)
        {
            node.weights = std::vector<float>(input.size());
            for (int i=0;i< input.size();i++)
            {
                node.prevs.push_back(&(input[i]));
            }
        }
        else {
            node.weights = std::vector<float>(grid[ln - 1].size());
            for (int i = 0; i < grid[ln-1].size(); i++)
            {
                node.prevs.push_back(&(grid[ln - 1][i]));
            }
        }
        grid.push_back(std::vector<MLNode>(nodes_nums[ln],node));


    }
    activate_functions = act_funs;
    der_act_funs.clear();
    for (auto af : activate_functions)
    {
        der_act_funs.push_back(af::getFunDer(af));
    }
    
}

void MLGrid::showGrid()
{
    std::cout << "-- MLGrid --" << std::endl;

    std::cout << "\t\t --IN--";
    std::cout << "\nV:";
    std::cout << std::setprecision(3) << std::showpos;
    for (auto i : input)
    {
        std::cout << "\t" << i.output;
    }
    std::cout << std::endl;

    for (int ln = 0; ln < grid.size(); ln++)
    {
        std::cout << std::setprecision(3) << std::showpos;
        std::cout << "\t\t --LAYER: " << ln << "--";// << "\n";

        for (int i = 0; i < grid[ln].size(); i++)
        {
            std::cout << "\nW: ";
            for (auto w : grid[ln][i].weights)
            {
                std::cout << "\t" << w ;
            }
            std::cout << "\tB:  ";
            std::cout << grid[ln][i].bias << "\t";
            std::cout << "V:  ";
            std::cout << grid[ln][i].output;
        }
        std::cout << std::endl;

    }

}

float MLGrid::getRandFloat()
{
    static auto st_time = std::chrono::system_clock::now().time_since_epoch();
    static unsigned int st_seed = std::chrono::duration_cast
        <std::chrono::milliseconds>(st_time).count();
    st_seed = ((st_seed + 327) * 1567 - 3) % 10000;
    return  (st_seed / 10000.0) * 2 - 1;
}

void MLGrid::initializeGrid()
{

    for (auto& layer : grid)
    {
        for(auto & node: layer)
        {
            node.bias = getRandFloat();
        }
    }
    for (auto& layer : grid)
    {
        for (auto& node : layer)
        {
            for (auto& weight : node.weights)
            {

            weight = getRandFloat();
            }
        }
    }

}

void MLGrid::setInput(const std::vector<float>& invals)
{
    if (invals.size() != input.size())
    {
        std::cout << "WRONG SIZE " << invals.size() << "!=" << input.size() << std::endl;
        return;
    }

    for (int i = 0; i < input.size(); i++)
    {
        input[i].output = invals[i];
    }

}

void MLGrid::calcOutput()
{
    int li = 0;
    for (auto& layer : grid)
    {
        for (auto& node : layer)
        {
            //node.prev_output = node.output;
            node.output = node.bias;
            for (int i = 0; i < node.prevs.size(); i++)
            {
                node.output += node.prevs[i]->output * node.weights[i]; // pobudzenie neuronu "e"
                node.bef_af = node.output;
            }
            //std::cout << "pobudzenie " << node.output << std::endl;
            node.output = activate_functions[li](node.output);
        }
        li++;
    }
    //std::cout << grid[grid.size() - 1][0].output << std::endl;
}

void MLGrid::correctWeights(const std::vector<float>& tar_out)
{
    // OUTPUT LAYER SIGMA
    int li = grid.size() - 1;
    for (int ni = 0; ni < grid[li].size(); ni++)
    {
        grid[li][ni].sigma = (tar_out[ni] - grid[li][ni].output); // d-y
    }

    // HIDDEN LAYERS SIGMA
    for (int li = grid.size() - 2; li > -1; li--) //backward
    {
        for (int ni = 0; ni < grid[li].size(); ni++)
        {
            grid[li][ni].sigma = 0;
            for (int nL = 0; nL < grid[li + 1].size(); nL++)
            {
                grid[li][ni].sigma += grid[li + 1][(nL)].weights[ni] * grid[li + 1][(nL)].sigma;
            }
        }
    }

    // UPDATE ALL WEIGHTS
    for (int li = grid.size() - 1; li > -1; li--) //backward
    {
        for (int ni = 0; ni < grid[li].size(); ni++)
        {
            //float delta = eta * grid[li][ni].sigma * (1); // r   f(prev output)
            //delta *= (1) * grid[li][ni].output;
            //float delta = eta * grid[li][ni].sigma * af::tanh_der(grid[li][ni].output)*grid[li][ni].output;
            for (int i = 0; i < grid[li][ni].prevs.size(); i++)
            {
                float delta = grid[li][ni].sigma * der_act_funs[li]( grid[li][ni].prevs[i]->output * grid[li][ni].weights[i] ); // r   f(prev output)
                grid[li][ni].weights[i] += eta * delta * grid[li][ni].prevs[i]->output;
                grid[li][ni].weights[i] += ((std::rand()%20000) / 20000 * 2 -1)*0.0001;
                //std::cout << li << " " << ni << " " << i << std::endl;
            }
        }
    }

    return;

    /*
    //int li = grid.size()-1;
    for (int ni = 0; ni < grid[li].size(); ni++)
    {
        grid[li][ni].sigma = (tar_out[ni] - grid[li][ni].output) * (1);
        float delta = eta * grid[li][ni].sigma;
        //delta *= (1) * grid[li][ni].output;
        //float delta = eta * grid[li][ni].sigma * af::tanh_der(grid[li][ni].output);// *grid[li][ni].output;
        //float delta = eta * (grid[li][ni].output - grid[li][ni].prev_output);
        for (int i = 0; i < grid[li][ni].prevs.size(); i++)
        {
            grid[li][ni].weights[i] += delta * (1) * grid[li][ni].prevs[i]->output;
            grid[li][ni].weights[i] += getRandFloat()/100;
        }
    }

    // NAPRAWIÆ TUTAJ COŒ
    for (int li = grid.size()-2; li > -1; li--) //backward
    {
        for (int ni = 0; ni < grid[li].size(); ni++)
        {
            grid[li][ni].sigma = 0;
            for (int nL = 0; nL < grid[li + 1].size(); nL++)
            {
                grid[li][ni].sigma += grid[li + 1][(nL)].weights[ni] * grid[li + 1][(nL)].sigma;
            }

            float delta = eta * grid[li][ni].sigma;
            //delta *= (1) * grid[li][ni].output;
            //float delta = eta * grid[li][ni].sigma * af::tanh_der(grid[li][ni].output)*grid[li][ni].output;
            for (int i = 0; i < grid[li][ni].prevs.size(); i++)
            {
                grid[li][ni].weights[i] += delta * (1) * grid[li][ni].prevs[i]->output;
                grid[li][ni].weights[i] += getRandFloat() / 100;
            }
        }
    }*/
}
 

std::vector<float>* MLGrid::getOutput()
{
    out.clear();
    for (auto v : grid[grid.size()-1])
    {
        out.push_back(v.output);
    }
    return &out;
}