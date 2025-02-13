#include "SMLGridP.h"

afP::afType afP::getFunDer(afP::afType fun)
{
    if (fun == &afP::linear) return &afP::linear_der;
    if (fun == &afP::sigmoid) return &afP::sigmoid_der;
    if (fun == &afP::sigmoid3) return &afP::sigmoid3_der;
    if (fun == &afP::tanh) return &afP::tanh_der;
    if (fun == &afP::stepBipolar) return &afP::stepBipolar_der;
    if (fun == &afP::bilinear) return &afP::bilinear_der;
    if (fun == &afP::sign) return &afP::sign_der;
}

double afP::linear(const double& input_sum)
{
    return input_sum;
}
double afP::linear_der(const double& input_sum)
{
    return 1;
}
double afP::sigmoid(const double& input_sum)
{
    return 1.0 / (1 + std::exp(-1 * input_sum));
}
double afP::sigmoid_der(const double& input_sum)
{
    double s = sigmoid(input_sum);
    return s = s * (1 - s);
}
double afP::sigmoid3(const double& input_sum)
{
    return 1.0 / (1 + std::exp(-3 * input_sum));
}
double afP::sigmoid3_der(const double& input_sum)
{
    double s = sigmoid(input_sum);
    return s = 3 * s * (1 - s);
}
double afP::tanh(const double& input_sum)
{
    return 2.0 / (1 + std::exp(-2 * input_sum))-1;
}
double afP::tanh_der(const double& input_sum)
{
    return 1.0 - std::pow(afP::tanh(input_sum),2);
}
double afP::stepBipolar(const double& input_sum)
{
    return (input_sum >0)?1:(-1);
}
double afP::stepBipolar_der(const double& input_sum)
{
    return 1;
}
double afP::bilinear(const double& input_sum)
{
    return (input_sum > 0) ? input_sum : (0.1*input_sum);
}
double afP::bilinear_der(const double& input_sum)
{
    return (input_sum > 0) ? 1 : (0.1*1);
}
double afP::sign(const double& input_sum)
{
    return (input_sum >= 0)?1:(-1);
}
double afP::sign_der(const double& input_sum)
{
    return 0.1;
}

void MLGrid::createGrid(
    int input_size,
    std::vector<int> nodes_nums,
    std::vector<double(*)(const double&)> act_funs
    )
{
    if (nodes_nums.size() > act_funs.size())
    {
        std::cout << "Too less act funs" << nodes_nums.size() << ">" << act_funs.size() << std::endl;
        return;
    }

    constNode.output = -1;
    
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
        node.prevs.push_back(&constNode);
        if (ln == 0)
        {
            node.weights = std::vector<double>(input.size()+1);
            for (int i=0;i< input.size();i++)
            {
                node.prevs.push_back(&(input[i]));
            }
        }
        else {
            node.weights = std::vector<double>(grid[ln - 1].size()+1);
            for (int i = 0; i < grid[ln-1].size(); i++)
            {
                node.prevs.push_back(&(grid[ln - 1][i]));
            }
        }
        node.afp = act_funs[ln];
        grid.push_back(std::vector<MLNode>(nodes_nums[ln],node));


    }
    activate_functions = act_funs;
    der_act_funs.clear();
    for (auto af : activate_functions)
    {
        der_act_funs.push_back(afP::getFunDer(af));
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

double getRanddoubleRRR()
{
    static auto st_time = std::chrono::system_clock::now().time_since_epoch();
    static unsigned int st_seed = std::chrono::duration_cast
        <std::chrono::milliseconds>(st_time).count();
    st_seed = ((st_seed + 327) * 1567 - 3) % 10000;
    //std::cout << "rand: " << (st_seed / 10000.0) * 2 - 1 << std::endl;
    return  (st_seed / 10000.0) * 2 - 1;
}

void MLGrid::initializeGrid()
{

    //std::cout << "initializing" << std::endl;

    for (auto& layer : grid)
    {
        for (auto& node : layer)
        {
            node.bias = getRanddoubleRRR();
            for (auto& weight : node.weights)
            {

            weight = getRanddoubleRRR();
            }
        }
    }

}

void MLGrid::setInput(const std::vector<double>& invals)
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
                //node.output += node.const_weight * (-1); // pobudzenie neuronu "e"
                //std::cout << node.prevs[i]->output;
                node.output += node.prevs[i]->output * node.weights[i]; // pobudzenie neuronu "e"
            }
            //std::cout << std::endl;
            node.e_input = node.output;
            //std::cout << "pobudzenie " << node.output << std::endl;
            node.output = activate_functions[li](node.output);
        }
        li++;
    }
    //std::cout << grid[grid.size() - 1][0].output << std::endl;
}

void MLGrid::calcErrors(const std::vector<double>& tar_out)
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
}

void MLGrid::correctWeights(const std::vector<double>& tar_out)
{
    calcErrors(tar_out);
    // UPDATE ALL WEIGHTS
    //double delta;
    for (int li = grid.size() - 1; li > -1; li--) //backward
    {
        for (int ni = 0; ni < grid[li].size(); ni++)
        {
            //double delta = eta * grid[li][ni].sigma * (1); // r   f(prev output)
            //delta *= (1) * grid[li][ni].output;
            //double delta = eta * grid[li][ni].sigma * af::tanh_der(grid[li][ni].output)*grid[li][ni].output;
            /*
            for (int i = 0; i < grid[li][ni].prevs.size(); i++)
            {
                delta = grid[li][ni].sigma * der_act_funs[li]( grid[li][ni].prevs[i]->output * grid[li][ni].weights[i] ); // r   f(prev output)
                grid[li][ni].weights[i] += eta * delta * grid[li][ni].prevs[i]->output;// -0.00001 * eta * grid[li][ni].weights[i];
                grid[li][ni].weights[i] += getRanddouble()*0.001;
            }
            */
            //grid[li][ni].const_weight += grid[li][ni].sigma * der_act_funs[li]((-1)* grid[li][ni].const_weight)* eta*(-1);
            grid[li][ni].teach(this);
        }
    }

    grid[0][1].teach(this);

    return;
}

void MLNode::teach(const MLGrid* g)
{
    for (int i = 0; i < this->prevs.size(); i++)
    {
        double delta = this->sigma * afP::getFunDer(afp)(this->prevs[i]->output * this->weights[i]); // r   f(prev output)
        this->weights[i] += g->eta * delta * this->prevs[i]->output;// -0.00001 * eta * grid[li][ni].weights[i];
        this->weights[i] += getRanddoubleRRR() * 0.0001;
    }
}

void MLGrid::correctWeightsOneByOne(const std::vector<double>& tar_out)
{

    for (int li = grid.size() - 1; li > -1; li--) //backward
    {
        for (int ni = 0; ni < grid[li].size(); ni++)
        {
            calcOutput();
            calcErrors(tar_out);
            grid[li][ni].teach(this);
        }
    }


}

void MLGrid::correctWeightsCoin(const std::vector<double>& tar_out)
{
    //eta *= 0.9999;
    //std::cout << "eta: " << eta << std::endl;
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
        //for (int ni = 0; ni < grid[li].size(); ni++)
        //{
            //double delta = eta * grid[li][ni].sigma * (1); // r   f(prev output)
            //delta *= (1) * grid[li][ni].output;
            //double delta = eta * grid[li][ni].sigma * af::tanh_der(grid[li][ni].output)*grid[li][ni].output;
        auto win_neur = std::max_element((grid[li]).begin(), (grid[li]).end(), [](const MLNode& a, const MLNode& b) {return (a.e_input)< (b.e_input); });
        for (int i = 0; i < win_neur->prevs.size(); i++)
        {
            double delta = win_neur->sigma * der_act_funs[li](win_neur->prevs[i]->output* win_neur->weights[i]); // r   f(prev output)
            win_neur->weights[i] += eta * delta * win_neur->prevs[i]->output;
            //win_neur->weights[i] += ((std::rand() % 20000) / 20000 * 2 - 1) * 0.0001;
            //std::cout << li << " " << ni << " " << i << std::endl;
        }
        //}
    }

    return;
}
 

std::vector<double>* MLGrid::getOutput()
{
    out.clear();
    for (auto v : grid[grid.size()-1])
    {
        out.push_back(v.output);
    }
    return &out;
}

void MLGrid::setEta(double e)
{
    eta = e;
}
