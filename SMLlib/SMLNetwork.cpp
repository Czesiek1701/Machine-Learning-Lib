# include "SMLNetwork.h"

af::afType af::getFunDer(af::afType fun)
{
    if (fun == &af::linear) return &af::linear_der;
    if (fun == &af::sigmoid) return &af::sigmoid_der;
    if (fun == &af::sigmoid3) return &af::sigmoid3_der;
    if (fun == &af::tanh) return &af::tanh_der;
    if (fun == &af::stepBipolar) return &af::stepBipolar_der;
    if (fun == &af::bilinear) return &af::bilinear_der;
    if (fun == &af::sign) return &af::sign_der;
}

double af::linear(const double& input_sum)
{
    return input_sum;
}
double af::linear_der(const double& input_sum)
{
    return 1;
}
double af::sigmoid(const double& input_sum)
{
    return 1.0 / (1 + std::exp(-1 * input_sum));
}
double af::sigmoid_der(const double& input_sum)
{
    double s = sigmoid(input_sum);
    return s = s * (1 - s);
}
double af::sigmoid3(const double& input_sum)
{
    return 1.0 / (1 + std::exp(-3 * input_sum));
}
double af::sigmoid3_der(const double& input_sum)
{
    double s = sigmoid(input_sum);
    return s = 3 * s * (1 - s);
}
double af::tanh(const double& input_sum)
{
    return 2.0 / (1 + std::exp(-2 * input_sum)) - 1;
}
double af::tanh_der(const double& input_sum)
{
    return 1.0 - std::pow(af::tanh(input_sum), 2);
}
double af::stepBipolar(const double& input_sum)
{
    return (input_sum > 0) ? 1 : (-1);
}
double af::stepBipolar_der(const double& input_sum)
{
    return 1;
}
double af::bilinear(const double& input_sum)
{
    return (input_sum > 0) ? input_sum : (0.1 * input_sum);
}
double af::bilinear_der(const double& input_sum)
{
    return (input_sum > 0) ? 1 : (0.1 * 1);
}
double af::sign(const double& input_sum)
{
    return (input_sum >= 0) ? 1 : (-1);
}
double af::sign_der(const double& input_sum)
{
    return 0.1;
}

double getRandDouble()
{
    static auto st_time = std::chrono::system_clock::now().time_since_epoch();
    static unsigned int st_seed = std::chrono::duration_cast
        <std::chrono::milliseconds>(st_time).count();
    st_seed = ((st_seed + 327) * 1567 - 3) % 10000;
    //std::cout << "rand: " << (st_seed / 10000.0) * 2 - 1 << std::endl;
    return  (st_seed / 10000.0) * 2 - 1;
}

Layer::~Layer()
{

}

Layer::Layer(int nn = 1, af::afType lfun = af::linear, double e = 0.1)
    : n(nn)
{
    bias.resize(n, 1);
    bias.setConstant(0);
    net.resize(n, 1);
    net.setConstant(0);
    output.resize(n, 1);
    output.setConstant(1);
    sigma.resize(n, 1);
    sigma.setConstant(1);
    delta.resize(n, 1);
    delta.setConstant(0);
    afp = lfun;
    afp_der = af::getFunDer(afp);
    eta = e;
    prev_layers.clear();
    next_layers.clear();
}

Layer::Layer()
    : Layer(1, af::linear, 0.1)
{
}
Layer::Layer(int n)
    : Layer(n, af::linear, 0.1)
{
}
Layer::Layer(int n, double e)
    : Layer(n, af::linear, e)
{
}
Layer::Layer(int n, af::afType f)
    : Layer(n, f, 0.1)
{
}

void Layer::calcOutput()
{
    auto wp = weight.begin();
    net = bias;
    for (const Layer* const lp : prev_layers)
    {
        //std::cout << "calc" << std::endl;
        net += wp->transpose() * lp->output; 
        ++wp;
    }
    output = net;
    for (auto& o : output) { o = afp(o); }
}

int Layer::getN() const
{
    return this->n;
}

void Layer::connectBack(Layer* pl)
{
    this->prev_layers.push_back(pl);
    pl->next_layers.push_back(this);
    this->weight.push_back(egn::MatrixXd(pl->getN(), this->getN()));
    //for (auto& m : (*(this->weight.rend())).reshaped()) m = getRandDouble();
    for (auto& m : weight.back().reshaped()) { m = getRandDouble(); }
}

void Layer::showOutput() const
{
    std::cout << this->output << std::endl;
}

void Layer::showLayer() const
{
    std::cout << "---\nLAYER:\t" << this << std::endl;
    auto w = weight.begin();
    for (auto l : prev_layers)
    {
        std::cout << "prev:\t" << l << std::endl;
        std::cout << "weight:\t" << std::endl;
        std::cout << *w << std::endl;
        ++w;
    }
    std::cout << "bias:\t";
    for (auto b : bias)
    {
        std::cout << b << " ";
    }
    std::cout << std::endl;
    std::cout << "net:\t";
    for (auto m : net)
    {
        std::cout << m << " ";
    }
    std::cout << std::endl;
    std::cout << "output:\t";
    for (auto o : output)
    {
        std::cout << o << " ";
    }
    std::cout << std::endl;
    std::cout << "nexts:\t";
    for (auto l : next_layers)
    {
        std::cout << l << " ";
    }
    std::cout << std::endl;
    std::cout << "sigma:\t";
    for (auto l : sigma)
    {
        std::cout << l << " ";
    }
    std::cout << std::endl;
    std::cout << "delta:\t";
    for (auto l : delta)
    {
        std::cout << l << " ";
    }
    std::cout << std::endl;

}

void Layer::calcSigma()
{
    //std::cout << "sigma " << this << std::endl;
    this->sigma.setConstant(0);
    for (auto nl : next_layers)
    {
        int wn =
            std::find(
                nl->prev_layers.begin(),
                nl->prev_layers.end(),
                this
            ) - nl->prev_layers.begin();
        //std::cout << "from: " << nl << " : " << wn << std::endl;
        //std::cout << sigma << std::endl;
        //std::cout << nl->sigma << std::endl;
        //std::cout << nl->weight[wn] << std::endl;
        this->sigma += nl->weight[wn] * nl->sigma;
    }
}

void ConstLayer::calcOutput()
{
    return;
}

ConstLayer::ConstLayer(int nn)
    : Layer(nn, nullptr, 0)
{
    output.setConstant(1);
}

OutputLayer::OutputLayer(int n, af::afType f, double e)
    : Layer(n, f, e)
{
    target.resize(n, 1);
    target.setConstant(0);
}
OutputLayer::OutputLayer()
    : OutputLayer(1, af::linear, 0.1)
{
}
OutputLayer::OutputLayer(int n)
    : OutputLayer(n, af::linear, 0.1)
{
}
OutputLayer::OutputLayer(int n, double e)
    : OutputLayer(n, af::linear, e)
{
}
OutputLayer::OutputLayer(int n, af::afType f)
    : OutputLayer(n, f, 0.1)
{
}

const egn::Matrix<double, egn::Dynamic, 1>& OutputLayer::getOutput()
{
    return output;
}
const egn::Matrix<double, egn::Dynamic, 1>& OutputLayer::getTarget()
{
    return target;
}

void InputLayer::calcOutput()
{
    return;
}

InputLayer::InputLayer(int nn)
    : Layer(nn, nullptr, 0)
{
    output.setConstant(1);
}



void OutputLayer::calcSigma()
{
    sigma = target - output;
}

void OutputLayer::setTargetOutput(egn::Matrix<double, egn::Dynamic, 1> m)
{
    target = m;
}

//ConstLayer::ConstLayer()
//    : Layer::Layer()
//{
//}
//ConstLayer::ConstLayer(int m)
//    : Layer::Layer(m)
//{
// 
// voi
//}

void Layer::calcDelta()
{
    //net = 
    delta = net;
    for (auto& d : delta) { d = afp_der(d); }
    delta = delta.cwiseProduct(sigma);
}

void Layer::correctWeight()
{
    //std::cout << "weight correction: " << this << std::endl;
    auto w = weight.begin();
    for (auto l : prev_layers)
    {
        //std::cout << "w:\n" << * w << std::endl;
        //std::cout << "d:\n" << delta << std::endl;
        //std::cout << "o:\n" << l->output << std::endl;
        //std::cout << "sigma\n" << sigma << std::endl;
        //std::cout << "delta\n" << delta << std::endl;
        //std::cout << "deltaW\n";
        //std::cout << eta * l->output * delta.transpose() << std::endl;
        *w += eta * (l->output) * delta.transpose();
        ++w;
    }
}

void InputLayer::setInput(egn::Matrix<double, egn::Dynamic, 1> in)
{
    output = in;
}