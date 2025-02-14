#include "Layer.h"

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


void Layer::calcDelta()
{
    //net = 
    delta = net;
    for (auto& d : delta) { d = afp_der(d); }
    delta = delta.cwiseProduct(sigma);
}

void Layer::correctAllWeights()
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
