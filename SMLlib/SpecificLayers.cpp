#include "SpecificLayers.h"

void ConstLayer::calcOutput()
{
    return;
}

ConstLayer::ConstLayer()
    : ConstLayer(1)
{
}

ConstLayer::ConstLayer(int nn)
    : Layer(nn, nullptr, 0)
{
    output.setConstant(1);
}

std::string ConstLayer::getName() const
{
    return "Layer Constant";
}

//---------------------------------------------------------

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

const egn::Matrix<double, egn::Dynamic, 1>& OutputLayer::getOutput()
{
    return output;
}

const egn::Matrix<double, egn::Dynamic, 1>& OutputLayer::getTarget()
{
    return target;
}

void OutputLayer::calcSigma()
{
    sigma = target - output;
}

void OutputLayer::setTargetOutput(egn::Matrix<double, egn::Dynamic, 1> m)
{
    target = m;
}

std::string OutputLayer::getName() const
{
    return "Layer Output";
}

//---------------------------------------------------------

void InputLayer::calcOutput()
{
    return;
}

InputLayer::InputLayer()
    :InputLayer(1)
{
}

InputLayer::InputLayer(int nn)
    : Layer(nn, nullptr, 0)
{
    output.setConstant(1);
}

void InputLayer::setInput(egn::Matrix<double, egn::Dynamic, 1> in)
{
    output = in;
}

std::string InputLayer::getName() const
{
    return "Layer Input";
}

//---------------------------------------------------------

HiddenLayer::HiddenLayer(int n, af::afType f, double e)
    : Layer(n, f, e)
{
}

HiddenLayer::HiddenLayer()
    : HiddenLayer(1, af::linear, 0.1)
{
}

HiddenLayer::HiddenLayer(int n)
    : HiddenLayer(n, af::linear, 0.1)
{
}

std::string HiddenLayer::getName() const
{
    return "Layer Hidden";
}