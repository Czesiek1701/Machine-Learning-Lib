
#include "SpecificLayers.h"

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


void InputLayer::setInput(egn::Matrix<double, egn::Dynamic, 1> in)
{
    output = in;
}