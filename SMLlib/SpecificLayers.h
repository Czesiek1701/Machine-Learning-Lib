#pragma once

#include "Layer.h"

class ConstLayer : public Layer
{
//protected:
    //const std::string name = "Const Layer";
public:
    explicit ConstLayer() : Layer() {}
    explicit ConstLayer(int nn);// : Layer(nn) {}
    virtual void calcOutput() override;
};

class HiddenLayer : public Layer
{
public:
    explicit HiddenLayer();
    explicit HiddenLayer(int);
    explicit HiddenLayer(int, double);
    explicit HiddenLayer(int, af::afType);
    explicit HiddenLayer(int, af::afType, double);
};

class OutputLayer : public Layer
{
protected:
    //const std::string name = "Output Layer";
    egn::Matrix<double, egn::Dynamic, 1> target;
public:
    explicit OutputLayer();
    explicit OutputLayer(int);
    explicit OutputLayer(int, double);
    explicit OutputLayer(int, af::afType);
    explicit OutputLayer(int, af::afType, double);
    virtual void calcSigma() override;
    void setTargetOutput(egn::Matrix<double, egn::Dynamic, 1>);
    const egn::Matrix<double, egn::Dynamic, 1>& getOutput();
    const egn::Matrix<double, egn::Dynamic, 1>& getTarget();
};

class InputLayer : public Layer
{
//protected:
    //const std::string name = "Input Layer";
public:
    explicit InputLayer() : Layer() {}
    explicit InputLayer(int nn);// : Layer(nn) {}
    virtual void calcOutput() override;
    void setInput(egn::Matrix<double, egn::Dynamic, 1>);
};