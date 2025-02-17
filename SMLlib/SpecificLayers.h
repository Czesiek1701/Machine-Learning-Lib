// =======================
// Author: Grzegorz Czaja
// =======================
#pragma once

#include "Layer.h"

class ConstLayer : public Layer
{
protected:
    std::string name = "Layer Constant";
public:
    explicit ConstLayer();
    explicit ConstLayer(int nn);// : Layer(nn) {}
    virtual void calcOutput() override;
    virtual std::string getName() const override;
    virtual void calcSigma() { return; };
};

class HiddenLayer : public Layer 
{
protected:
    std::string name = "Layer Hidden";
public:
    explicit HiddenLayer();
    explicit HiddenLayer(int);
    explicit HiddenLayer(int, af::afType, double);
    virtual std::string getName() const override;
};

class OutputLayer : public Layer
{
protected:
    std::string name = "Layer Output";
    //const std::string name = "Output Layer";
    egn::Matrix<double, egn::Dynamic, 1> target;
public:
    explicit OutputLayer();
    explicit OutputLayer(int);
    explicit OutputLayer(int, af::afType, double);
    virtual void calcSigma() override;
    void setTargetOutput(egn::Matrix<double, egn::Dynamic, 1>);
    const egn::Matrix<double, egn::Dynamic, 1>& getOutput();
    const egn::Matrix<double, egn::Dynamic, 1>& getTarget();
    virtual std::string getName() const override;
};

class InputLayer : public Layer
{
protected:
    std::string name = "Layer Input";
public:
    explicit InputLayer();
    explicit InputLayer(int nn);// : Layer(nn) {}
    virtual void calcOutput() override;
    void setInput(egn::Matrix<double, egn::Dynamic, 1>);
    virtual std::string getName() const override;
    virtual void calcSigma() { return; };
};