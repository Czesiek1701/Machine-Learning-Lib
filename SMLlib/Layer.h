// =======================
// Author: Grzegorz Czaja
// =======================
#pragma once

#include <vector>
#include <chrono>
#include <iostream>
#include <Eigen/Dense>

#include "ActFun.h"

namespace egn = Eigen;

double getRandDouble();

class NNetwork;



class Layer
{
protected:
    int n = 0;
    std::vector<Layer*> prev_layers;
    std::vector<egn::MatrixXd> weight;
    egn::Matrix<double, egn::Dynamic, 1> bias;
    egn::Matrix<double, egn::Dynamic, 1> net;
    egn::Matrix<double, egn::Dynamic, 1> output;
    egn::Matrix<double, egn::Dynamic, 1> sigma;
    egn::Matrix<double, egn::Dynamic, 1> delta;
    egn::Matrix<double, egn::Dynamic, 1> difOutput;
    egn::Matrix<double, egn::Dynamic, 1> difSigma;
    std::vector<Layer*> next_layers;
    af::afType afp = af::linear;
    af::afType afp_der = af::linear_der;
    double eta = 0.1;
public:
    explicit Layer();
    virtual ~Layer();
    explicit Layer(int);
    explicit Layer(int, af::afType, double);
    virtual void calcOutput();
    void connectBack(Layer*);
    int getN() const;
    void showOutput() const;
    void showLayer() const;
    virtual void calcSigma();
    virtual void calcNeuronSigma(int);
    virtual void calcDelta();
    virtual void calcNeuronDelta(int);
    virtual void correctAllWeights();
    virtual void correctNeuronWeight(int);
    virtual std::string getName() const;
    const egn::Matrix<double, egn::Dynamic, 1>& getOutput();
    void disconnect(Layer*);
    void presentAsNode();
    //virtual egn::Matrix<double, egn::Dynamic, 1> getOutput();

    friend NNetwork;
};