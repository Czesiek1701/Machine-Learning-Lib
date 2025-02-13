#ifndef H_SMLNETWORK
#define H_SMLNETWORK

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <chrono>
#include <iomanip>
#include <map> 
#include <algorithm>

#include <Eigen/Dense>

namespace egn = Eigen;

namespace af
{
    typedef double (*afType)(const double&);

    afType getFunDer(afType);

    double linear(const double& input_sum);
    double linear_der(const double& input_sum);

    double sigmoid(const double& input_sum);
    double sigmoid_der(const double& input_sum);

    double sigmoid3(const double& input_sum);
    double sigmoid3_der(const double& input_sum);

    double tanh(const double& input_sum);
    double tanh_der(const double& input_sum);

    double stepBipolar(const double& input_sum);
    double stepBipolar_der(const double& input_sum);

    double bilinear(const double& input_sum);
    double bilinear_der(const double& input_sum);

    double sign(const double& input_sum);
    double sign_der(const double& input_sum);

};

double getRandDouble();



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
    std::vector<Layer*> next_layers;
    af::afType afp = af::linear;
    af::afType afp_der = af::linear_der;
    double eta = 0.1;
public:
    explicit Layer();
    virtual ~Layer();
    explicit Layer(int);
    explicit Layer(int, double);
    explicit Layer(int, af::afType);
    explicit Layer(int, af::afType, double);
    virtual void calcOutput();
    void connectBack(Layer*);
    int getN() const;
    void showOutput() const;
    void showLayer() const;
    virtual void calcSigma();
    virtual void calcDelta();
    virtual void correctWeight();
};

class ConstLayer : public Layer
{
public:
    explicit ConstLayer() : Layer() {}
    explicit ConstLayer(int nn);// : Layer(nn) {}
    virtual void calcOutput() override;
};

class OutputLayer : public Layer
{
private:
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
public:
    explicit InputLayer() : Layer() {}
    explicit InputLayer(int nn);// : Layer(nn) {}
    virtual void calcOutput() override;
    void setInput(egn::Matrix<double, egn::Dynamic, 1>);
};

#endif