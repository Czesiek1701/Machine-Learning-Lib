#include "ActFun.h"

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
