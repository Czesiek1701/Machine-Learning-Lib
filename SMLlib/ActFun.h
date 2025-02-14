#pragma once

#include <cmath>

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