#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <map>  
#include <algorithm>

#include "ActFun.h"
#include "Layer.h"
#include "SpecificLayers.h"

#include <Eigen/Dense>

//----------------------------------------------------------------------------------------------------------------

class NNetwork
{
private:
	std::vector<Layer*> layers_all; // 1-input , 0-const, last- output
	std::unique_ptr<ConstLayer> const_layer;
	std::unique_ptr<InputLayer> input_layer;
	std::unique_ptr<OutputLayer> output_layer;
	std::vector<std::unique_ptr<HiddenLayer>> hidden_layers;
	std::vector<int> learning_order;
	std::vector<int> calc_order;
public:
	NNetwork(int,std::vector<int>, bool, std::vector<af::afType>, std::vector<double>); // create and add to layers
	void showLayers() const;
	void setCalcOrder();
	void setLearningOrder();
	//void setInput(const egn::Matrix<double, egn::Dynamic, 1>&);
	//void setTargetOutput(const egn::Matrix<double, egn::Dynamic, 1>&);

};

