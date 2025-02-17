// =======================
// Author: Grzegorz Czaja
// =======================
#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>
#include <map>  
#include <algorithm>
#include <numeric>

#include "ActFun.h"
#include "Layer.h"
#include "SpecificLayers.h" 

#include <Eigen/Dense>

//----------------------------------------------------------------------------------------------------------------

class NNetwork
{
private:
	std::vector<Layer*> layers_all; // 1-input , 0-const
	std::unique_ptr<ConstLayer> const_layer;
	std::unique_ptr<InputLayer> input_layer;
	std::unique_ptr<OutputLayer> output_layer;
	std::vector<std::unique_ptr<HiddenLayer>> hidden_layers;
	std::vector<Layer*> learning_order;
	std::vector<Layer*> calc_order;
	std::vector<Layer*> unvisited;
	double diff_error_output = 0;
	double diff_error_sigma = 0;
public:
	NNetwork(int,std::vector<int>, bool, std::vector<af::afType>, std::vector<double>); // create and add to layers
	void showLayers() const;
	void setCalcOrder();
	void setCalcOrder(Layer*);
	void setLearningOrder();
	void setLearningOrder(Layer*);
	void setInput(const egn::Matrix<double, egn::Dynamic, 1>&);
	void setTargetOutput(const egn::Matrix<double, egn::Dynamic, 1>&);
	void calcOutput();
	void calcOutput(int);
	void calcSigma();
	void calcDelta();
	void correctWeightsAll();
	void correctWeightsOneByOne(int);
	void correctWeightsOneByOne();
	void correctWeightsWinnigOne();
	void showOutput() const;
	void showOutputs();
	void showSigmas();
	void deleteLayer(int);
	int getLayerIndex(Layer*);
	OutputLayer* getOutputLayer();
	Layer* getLayer(int);

	Layer* addLayer(HiddenLayer);
	void connectLayers(Layer*, Layer*);
	void connectLayers(int, int);
	Layer* insertLayerBetween(HiddenLayer, int, int);

	void showConnections();

	void showCalcOrder();
	void showLearningOrder();

};

