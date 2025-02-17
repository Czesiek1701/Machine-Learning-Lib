// =======================
// Author: Grzegorz Czaja
// =======================
#include <iostream>
#include "SMLNetwork.h"
#include <numeric>

double refFun(float x)
{
	//x = (-x);
	//return (1-(x-0.2) * (x-0.5) - 0.5*(x-0.5)*(x - 0.5))*0.5+0.2;
	//return 0.3 - 0.5 * x;
	//return (7 * x * x * x - 0.15 * x * x - 3.29 * x + 0.097)*0.5;
	//return (1.4 * x * x * x - 0.082 * x * x - 1.4 * x + 0.08);
	//return ((0.8 * x + 0.4) * (0.8 * x + 0.4) - 0.2)*0.75;
	//return af::sigmoid(3 * (x+0.3) + 0.😎 - af::sigmoid(3 * (2*x+0.3)-0.8);
	//return af::sigmoid_der(4*x);
	//return af::tanh(5*(x-0.5));
	//if (x < 0.2 && x > -0.4) return 0.6;
	//return 0;
	if (x < -0.5) return 1;
	if (x < 0.1) return -0.6; 
	if (x < 0.4) return 0.1;
	return 0.5;
	//return 0.5;
	//return af::sigmoid(x);
}

namespace egn = Eigen; 

// connecting nets, better orders

int main()
{
	NNetwork nnet(1, { 5,3,1 }, true, { af::tanh }, { 0.3 });

	Layer* nl = nnet.insertLayerBetween(HiddenLayer(5, af::linear, 0.3), 2,3);
	nnet.connectLayers(2, 5);

	nnet.showConnections();
	nnet.showLayers();

	nnet.setCalcOrder();
	nnet.showConnections();
	nnet.showCalcOrder();


	nnet.setLearningOrder();
	nnet.showLearningOrder();

	nnet.showLayers();

	for (int i = 0; i < 1000; i++)
	{
		float input = getRandDouble();
		float tar_output = input*input*input;
		nnet.setInput(egn::Matrix<double, 1, 1>(input));
		nnet.setTargetOutput(egn::Matrix<double, 1, 1>(tar_output));
		nnet.calcOutput();
		nnet.correctWeightsOneByOne(1);
		std::cout << std::fixed << std::showpos << std::setprecision(6) << input << ": "<< tar_output - nnet.getOutputLayer()->getOutput()[0] << std::endl;
	}

	//nnet.showLayers();

	return 0;
}

	/*
	std::vector<Layer*> layers;

	InputLayer li = InputLayer(1);
	ConstLayer l0 = ConstLayer(1);
	Layer l1 = OutputLayer(5, af::linear);
	Layer l2 = OutputLayer(5, af::linear);
	OutputLayer l3 = OutputLayer(1, af::linear);

	// GRID ONEBYONE

	l1.connectBack(&li);
	l2.connectBack(&li);
	l1.connectBack(&l0);
	l2.connectBack(&l0);
	l3.connectBack(&l0);
	l3.connectBack(&l1);
	l3.connectBack(&l2);

	layers.push_back(&li);
	layers.push_back(&l0);
	layers.push_back(&l1);
	layers.push_back(&l2);
	layers.push_back(&l3);

	const int NP = 18;
	std::vector<float> inputs_vec(NP);
	std::iota(inputs_vec.begin(), inputs_vec.end(), -NP / 2);
	std::for_each(inputs_vec.begin(), inputs_vec.end(), [NP](float& n) { n /= 1.0 * (NP / 2); });

	egn::Matrix<double, 1, 1> d(-5);

	for (int i = 0; i < 100; i++)
	{
		float eps = 0;
		std::vector<float> inputs_temp = inputs_vec;

		// teaching on vector
		for (int sample = 0; sample < inputs_vec.size(); sample++)
		{
			//float input = (std::rand()%1000)/1000.0*2-1;
			int id = static_cast<int>(10000*getRandDouble()) % inputs_temp.size();
			d << inputs_temp[id];
			//std::cout << input << std::endl;
			inputs_temp.erase(inputs_temp.begin() + id);

			li.setInput(d);
			l3.setTargetOutput(d);

			for (auto itr = layers.rbegin(); itr != layers.rend(); ++itr)
			{
				(*itr)->calcSigma();
			}
			for (auto itr = layers.rbegin(); itr != layers.rend(); ++itr)
			{
				(*itr)->calcDelta();
			}
			for (auto itr = layers.rbegin(); itr != layers.rend(); ++itr)
			{
				(*itr)->correctAllWeights();
			}
			for (auto& l : layers)
			{
				l->calcOutput();
			}
			//.showOutput();
			//std::cout << l3.getTarget() - l3.getOutput() << std::endl;
		}

		//getting error
		for (auto in : inputs_vec)
		{
			d << in;
			li.setInput(d);
			l3.setTargetOutput(d);
			for (auto& l : layers)
			{
				l->calcOutput();
			}
			//.showOutput();
			//std::cout << l3.getTarget() - l3.getOutput() << std::endl;
			eps += (l3.getTarget() - l3.getOutput())[0];
		}
		std::cout << eps / inputs_vec.size() << std::endl;

	}

	return 0;
	*/
