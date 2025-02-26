// =======================
// Author: Grzegorz Czaja
// =======================
# include "SMLNetwork.h"
# include "SpecificLayers.h"

NNetwork::NNetwork(int input_size, std::vector<int> neuron_nums, bool connect, std::vector<af::afType> actFuns, std::vector<double> etas)
{
	const_layer = std::make_unique<ConstLayer>(1);
	layers_all.push_back(const_layer.get());

	input_layer = std::make_unique<InputLayer>(input_size);
	layers_all.push_back(input_layer.get());

	int lastN = neuron_nums.size() - 1;
	if (etas.size() > 1 && actFuns.size() > 1)
	{
		for (int ln = 0; ln < lastN; ++ln)
		{
			hidden_layers.push_back(std::make_unique<HiddenLayer>(neuron_nums[ln], actFuns[ln], etas[ln]));
			layers_all.push_back( hidden_layers.back().get() );
		}

		output_layer = std::make_unique<OutputLayer>(neuron_nums[lastN], actFuns[lastN], etas[lastN]);
		layers_all.push_back(output_layer.get());
	}
	else if(etas.size() == 1 && actFuns.size() > 1)
	{
		for (int ln = 0; ln < lastN; ++ln)
		{
			hidden_layers.push_back(std::make_unique<HiddenLayer>(neuron_nums[ln], actFuns[ln], etas[0]));
			layers_all.push_back(hidden_layers.back().get());
		}

		output_layer = std::make_unique<OutputLayer>(neuron_nums[lastN], actFuns[lastN], etas[0]);
		layers_all.push_back(output_layer.get());
	}
	else if (etas.size() > 1 && actFuns.size() == 1)
	{
		for (int ln = 0; ln < lastN; ++ln)
		{
			hidden_layers.push_back(std::make_unique<HiddenLayer>(neuron_nums[ln], actFuns[0], etas[ln]));
			layers_all.push_back(hidden_layers.back().get());
		}

		output_layer = std::make_unique<OutputLayer>(neuron_nums[lastN], actFuns[0], etas[lastN]);
		layers_all.push_back(output_layer.get());
	}
	else if (etas.size() == 1 && actFuns.size() == 1)
	{
		for (int ln = 0; ln < lastN; ++ln)
		{
			hidden_layers.push_back(std::make_unique<HiddenLayer>(neuron_nums[ln], actFuns[0], etas[0]));
			layers_all.push_back(hidden_layers.back().get());
		}

		output_layer = std::make_unique<OutputLayer>(neuron_nums[lastN], actFuns[0], etas[0]);
		layers_all.push_back(output_layer.get());
	}


	if (connect)
	{
		for (int i = 2; i < layers_all.size(); i++)
		{
			layers_all[i]->connectBack(layers_all[0]);	// 0-const
			layers_all[i]->connectBack(layers_all[i-1]);	
		}
	}
}

void NNetwork::setCalcOrder()
{
	//calc_order = std::vector<int>(layers_all.size());
	//std::iota(calc_order.begin(), calc_order.end(), 0); // !!!!!
	calc_order.clear();
	unvisited = layers_all;

	this->setCalcOrder(output_layer.get());
}

void NNetwork::setCalcOrder(Layer* layer)
{
	for (auto pL : layer->prev_layers)
	{
		if (std::find(unvisited.begin(), unvisited.end(), pL) != unvisited.end())
		{
			unvisited.erase(std::find(unvisited.begin(), unvisited.end(), pL));
			setCalcOrder(pL);
		}
	}
	calc_order.push_back(layer);
	std::cout << "calc order set" << std::endl;
}

void NNetwork::showCalcOrder()
{
	for (auto layer : calc_order)
	{
		std::cout << getLayerIndex(layer) << " ";
	}
	std::cout << std::endl;
}

void NNetwork::setLearningOrder()
{
	//learning_order = calc_order;
	//std::reverse(calc_order.begin(), calc_order.end()); // !!!!!
	learning_order.clear();
	unvisited = layers_all;
	this->setLearningOrder(input_layer.get());
	learning_order.push_back(const_layer.get());
	std::cout << "learning order set" << std::endl;
}

void NNetwork::setLearningOrder(Layer* layer)
{
	for (auto nL : layer->next_layers)
	{
		if (std::find(unvisited.begin(), unvisited.end(), nL) != unvisited.end())
		{
			unvisited.erase(std::find(unvisited.begin(), unvisited.end(), nL));
			setLearningOrder(nL);
		}
	}
		//std::cout << getLayerIndex(layer) << std::endl;
	learning_order.push_back(layer);
}

void NNetwork::showLearningOrder()
{
	for (auto layer : learning_order)
	{
		std::cout << getLayerIndex(layer) << " ";
	}
	std::cout << std::endl;
}


void NNetwork::setInput(const egn::Matrix<double, egn::Dynamic, 1>& in)
{
	input_layer->setInput(in);
}

void NNetwork::setTargetOutput(const egn::Matrix<double, egn::Dynamic, 1>& in)
{
	output_layer->setTargetOutput(in);
}



void NNetwork::showLayers() const
{
	for (int ln = 0; ln < layers_all.size(); ++ln)
	{
		std::cout << "--- LAYER: " << ln << std::endl;
		layers_all[ln]->showLayer();
	}
}

void NNetwork::showResult()
{
	std::cout << std::fixed << std::showpos << std::setprecision(6)
		<< " in: " << input_layer->getOutput().transpose()
		<< "\t out: " << output_layer->getOutput().transpose()
		<< "\t error: " << (output_layer->getOutput() - output_layer->getTarget()).transpose()
		<< std::endl;
}

void NNetwork::calcOutput()
{
	int it = 0;
	diff_error_output = DBL_MAX;
	while (diff_error_output > 0.01 && it<50)
	{
		diff_error_output = 0;
		for (auto layer : calc_order)
		{
			layer->calcOutput();
			diff_error_output += layer->difOutput.sum();
			//std::cout << "output sum: " << layer->output.sum() << std::endl;
		}
		it++;
		//this->showSigmas();
		//this->showOutputs();
		//std::cout << "calc iter error"<< diff_error << std::endl;
	}
}

void NNetwork::calcOutput(int nit)
{
	for (int i = 0; i < nit; i++)
	{
		for (auto layer : calc_order)
		{
			//std::cout << ln << ", ";
			layer->calcOutput();
		}
		//std::cout << std::endl;
	}
}

void NNetwork::calcSigma(int nit)
{
	for (int i = 0; i < nit; i++)
	{
		for (auto layer : learning_order)
		{
			//std::cout << ln << ", ";
			layer->calcSigma();
		}
		//std::cout << std::endl;
	}
}

void NNetwork::calcSigma()
{
	int it = 0;
	diff_error_sigma = DBL_MAX;
	while (diff_error_sigma > 0.01 && it < 50)
	{
		diff_error_sigma = 0;
		for (auto layer : learning_order)
		{
			layer->calcSigma();
			diff_error_sigma += layer->difSigma.sum();
			//std::cout << "output sum: " << layer->difSigma << std::endl;
		}
		it++;
		//std::cout << std::showpos << std::setprecision(6) << this->layers_all[0]->sigma.transpose() << std::endl;
		//std::cout << std::showpos << std::setprecision(6) << this->layers_all[1]->sigma.transpose() << std::endl;
		//std::cout << std::showpos << std::setprecision(6) << this->layers_all[2]->sigma.transpose() << std::endl;
		//std::cout << std::showpos << std::setprecision(6) << this->layers_all[3]->sigma.transpose() << std::endl;
		//std::cout << std::showpos << std::setprecision(6) << this->layers_all[4]->sigma.transpose() << std::endl;
		//this->showSigmas();
		//this->showOutputs();
		//std::cout << "calc iter error sigma "<< diff_error_sigma << std::endl;
	}
}

void NNetwork::calcDelta()
{
	for (auto layer : learning_order)
	{
		layer->calcDelta();
		//std::cout << "output sum: " << layer->output.sum() << std::endl;
	}
}

void NNetwork::correctWeightsAll()
{
	this->calcOutput();
	for (auto layer: learning_order)
	{
		layer->calcSigma();
		layer->calcDelta();
		layer->correctAllWeights();
	}
	//std::cout << std::endl;
}

void NNetwork::correctWeightsOneByOne(int itnum)
{
	for (auto layer : learning_order)
	{
		for (int nid = 0; nid < layer->getN(); nid++)
		{
			for (int i = 0; i < itnum; i++)
			{
				this->calcOutput();
				for(auto l: learning_order)
				{
					l->calcSigma();
					l->calcDelta();
				}
				layer->correctNeuronWeight(nid);
			}
		}
	}
}

void NNetwork::correctWeightsOneByOne()
{
	this->calcOutput();
	for (auto layer : learning_order)
	{
		for (int nid = 0; nid < layer->getN(); nid++)
		{
			//std::cout << "=== Layer: " << getLayerIndex(layer) << " Neuron: " << nid << " ===" << std::endl;
			//diff_error_sigma = DBL_MAX;
			//while (diff_error_sigma > 0.01)
			{
				this->calcSigma();
				this->calcDelta();
				layer->correctNeuronWeight(nid);
			}
		}
	}
}

void NNetwork::correctWeightsWinnigOne()
{
	/*
	for (auto layer : learning_order)
	{
		//for (int nid = 0; nid < layer->getN(); nid++)
		//this->showOutputs();
		int win_neuron;
		(layer->output.cwiseProduct(layer->output)).maxCoeff(&win_neuron);
		//std::cout << win_neuron << std::endl;
		std::cout << layer << std::endl;
		this->calcOutput();
		this->calcSigma();
		this->calcDelta();
		layer->correctNeuronWeight(win_neuron);
		this->calcOutput();
		std::cout << this->getOutputLayer()->getOutput()[0] << std::endl;
	}
	*/

	this->calcOutput();

	

	for (int li = 0; li < learning_order.size(); li++)
	{
		Layer* layer = learning_order[li]; 

		int win_neuron;
		(layer->output.cwiseProduct(layer->output)).maxCoeff(&win_neuron);
		//std::cout << layer << ": " << win_neuron << std::endl;
		//std::cout << win_neuron << std::endl;
		//std::cout << layer << std::endl;

		//for(int j=0; j<=li; j++)
		//{
		layer->calcNeuronSigma(win_neuron); // TO CORRECTION - correct win_neuron
		//layer->calcSigma();
		//}
		layer->calcNeuronDelta(win_neuron);
		//layer->calcDelta();

		layer->correctNeuronWeight(win_neuron); // TO CORRECTION - correct win_neuron


		/*if (li == 2)
		{
			std::cout << layers_all[li]->sigma.transpose() << std::endl;
		}*/

		//this->calcOutput();
		//std::cout << this->getOutputLayer()->getOutput()[0] << std::endl;
	}

	//for (int li = 0; li < learning_order.size(); li++)
	//{
	//	Layer* layer = learning_order[li];
	//	int win_neuron;
	//	(layer->output.cwiseProduct(layer->output)).maxCoeff(&win_neuron);
	//	layer->correctNeuronWeight(win_neuron);
	//	// std::cout << layer << ": " << win_neuron << std::endl;
	//}

}

void NNetwork::showOutput() const
{
	output_layer->showOutput();
}

void NNetwork::deleteLayer(int n)
{
	if (n<2 || n>=layers_all.size() - 1)
	{
		std::cout << "No layer to delete" << std::endl;
		return;
	}

	Layer* deleting_Layer = layers_all[n];

	//std::remove(layers_all.begin(), layers_all.end(), deleting_Layer);

	layers_all.erase(std::find(layers_all.begin(), layers_all.end(), deleting_Layer));
	
	hidden_layers.erase( std::find_if(hidden_layers.begin(), hidden_layers.end(), 
		[deleting_Layer](std::unique_ptr<HiddenLayer>& up) { return up.get()==deleting_Layer; }
		) );

	layers_all[n]->connectBack(layers_all[n - 1]);
	
	setCalcOrder();
	setLearningOrder();

}

Layer* NNetwork::addLayer(HiddenLayer hl)
{
	hidden_layers.push_back( std::make_unique<HiddenLayer>(HiddenLayer(hl)) );
	auto nLit = layers_all.insert(layers_all.end(), hidden_layers.back().get());
	(*nLit)->connectBack(const_layer.get());
	return *nLit;
}

void NNetwork::connectLayers(Layer* prev, Layer* next)
{
	next->connectBack(prev);
}

void NNetwork::connectLayers(int prev, int next)
{
	connectLayers(layers_all[prev], layers_all[next]);
}

Layer* NNetwork::insertLayerBetween(HiddenLayer hL, int prev, int next)
{
	Layer* nL = addLayer(hL);
	connectLayers(prev, getLayerIndex(nL));
	connectLayers(getLayerIndex(nL), next);
	return nL;
}

void NNetwork::showConnections()
{
	for (auto layer : calc_order)
	{
		for (auto prev : layer->prev_layers)
		{
			std::cout << getLayerIndex(prev) << " ";
		}
		std::cout << "-> " << getLayerIndex(layer) << " -> ";
		for (auto next : layer->next_layers)
		{
			std::cout << getLayerIndex(next) << " ";
		}
		std::cout << std::endl;
	}

}

void NNetwork::showWeights()
{
	std::cout << "weights" << std::endl;
	for (auto layer : calc_order)
	{
		for (auto w : layer->weight)
		{
			std::cout << w << " | ";
		}
		std::cout << std::endl;
	}

}

Layer* NNetwork::getLayer(int nid)
{
	return layers_all[nid];
}

OutputLayer* NNetwork::getOutputLayer()
{
	return output_layer.get();
}

int NNetwork::getLayerIndex(Layer* layer)
{
	return std::find(layers_all.begin(), layers_all.end(), layer) - layers_all.begin();
}

void NNetwork::showOutputs()
{
	for (auto layer : calc_order)
	{
		std::cout << getLayerIndex(layer) << ":\t" << layer->output.transpose() << std::endl;
	}

}

void NNetwork::showSigmas()
{
	std::cout << "sigmas" << std::endl;
	for (auto layer : calc_order)
	{
		std::cout << layer->sigma.transpose() << " | ";
		std::cout << std::endl;
	}
}