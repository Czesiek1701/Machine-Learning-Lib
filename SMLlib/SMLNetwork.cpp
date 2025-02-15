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
		layers_all.push_back(input_layer.get());
	}
	else if(etas.size() == 1 && actFuns.size() > 1)
	{
		for (int ln = 0; ln < lastN; ++ln)
		{
			hidden_layers.push_back(std::make_unique<HiddenLayer>(neuron_nums[ln], actFuns[ln], etas[0]));
			layers_all.push_back(hidden_layers.back().get());
		}

		output_layer = std::make_unique<OutputLayer>(neuron_nums[lastN], actFuns[lastN], etas[0]);
		layers_all.push_back(input_layer.get());
	}
	else if (etas.size() > 1 && actFuns.size() == 1)
	{
		for (int ln = 0; ln < lastN; ++ln)
		{
			hidden_layers.push_back(std::make_unique<HiddenLayer>(neuron_nums[ln], actFuns[0], etas[ln]));
			layers_all.push_back(hidden_layers.back().get());
		}

		output_layer = std::make_unique<OutputLayer>(neuron_nums[lastN], actFuns[0], etas[lastN]);
		layers_all.push_back(input_layer.get());
	}
	else if (etas.size() == 1 && actFuns.size() == 1)
	{
		for (int ln = 0; ln < lastN; ++ln)
		{
			hidden_layers.push_back(std::make_unique<HiddenLayer>(neuron_nums[ln], actFuns[0], etas[0]));
			layers_all.push_back(hidden_layers.back().get());
		}

		output_layer = std::make_unique<OutputLayer>(neuron_nums[lastN], actFuns[0], etas[0]);
		layers_all.push_back(input_layer.get());
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
	calc_order = { 0 ,1 , 2 ,3 };
}

void NNetwork::setLearningOrder()
{
	learning_order = { 3, 2, 1 ,0 };
}


//void NNetwork::setInput(const egn::Matrix<double, egn::Dynamic, 1>& in)
//{
//	input_layer->setInput(in);
//}
//
//void NNetwork::setTargetOutput(const egn::Matrix<double, egn::Dynamic, 1>& in)
//{
//	output_layer->setTargetOutput(in);
//}



void NNetwork::showLayers() const
{
	for (int ln = 0; ln < layers_all.size(); ++ln)
	{
		std::cout << "ln" << std::endl;
		layers_all[ln]->showLayer();
	}
}