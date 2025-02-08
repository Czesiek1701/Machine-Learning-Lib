#include <iostream>
#include "SMLGrid.h"
#include <fstream>
#include<map>
#include <algorithm>
#include <numeric>

float refFun(float x)
{
	//x = (-x);
	//return (1-(x-0.2) * (x-0.5) - 0.5*(x-0.5)*(x - 0.5))*0.5+0.2;
	//return 0.3 - 0.5 * x;
	//return (7 * x * x * x - 0.15 * x * x - 3.29 * x + 0.097)*0.5;
	//return (1.4 * x * x * x - 0.082 * x * x - 1.4 * x + 0.08);
	//return ((0.8 * x + 0.4) * (0.8 * x + 0.4) - 0.2)*0.75;
	return 0.3-af::sigmoid(6*(x-0.4))+ af::sigmoid(6 * (x - 0.1));
	//return af::tanh(5*(x-0.5));
	if (x < 0.2 && x > -0.4) return 0.6;
	return 0;
	if (x < -0.1) return 0;
	if (x < 0.1) return 0.3;
	if (x < 0.4) return 0.6;
	return 0.8;
	//return 0.5;
	//return af::sigmoid(x);
}

int main()
{

	MLGrid grid;

	grid.createGrid(1, { 8, 6,3,1 }, { &af::linear,&af::sigmoid , &af::linear, &af::linear,af::tanh,&af::linear,&af::linear });
	grid.initializeGrid();

	grid.setInput({ 0 });
	grid.calcOutput();
	grid.showGrid();

	std::ofstream refFile("graphs/output.txt");

	float eps = 0;

	std::vector<float> inputs_vec(61);
	std::iota(inputs_vec.begin(), inputs_vec.end(), -30);
	std::for_each(inputs_vec.begin(), inputs_vec.end(), [](float& n) { n /= 30.0; });

	for (int show = 0; show < 100; show++) 
	{
		eps = 0;
		for (int sample = 0; sample < inputs_vec.size(); sample++)
		{
			//float input = (std::rand()%1000)/1000.0*2-1;
			int id = rand() % inputs_vec.size();
			float input = inputs_vec[id];
			//std::cout << input << std::endl;
			//inputs_vec.erase(inputs_vec.begin()+id);

			grid.setInput({ input });

			//for (int i = 0; i < 10; i++)
			{
				grid.calcOutput();
				grid.correctWeights({ refFun(input)});
			}

			//grid.showGrid();

			eps += abs(refFun(input) - (*grid.getOutput())[0]);
		}
		std::cout << eps/ inputs_vec.size() << std::endl;
	}

	//std::cout << "przed loop" << std::endl;
	for ( int i=0; i<inputs_vec.size(); i++ )
	{
		float input = inputs_vec[i];
		//std::cout << input << std::endl;
		grid.setInput({ input });
		grid.calcOutput();
		//std::cout << "float " << input << std::endl;
		refFile << input << "\t" << refFun(input) << "\t" << (*grid.getOutput())[0] << std::endl;
		//std::cout << input << "\t" << refFun(input) << "\t" << (*grid.getOutput())[0] << std::endl;
	}

	grid.setInput({ -0.1 });
	grid.calcOutput();
	grid.showGrid();

	refFile.close();



	return 0;
}