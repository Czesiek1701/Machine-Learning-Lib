#include <iostream>
#include "SMLGrid.h"
#include <fstream>
#include<map>

float refFun(float x)
{
	//x = (-x);
	//return (1-(x-0.2) * (x-0.5) - 0.5*(x-0.5)*(x - 0.5))*0.5+0.2;
	//return 0.3 - 0.5 * x;
	//return (7 * x * x * x - 0.15 * x * x - 3.29 * x + 0.097)*0.5;
	//return (1.4 * x * x * x - 0.082 * x * x - 1.4 * x + 0.08);
	return ((0.8 * x + 0.4) * (0.8 * x + 0.4) - 0.2)*0.75;
	//return 0.5;
	//return af::sigmoid(x);
}

int main()
{

	MLGrid grid;

	grid.createGrid(1, { 20,8,8,1 }, { &af::linear, &af::tanh,&af::linear,&af::linear, &af::linear, &af::sigmoid, &af::linear, &af::sigmoid, &af::linear });
	grid.initializeGrid();

	grid.setInput({ 0 });
	grid.calcOutput();
	grid.showGrid();

	std::ofstream refFile("graphs/output.txt");

	float eps = 0;

	for (int show = 0; show < 1000; show++) 
	{
		eps = 0;
		std::vector<float> inputs_vec{ -1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1 };
		for (int sample = 0; sample < 11; sample++)
		{
			//float input = (std::rand()%1000)/1000.0*2-1;
			int id = rand() % inputs_vec.size();
			float input = inputs_vec[id];
			//std::cout << input << std::endl;
			inputs_vec.erase(inputs_vec.begin()+id);

			grid.setInput({ input });

			//for (int i = 0; i < 10; i++)
			{
				grid.calcOutput();
				grid.correctWeights({ refFun(input)});
			}

			//grid.showGrid();

			eps += abs(refFun(input) - (*grid.getOutput())[0]);
		}
		std::cout << eps/11.0 << std::endl;
	}

	for (int sample = 10; sample > -11; sample -= 2)
	{
		float input = sample/10.0;
		grid.setInput({ input });
		grid.calcOutput();
		refFile << input << "\t" << refFun(input) << "\t" << (*grid.getOutput())[0] << std::endl;
		std::cout << input << "\t" << refFun(input) << "\t" << (*grid.getOutput())[0] << std::endl;
	}

	grid.showGrid();

	refFile.close();



	return 0;
}