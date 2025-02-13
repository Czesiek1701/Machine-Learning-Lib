#include<iostream>
#include"SMLgrid.h"

int main()
{
    MLGrid mlgrid;

    std::vector<float> in = {0,1,2,3,4};
    

   // mlgrid.createGrid( &in, {5,5,1}, {&af::linear,&af::linear,&af::linear} );
    mlgrid.createGrid();

    std::cout << " hghjg" << std::endl;

    return 0;
}