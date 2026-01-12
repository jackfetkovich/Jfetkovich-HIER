#include<iostream>
#include "MPPI.hpp"

MPPI::MPPI(int K, int T) : K(K), T(T){};
void MPPI::output(){
    std::cout << "K: " << K << ", T: " << T << std::endl;
};
