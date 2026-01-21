#include "MathFuncs.hpp"

// Factorial operator
int fact(int n){
    int out = n;
    for(int i = n-1; i > 1; i--){
        out *= i;
    }
    return out;
}

// Combinations operator
int choose(int n, int k){
    return fact(n) / ( fact(k) * fact (n-k) );
}
