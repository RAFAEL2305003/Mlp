#include "../include/mlp.h"

int main()
{
    Mlp mlp(100'000, 0.001, 2, 2, 1);
    mlp.forward();

    // backward stuff
    // double d_e = MSE_derivative(Y, y); // E(MSE) derivative with respect to ŷ
    // Matrix d_zo = ReLU_derivative(zo); // a(ReLU) derivative with respect to zo

    // Matrix d_ezo = d_zo;
    // d_ezo *= d_e; // MSE derivative * ReLU derivative
    // d_ezo.print();
    // Matrix d_w1 = d_ezo;
    // d_w1(0, 0) *= a(0, 0);
    // d_w1(0, 1) *= a(0, 1);
    return 0;
}