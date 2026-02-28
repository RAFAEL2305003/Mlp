#pragma once

#include "matrix/matrix.h"

double MSE_derivative(const Matrix<int>& Y, const Matrix<double>& y)
{
    double result = 0.0;
    auto [rows, cols] = y.shape();
    for(size_t i = 0; i < rows; i++)
    {
        result += (Y(i, 0) - y(i, 0));
    }

    return (-2.0/rows) * result;
}