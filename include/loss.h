#pragma once

#include "matrix/matrix.h"

Matrix<double> mse_derivative(const Matrix<int>& Y, const Matrix<double>& y)
{
    auto [rows, cols] = y.shape();
    Matrix<double> result(rows, cols);
    for(size_t i = 0; i < rows; i++)
    {
        result(i, 0) = (2.0 / rows) * (y(i, 0) - Y(i, 0));
    }

    return result;
}
