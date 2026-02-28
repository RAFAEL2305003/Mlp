#pragma once

#include "matrix/matrix.h"

// TODO: add template here, too
Matrix<double> ReLU(const Matrix<double>& z)
{
    auto [rows, cols] = z.shape();
    Matrix<double> r(rows, cols);

    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            r(i, j) = std::max(0.0, z(i, j));
        }
    }

    return r;
}

Matrix<int> ReLU_derivative(const Matrix<double>& a)
{
    auto [rows, cols] = a.shape();
    Matrix<int> r(rows, cols);
    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            r(i, j) = (a(i, j) > 0) ? 1 : 0;
        }
    }

    return r;
}