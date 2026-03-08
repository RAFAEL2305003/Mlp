#pragma once

#include "matrix/matrix.h"

// TODO: add template here, too
Matrix<double> relu(const Matrix<double>& a)
{
    auto [rows, cols] = a.shape();
    Matrix<double> r(rows, cols);

    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            r(i, j) = std::max(0.0, a(i, j));
        }
    }

    return r;
}

Matrix<double> relu_derivative(const Matrix<double>& a)
{
    auto [rows, cols] = a.shape();
    Matrix<double> r(rows, cols);
    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            r(i, j) = (a(i, j) > 0) ? 1 : 0;
        }
    }

    return r;
}

Matrix<double> sigmoid(const Matrix<double>& z)
{
    auto [rows, cols] = z.shape();
    Matrix<double> r(rows, cols);
    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            r(i, j) = 1 / (1 + std::exp(-z(i, j)));
        }
    }

    return r;
}

Matrix<double> sigmoid_derivative(const Matrix<double>& a)
{
    auto [rows, cols] = a.shape();
    Matrix<double> r(rows, cols);
    for(size_t i = 0; i < rows; i++)
    {
        for(size_t j = 0; j < cols; j++)
        {
            r(i, j) = a(i, j) * (1 - a(i, j));
        }
    }

    return r;
}
