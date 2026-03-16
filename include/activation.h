#pragma once

#include "math/matrix.h"

// TODO: Add softmax activation

namespace activation
{
	enum class type
	{
		relu,
		sigmoid,
	};

	math::matrix<double> relu(const math::matrix<double>& a)
	{
		auto [rows, cols] = a.shape();
		math::matrix<double> r(rows, cols);

		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				r(i, j) = std::max(0.0, a(i, j));
			}
		}

		return r;
	}

	math::matrix<double> relu_derivative(const math::matrix<double>& a)
	{
		auto [rows, cols] = a.shape();
		math::matrix<double> r(rows, cols);
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				r(i, j) = (a(i, j) > 0) ? 1.0 : 0.0;
			}
		}

		return r;
	}

	math::matrix<double> sigmoid(const math::matrix<double>& z)
	{
		auto [rows, cols] = z.shape();
		math::matrix<double> r(rows, cols);
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				r(i, j) = 1 / (1 + std::exp(-z(i, j)));
			}
		}

		return r;
	}

	math::matrix<double> sigmoid_derivative(const math::matrix<double>& a)
	{
		auto [rows, cols] = a.shape();
		math::matrix<double> r(rows, cols);
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				r(i, j) = a(i, j) * (1 - a(i, j));
			}
		}

		return r;
	}
};
