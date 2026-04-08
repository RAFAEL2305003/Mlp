#pragma once

#include "math/matrix.h"

// TODO: Add CrossEntropy loss

namespace loss
{
	enum class type { 
		mse
	};

	double mse(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();
		double result = 0.0;
		for(size_t i = 0; i < rows; i++)
		{
			result += std::pow((y(i, 0) - Y(i, 0)), 2);
		}

		return result;
	}

	math::matrix<double> mse_derivative(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();
		math::matrix<double> result(rows, cols);
		for(size_t i = 0; i < rows; i++)
		{
			result(i, 0) = (2.0 / rows) * (y(i, 0) - Y(i, 0));
		}

		return result;
	}
};
