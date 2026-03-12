#pragma once

#include "math/matrix.h"

namespace loss
{
	enum class type { 
		mse
	};

	math::matrix<double> mse_derivative(const math::matrix<int>& Y, const math::matrix<double>& y)
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
