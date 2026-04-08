#pragma once

#include "math/matrix.h"

// TODO: Add CrossEntropy loss

namespace loss
{
	enum class type { 
		mse,
		cross_entropy
	};

	double mse(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();
		double loss = 0.0;
		for(size_t i = 0; i < rows; i++)
		{
			loss += std::pow((y(i, 0) - Y(i, 0)), 2);
		}

		return loss;
	}

	math::matrix<double> mse_derivative(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();
		math::matrix<double> loss(rows, cols);
		std::cout << "Batch size = " << rows << "\n";
		for(size_t i = 0; i < rows; i++)
		{
			loss(i, 0) = (2.0 / rows) * (y(i, 0) - Y(i, 0));
		}

		return loss;
	}
	
	// warning: this only works when the labels are only 0 or 1
	double cross_entropy(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();

		const double epsilon = 1e-15;
		double loss = 0.0;
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				if(Y(i, j) != 0)
				{
					double p = std::max(y(i, j), epsilon);
					loss += -std::log(p);
					break;
				}
			}
		}
		
		return loss / rows;
	}

	math::matrix<double> cross_entropy_derivative(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		return y - Y;
	}
};
