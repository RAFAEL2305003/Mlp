#pragma once

#include <algorithm>
#include "math/matrix.h"

namespace loss
{
	enum class type { 
		mse,
		bce,
		ce
	};

	double mse(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();
		
		assert(cols == 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);
	
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
		
		assert(cols == 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

		math::matrix<double> loss(rows, cols);
		std::cout << "Batch size = " << rows << "\n";
		for(size_t i = 0; i < rows; i++)
		{
			loss(i, 0) = (2.0 / rows) * (y(i, 0) - Y(i, 0));
		}

		return loss;
	}
	
	double ce(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();

		assert(cols > 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

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

	math::matrix<double> ce_derivative(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();
		assert(rows == Y.shape().first && cols == Y.shape().second);

		math::matrix<double> grad(rows, cols);
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				grad(i, j) = (y(i, j) - Y(i, j)) / static_cast<double>(rows);
			}
		}
		return grad;
	}

	double bce(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();

		assert(rows == Y.shape().first && cols == Y.shape().second);
		assert(cols == 1);

		const double epsilon = 1e-15;
		double loss = 0.0;

		for(size_t i = 0; i < rows; i++)
		{
			double pred = std::clamp(y(i, 0), epsilon, 1.0 - epsilon);
			double target = Y(i, 0);
			loss += -(target * std::log(pred) + (1.0 - target) * std::log(1.0 - pred));
		}
		return loss / rows;
	}

	math::matrix<double> bce_derivative(const math::matrix<double>& y,
										 const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();

		assert(rows == Y.shape().first && cols == Y.shape().second);
		assert(cols == 1);

		math::matrix<double> grad(rows, cols);
		const double epsilon = 1e-15;

		for(size_t i = 0; i < rows; i++)
		{
			double pred = std::clamp(y(i, 0), epsilon, 1.0 - epsilon);
			double target = Y(i, 0);
			grad(i, 0) = ((pred - target) / (pred * (1.0 - pred))) / rows;
		}
		return grad;
	}
};
