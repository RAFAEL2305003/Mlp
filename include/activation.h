#pragma once

#include "math/matrix.h"

namespace activation
{
	enum class type
	{
		relu,
		sigmoid,
		softmax,
		argmax
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

	math::matrix<double> softmax(const math::matrix<double>& z)
	{
		auto [rows, cols] = z.shape();
		math::matrix<double> r(rows, cols);

		for(size_t i = 0; i < rows; i++)
		{
			double max_val = z(i, 0);
			for(size_t k = 1; k < cols; k++)
			{
				max_val = std::max(z(i, k), max_val);
			}

			double sum = 0.0;
			for(size_t k = 0; k < cols; k++)
			{
				sum += std::exp(z(i, k) - max_val);
			}

			for(size_t j = 0; j < cols; j++)
			{
				r(i, j) = std::exp(z(i, j) - max_val) / sum;	
			}
		}

		return r;
	}

	std::vector<size_t> argmax(const math::matrix<double>& z)
	{
		auto [rows, cols] = z.shape();
		std::vector<size_t> indices(rows);
		for(size_t i = 0; i < rows; i++)
		{
			double max_val = z(i, 0);
			size_t max_idx = 0;
			for(size_t j = 1; j < cols; j++)
			{
				if(max_val < z(i, j))
				{
					max_val = z(i, j);
					max_idx = j;
				}
			}
			indices[i] = max_idx;	
		}

		return indices;
	}
};
