#pragma once

#include "math/matrix.h"

namespace activation
{
	enum class type
	{
		relu,
		sigmoid,
		softmax
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
		for (std::size_t i = 0; i < rows; i++)
		{
			double max_val = z(i, 0);
			for (std::size_t j = 1; j < cols; j++)
				max_val = std::max(max_val, z(i, j));

				double sum = 0;

				for (std::size_t j = 0; j < cols; j++)
				{
					r(i,j) = std::exp(z(i,j) - max_val);
					sum += r(i,j);
				}

				for (std::size_t j = 0; j < cols; j++)
					r(i,j) /= sum;
		}

		return r;
	}

	math::matrix<double> softmax_derivative(const math::matrix<double>& a, const math::matrix<double>& da)
	{
		auto [rows, cols] = a.shape();
		math::matrix<double> r(rows, cols);
		for (std::size_t i = 0; i < rows; i++)
		{
			for (std::size_t j = 0; j < cols; j++)
			{
				double grad = 0;

				for (std::size_t k = 0; k < cols; k++)
				{
					if (j == k)
						grad += da(i,k) * a(i,j) * (1 - a(i,j));
					else
						grad += da(i,k) * (-a(i,j) * a(i,k));
				}

				r(i,j) = grad;
			}
		}

		return r;
	}
};
