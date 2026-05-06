#pragma once

#include <cassert>
#include <stdexcept>
#include "math/matrix.h"

namespace conv1d
{
	enum class type
	{
		valid,
		same,
		full
	};

	struct config
	{
		size_t filters;
		size_t kernel_size;
		conv1d::type conv_type;
		size_t stride;
	};

	std::size_t padding(type conv_type, std::size_t kernel_size)
	{
		switch(conv_type)
		{
			case type::valid:
			{
				return 0;
			}
			case type::same:
			{
				assert(kernel_size % 2 == 1);
				return kernel_size / 2;
			}
			case type::full:
			{
				return kernel_size - 1;
			}
			default:
			{
				throw std::runtime_error("Invalid conv1d type.");
			}
		}
	}

	std::size_t output_size(std::size_t input_size,
							std::size_t kernel_size,
							std::size_t stride,
							std::size_t padding)
	{
		assert(stride > 0);
		assert(input_size + (2 * padding) >= kernel_size);
		return ((input_size + (2 * padding) - kernel_size) / stride) + 1;
	}

	math::matrix<double> forward(const math::matrix<double>& x,
								 const math::matrix<double>& w,
								 const math::matrix<double>& b,
								 std::size_t stride,
								 std::size_t padding)
	{
		auto [batch_size, input_size] = x.shape();
		auto [filters, kernel_size] = w.shape();
		auto [b_rows, b_cols] = b.shape();

		assert(b_rows == 1 && b_cols == filters);

		std::size_t out_size = output_size(input_size, kernel_size, stride, padding);
		math::matrix<double> z(batch_size, filters * out_size);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t f = 0; f < filters; f++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					double sum = b(0, f);
					for(std::size_t k = 0; k < kernel_size; k++)
					{
						std::size_t padded_idx = (out_pos * stride) + k;
						if(padded_idx >= padding)
						{
							std::size_t input_idx = padded_idx - padding;
							if(input_idx < input_size)
							{
								sum += x(i, input_idx) * w(f, k);
							}
						}
					}

					z(i, (f * out_size) + out_pos) = sum;
				}
			}
		}

		return z;
	}

	math::matrix<double> input_derivative(const math::matrix<double>& delta,
										  const math::matrix<double>& w,
										  std::size_t input_size,
										  std::size_t stride,
										  std::size_t padding)
	{
		auto [batch_size, delta_cols] = delta.shape();
		auto [filters, kernel_size] = w.shape();

		assert(delta_cols % filters == 0);

		std::size_t out_size = delta_cols / filters;
		math::matrix<double> dx(batch_size, input_size, 0.0);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t f = 0; f < filters; f++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					double delta_value = delta(i, (f * out_size) + out_pos);
					for(std::size_t k = 0; k < kernel_size; k++)
					{
						std::size_t padded_idx = (out_pos * stride) + k;
						if(padded_idx >= padding)
						{
							std::size_t input_idx = padded_idx - padding;
							if(input_idx < input_size)
							{
								dx(i, input_idx) += delta_value * w(f, k);
							}
						}
					}
				}
			}
		}

		return dx;
	}

	math::matrix<double> weight_derivative(const math::matrix<double>& x,
										   const math::matrix<double>& delta,
										   std::size_t filters,
										   std::size_t kernel_size,
										   std::size_t stride,
										   std::size_t padding)
	{
		auto [batch_size, input_size] = x.shape();
		auto [delta_rows, delta_cols] = delta.shape();

		assert(batch_size == delta_rows);
		assert(delta_cols % filters == 0);

		std::size_t out_size = delta_cols / filters;
		math::matrix<double> dw(filters, kernel_size, 0.0);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t f = 0; f < filters; f++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					double delta_value = delta(i, (f * out_size) + out_pos);
					for(std::size_t k = 0; k < kernel_size; k++)
					{
						std::size_t padded_idx = (out_pos * stride) + k;
						if(padded_idx >= padding)
						{
							std::size_t input_idx = padded_idx - padding;
							if(input_idx < input_size)
							{
								dw(f, k) += x(i, input_idx) * delta_value;
							}
						}
					}
				}
			}
		}

		return dw;
	}

	math::matrix<double> bias_derivative(const math::matrix<double>& delta,
										 std::size_t filters)
	{
		auto [batch_size, delta_cols] = delta.shape();

		assert(delta_cols % filters == 0);

		std::size_t out_size = delta_cols / filters;
		math::matrix<double> db(1, filters, 0.0);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t f = 0; f < filters; f++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					db(0, f) += delta(i, (f * out_size) + out_pos);
				}
			}
		}

		return db;
	}
};
