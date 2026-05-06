#pragma once

#include <cassert>
#include <limits>
#include <stdexcept>
#include "math/matrix.h"

namespace pooling
{
	enum class type
	{
		max,
		avg
	};

	struct config
	{
		pooling::type pool_type;
		size_t pool_size;
		size_t stride;
	};

	std::size_t output_size(std::size_t input_size,
							std::size_t pool_size,
							std::size_t stride)
	{
		assert(pool_size > 0);
		assert(stride > 0);
		assert(input_size >= pool_size);
		return ((input_size - pool_size) / stride) + 1;
	}

	math::matrix<double> max_forward(const math::matrix<double>& x,
									 std::size_t channels,
									 std::size_t input_size,
									 std::size_t pool_size,
									 std::size_t stride)
	{
		auto [batch_size, cols] = x.shape();

		assert(cols == channels * input_size);

		std::size_t out_size = output_size(input_size, pool_size, stride);
		math::matrix<double> y(batch_size, channels * out_size);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t c = 0; c < channels; c++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					std::size_t window_start = out_pos * stride;
					double max_value = std::numeric_limits<double>::lowest();

					for(std::size_t k = 0; k < pool_size; k++)
					{
						double value = x(i, (c * input_size) + window_start + k);
						if(value > max_value)
						{
							max_value = value;
						}
					}

					y(i, (c * out_size) + out_pos) = max_value;
				}
			}
		}

		return y;
	}

	math::matrix<double> avg_forward(const math::matrix<double>& x,
									 std::size_t channels,
									 std::size_t input_size,
									 std::size_t pool_size,
									 std::size_t stride)
	{
		auto [batch_size, cols] = x.shape();

		assert(cols == channels * input_size);

		std::size_t out_size = output_size(input_size, pool_size, stride);
		math::matrix<double> y(batch_size, channels * out_size);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t c = 0; c < channels; c++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					std::size_t window_start = out_pos * stride;
					double sum = 0.0;

					for(std::size_t k = 0; k < pool_size; k++)
					{
						sum += x(i, (c * input_size) + window_start + k);
					}

					y(i, (c * out_size) + out_pos) = sum / pool_size;
				}
			}
		}

		return y;
	}

	math::matrix<double> forward(const math::matrix<double>& x,
								 type pool_type,
								 std::size_t channels,
								 std::size_t input_size,
								 std::size_t pool_size,
								 std::size_t stride)
	{
		switch(pool_type)
		{
			case type::max:
			{
				return max_forward(x, channels, input_size, pool_size, stride);
			}
			case type::avg:
			{
				return avg_forward(x, channels, input_size, pool_size, stride);
			}
			default:
			{
				throw std::runtime_error("Invalid pooling type.");
			}
		}
	}

	math::matrix<double> max_backward(const math::matrix<double>& x,
									  const math::matrix<double>& delta,
									  std::size_t channels,
									  std::size_t input_size,
									  std::size_t pool_size,
									  std::size_t stride)
	{
		auto [batch_size, cols] = x.shape();
		auto [delta_rows, delta_cols] = delta.shape();

		std::size_t out_size = output_size(input_size, pool_size, stride);

		assert(cols == channels * input_size);
		assert(delta_rows == batch_size);
		assert(delta_cols == channels * out_size);

		math::matrix<double> dx(batch_size, cols, 0.0);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t c = 0; c < channels; c++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					std::size_t window_start = out_pos * stride;
					std::size_t max_idx = window_start;
					double max_value = x(i, (c * input_size) + max_idx);

					for(std::size_t k = 1; k < pool_size; k++)
					{
						std::size_t input_idx = window_start + k;
						double value = x(i, (c * input_size) + input_idx);
						if(value > max_value)
						{
							max_value = value;
							max_idx = input_idx;
						}
					}

					dx(i, (c * input_size) + max_idx) += delta(i, (c * out_size) + out_pos);
				}
			}
		}

		return dx;
	}

	math::matrix<double> avg_backward(const math::matrix<double>& delta,
									  std::size_t channels,
									  std::size_t input_size,
									  std::size_t pool_size,
									  std::size_t stride)
	{
		auto [batch_size, delta_cols] = delta.shape();

		std::size_t out_size = output_size(input_size, pool_size, stride);

		assert(delta_cols == channels * out_size);

		math::matrix<double> dx(batch_size, channels * input_size, 0.0);

		for(std::size_t i = 0; i < batch_size; i++)
		{
			for(std::size_t c = 0; c < channels; c++)
			{
				for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
				{
					std::size_t window_start = out_pos * stride;
					double gradient = delta(i, (c * out_size) + out_pos) / pool_size;

					for(std::size_t k = 0; k < pool_size; k++)
					{
						dx(i, (c * input_size) + window_start + k) += gradient;
					}
				}
			}
		}

		return dx;
	}

	math::matrix<double> backward(const math::matrix<double>& x,
								  const math::matrix<double>& delta,
								  type pool_type,
								  std::size_t channels,
								  std::size_t input_size,
								  std::size_t pool_size,
								  std::size_t stride)
	{
		switch(pool_type)
		{
			case type::max:
			{
				return max_backward(x, delta, channels, input_size, pool_size, stride);
			}
			case type::avg:
			{
				return avg_backward(delta, channels, input_size, pool_size, stride);
			}
			default:
			{
				throw std::runtime_error("Invalid pooling type.");
			}
		}
	}
};