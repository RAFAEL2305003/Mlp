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

	math::matrix<float> max_forward(const math::matrix<float>& x,
									 std::size_t channels,
									 std::size_t input_size,
									 std::size_t pool_size,
									 std::size_t stride)
	{
		size_t batch_size = x.shape().first;
		size_t cols = x.shape().second;

		assert(cols == channels * input_size);

		std::size_t out_size = output_size(input_size, pool_size, stride);
		math::matrix<float> y(batch_size, channels * out_size);

		const float* x_p = x.elements;
		float* y_p = y.elements;
		std::size_t is = input_size;
		std::size_t ps = pool_size;
		std::size_t os = out_size;
		std::size_t st = stride;
		std::size_t cs = channels;

		math::q.parallel_for(sycl::range<3>(batch_size, channels, out_size), [=](sycl::id<3> idx) {
			std::size_t i = idx[0];
			std::size_t c = idx[1];
			std::size_t out_pos = idx[2];
			std::size_t window_start = out_pos * st;
			float max_value = -std::numeric_limits<float>::infinity();

			for(std::size_t k = 0; k < ps; k++)
			{
				float value = x_p[i * (cs * is) + (c * is) + window_start + k];
				if(value > max_value)
				{
					max_value = value;
				}
			}

			y_p[i * (cs * os) + (c * os) + out_pos] = max_value;
		});

		return y;
	}

	math::matrix<float> avg_forward(const math::matrix<float>& x,
									 std::size_t channels,
									 std::size_t input_size,
									 std::size_t pool_size,
									 std::size_t stride)
	{
		size_t batch_size = x.shape().first;
		size_t cols = x.shape().second;

		assert(cols == channels * input_size);

		std::size_t out_size = output_size(input_size, pool_size, stride);
		math::matrix<float> y(batch_size, channels * out_size);

		const float* x_p = x.elements;
		float* y_p = y.elements;
		std::size_t is = input_size;
		std::size_t ps = pool_size;
		std::size_t os = out_size;
		std::size_t st = stride;
		std::size_t cs = channels;
		float inv_ps = 1.0 / static_cast<float>(pool_size);

		math::q.parallel_for(sycl::range<3>(batch_size, channels, out_size), [=](sycl::id<3> idx) {
			std::size_t i = idx[0];
			std::size_t c = idx[1];
			std::size_t out_pos = idx[2];
			std::size_t window_start = out_pos * st;
			float sum = 0.0;

			for(std::size_t k = 0; k < ps; k++)
			{
				sum += x_p[i * (cs * is) + (c * is) + window_start + k];
			}

			y_p[i * (cs * os) + (c * os) + out_pos] = sum * inv_ps;
		});

		return y;
	}

	math::matrix<float> forward(const math::matrix<float>& x,
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

	math::matrix<float> max_backward(const math::matrix<float>& x,
									  const math::matrix<float>& delta,
									  std::size_t channels,
									  std::size_t input_size,
									  std::size_t pool_size,
									  std::size_t stride)
	{
		size_t batch_size = x.shape().first;
		size_t cols = x.shape().second;
		size_t delta_rows = delta.shape().first;
		size_t delta_cols = delta.shape().second;

		std::size_t out_size = output_size(input_size, pool_size, stride);

		assert(cols == channels * input_size);
		assert(delta_rows == batch_size);
		assert(delta_cols == channels * out_size);

		math::matrix<float> dx(batch_size, cols);

		const float* x_p = x.elements;
		const float* delta_p = delta.elements;
		float* dx_p = dx.elements;
		std::size_t is = input_size;
		std::size_t ps = pool_size;
		std::size_t os = out_size;
		std::size_t st = stride;
		std::size_t cs = channels;

		math::q.parallel_for(sycl::range<3>(batch_size, channels, input_size), [=](sycl::id<3> idx) {
			std::size_t i = idx[0];
			std::size_t c = idx[1];
			std::size_t input_idx = idx[2];
			float sum = 0.0;

			for(std::size_t out_pos = 0; out_pos < os; out_pos++)
			{
				std::size_t window_start = out_pos * st;
				if(input_idx >= window_start && input_idx < window_start + ps)
				{
					std::size_t max_idx = window_start;
					float max_value = x_p[i * (cs * is) + (c * is) + max_idx];

					for(std::size_t k = 1; k < ps; k++)
					{
						std::size_t candidate_idx = window_start + k;
						float value = x_p[i * (cs * is) + (c * is) + candidate_idx];
						if(value > max_value)
						{
							max_value = value;
							max_idx = candidate_idx;
						}
					}

					if(input_idx == max_idx)
					{
						sum += delta_p[i * (cs * os) + (c * os) + out_pos];
					}
				}
			}

			dx_p[i * (cs * is) + (c * is) + input_idx] = sum;
		});

		return dx;
	}

	math::matrix<float> avg_backward(const math::matrix<float>& delta,
									  std::size_t channels,
									  std::size_t input_size,
									  std::size_t pool_size,
									  std::size_t stride)
	{
		size_t batch_size = delta.shape().first;
		size_t delta_cols = delta.shape().second;

		std::size_t out_size = output_size(input_size, pool_size, stride);

		assert(delta_cols == channels * out_size);

		math::matrix<float> dx(batch_size, channels * input_size);

		const float* delta_p = delta.elements;
		float* dx_p = dx.elements;
		std::size_t is = input_size;
		std::size_t ps = pool_size;
		std::size_t os = out_size;
		std::size_t st = stride;
		std::size_t cs = channels;
		float inv_ps = 1.0 / static_cast<float>(pool_size);

		math::q.parallel_for(sycl::range<3>(batch_size, channels, input_size), [=](sycl::id<3> idx) {
			std::size_t i = idx[0];
			std::size_t c = idx[1];
			std::size_t input_idx = idx[2];
			float sum = 0.0;

			for(std::size_t out_pos = 0; out_pos < os; out_pos++)
			{
				std::size_t window_start = out_pos * st;
				if(input_idx >= window_start && input_idx < window_start + ps)
				{
					sum += delta_p[i * (cs * os) + (c * os) + out_pos] * inv_ps;
				}
			}

			dx_p[i * (cs * is) + (c * is) + input_idx] = sum;
		});

		return dx;
	}

	math::matrix<float> backward(const math::matrix<float>& x,
								  const math::matrix<float>& delta,
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
