#pragma once

#include <stdexcept>
#include "math/matrix.h"
#include "pooling.h"

namespace nn
{
	class pooling_layer
	{
	private:
		pooling::type pool_type;
		math::matrix<float> x;
		math::matrix<float> y;
		math::matrix<float> dx;

	public:
		std::size_t input_size;
		std::size_t channels;
		std::size_t pool_size;
		std::size_t stride;
		std::size_t output_size;

		pooling_layer()
			: pool_type(pooling::type::max),
			  input_size(0),
			  channels(0),
			  pool_size(0),
			  stride(1),
			  output_size(0) {}

		pooling_layer(pooling::type pool_type,
					  std::size_t input_size,
					  std::size_t channels,
					  std::size_t pool_size,
					  std::size_t stride = 1)
			: pool_type(pool_type),
			  input_size(input_size),
			  channels(channels),
			  pool_size(pool_size),
			  stride(stride)
		{
			assert(input_size > 0);
			assert(channels > 0);
			assert(pool_size > 0);
			assert(stride > 0);

			output_size = pooling::output_size(input_size, pool_size, stride);
		}

		math::matrix<float> forward(const math::matrix<float>& x)
		{
			assert(x.shape().second == channels * input_size);

			this->x = x;

			switch(pool_type)
			{
				case pooling::type::max:
				case pooling::type::avg:
				{
					y = pooling::forward(x, pool_type, channels, input_size, pool_size, stride);
					return y;
				}
				default:
				{
					throw std::runtime_error("Invalid pooling type.");
				}
			}
		}

		math::matrix<float> backward(const math::matrix<float>& delta)
		{
			assert(delta.shape().second == channels * output_size);

			switch(pool_type)
			{
				case pooling::type::max:
				case pooling::type::avg:
				{
					dx = pooling::backward(x, delta, pool_type, channels, input_size, pool_size, stride);
					return dx;
				}
				default:
				{
					throw std::runtime_error("Invalid pooling type.");
				}
			}
		}
	};
};