#pragma once

#include <stdexcept>
#include "math/matrix.h"
#include "math/random.h"
#include "conv1d.h"

namespace nn
{
	class conv1d_layer
	{
		private:
		conv1d::type conv_type;
		math::matrix<float> x;
		math::matrix<float> w;
		math::matrix<float> b;
		math::matrix<float> z;

		math::matrix<float> dw;
		math::matrix<float> dx;
		math::matrix<float> db;

		std::size_t current_padding()
		{
			return conv1d::padding(conv_type, kernel_size);
		}

		public:
		std::size_t input_size;
		std::size_t filters;
		std::size_t kernel_size;
		std::size_t stride;
		std::size_t padding;
		std::size_t output_size;

		conv1d_layer()
			: conv_type(conv1d::type::valid),
			  input_size(0),
			  filters(0),
			  kernel_size(0),
			  stride(1),
			  padding(0),
			  output_size(0) {}

		conv1d_layer(std::size_t input_size,
					 std::size_t filters,
					 std::size_t kernel_size,
					 math::random<float>& rng,
					 conv1d::type conv_type = conv1d::type::valid,
					 std::size_t stride = 1)
			: conv_type(conv_type),
			  w(filters, kernel_size),
			  b(1, filters, 0.0),
			  input_size(input_size),
			  filters(filters),
			  kernel_size(kernel_size),
			  stride(stride)
		{
			assert(input_size > 0);
			assert(filters > 0);
			assert(kernel_size > 0);
			assert(stride > 0);

			padding = current_padding();
			output_size = conv1d::output_size(input_size, kernel_size, stride, padding);

			math::fill_random(w, rng, -1.0f, 1.0f);
		}

		math::matrix<float> forward(const math::matrix<float>& x)
		{
			assert(x.shape().second == input_size);

			this->x = x;

			switch(conv_type)
			{
				case conv1d::type::valid:
				case conv1d::type::same:
				case conv1d::type::full:
				{
					z = conv1d::forward(x, w, b, stride, padding);
					return z;
				}
				default:
				{
					throw std::runtime_error("Invalid conv1d type.");
				}
			}
		}

		math::matrix<float> backward(const math::matrix<float>& delta)
		{
			assert(delta.shape().second == filters * output_size);

			switch(conv_type)
			{
				case conv1d::type::valid:
				case conv1d::type::same:
				case conv1d::type::full:
				{
					dw = conv1d::weight_derivative(x, delta, filters, kernel_size, stride, padding);
					db = conv1d::bias_derivative(delta, filters);
					dx = conv1d::input_derivative(delta, w, input_size, stride, padding);
					return dx;
				}
				default:
				{
					throw std::runtime_error("Invalid conv1d type.");
				}
			}
		}

		void update(float lr)
		{
			math::subtract_scaled_inplace(w, lr, dw);
			math::subtract_scaled_inplace(b, lr, db);
		}
	};
};
