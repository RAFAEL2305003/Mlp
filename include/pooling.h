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
		size_t batch_size = x.shape().first;
		size_t cols = x.shape().second;

		assert(cols == channels * input_size);

		std::size_t out_size = output_size(input_size, pool_size, stride);
		math::matrix<double> y(batch_size, channels * out_size);

		{
			sycl::buffer<const double, 2> x_buf(x.elements.data(), sycl::range<2>(batch_size, cols));
			sycl::buffer<double, 2> y_buf(y.elements.data(), sycl::range<2>(batch_size, channels * out_size));

			math::q.submit([&](sycl::handler& h) {
				auto x_acc = x_buf.template get_access<sycl::access::mode::read>(h);
				auto y_acc = y_buf.template get_access<sycl::access::mode::write>(h);

				h.parallel_for(sycl::range<3>(batch_size, channels, out_size), [=](sycl::id<3> idx) {
					std::size_t i = idx[0];
					std::size_t c = idx[1];
					std::size_t out_pos = idx[2];
					std::size_t window_start = out_pos * stride;
					double max_value = std::numeric_limits<double>::lowest();

					for(std::size_t k = 0; k < pool_size; k++)
					{
						double value = x_acc[sycl::id<2>(i, (c * input_size) + window_start + k)];
						if(value > max_value)
						{
							max_value = value;
						}
					}

					y_acc[sycl::id<2>(i, (c * out_size) + out_pos)] = max_value;
				});
			}).wait();
		}

		return y;
	}

	math::matrix<double> avg_forward(const math::matrix<double>& x,
									 std::size_t channels,
									 std::size_t input_size,
									 std::size_t pool_size,
									 std::size_t stride)
	{
		size_t batch_size = x.shape().first;
		size_t cols = x.shape().second;

		assert(cols == channels * input_size);

		std::size_t out_size = output_size(input_size, pool_size, stride);
		math::matrix<double> y(batch_size, channels * out_size);

		{
			sycl::buffer<const double, 2> x_buf(x.elements.data(), sycl::range<2>(batch_size, cols));
			sycl::buffer<double, 2> y_buf(y.elements.data(), sycl::range<2>(batch_size, channels * out_size));

			math::q.submit([&](sycl::handler& h) {
				auto x_acc = x_buf.template get_access<sycl::access::mode::read>(h);
				auto y_acc = y_buf.template get_access<sycl::access::mode::write>(h);

				h.parallel_for(sycl::range<3>(batch_size, channels, out_size), [=](sycl::id<3> idx) {
					std::size_t i = idx[0];
					std::size_t c = idx[1];
					std::size_t out_pos = idx[2];
					std::size_t window_start = out_pos * stride;
					double sum = 0.0;

					for(std::size_t k = 0; k < pool_size; k++)
					{
						sum += x_acc[sycl::id<2>(i, (c * input_size) + window_start + k)];
					}

					y_acc[sycl::id<2>(i, (c * out_size) + out_pos)] = sum / pool_size;
				});
			}).wait();
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
		size_t batch_size = x.shape().first;
		size_t cols = x.shape().second;
		size_t delta_rows = delta.shape().first;
		size_t delta_cols = delta.shape().second;

		std::size_t out_size = output_size(input_size, pool_size, stride);

		assert(cols == channels * input_size);
		assert(delta_rows == batch_size);
		assert(delta_cols == channels * out_size);

		math::matrix<double> dx(batch_size, cols);

		{
			sycl::buffer<const double, 2> x_buf(x.elements.data(), sycl::range<2>(batch_size, cols));
			sycl::buffer<const double, 2> delta_buf(delta.elements.data(), sycl::range<2>(delta_rows, delta_cols));
			sycl::buffer<double, 2> dx_buf(dx.elements.data(), sycl::range<2>(batch_size, cols));

			math::q.submit([&](sycl::handler& h) {
				auto x_acc = x_buf.template get_access<sycl::access::mode::read>(h);
				auto delta_acc = delta_buf.template get_access<sycl::access::mode::read>(h);
				auto dx_acc = dx_buf.template get_access<sycl::access::mode::write>(h);

				h.parallel_for(sycl::range<3>(batch_size, channels, input_size), [=](sycl::id<3> idx) {
					std::size_t i = idx[0];
					std::size_t c = idx[1];
					std::size_t input_idx = idx[2];
					double sum = 0.0;

					for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
					{
						std::size_t window_start = out_pos * stride;
						if(input_idx >= window_start && input_idx < window_start + pool_size)
						{
							std::size_t max_idx = window_start;
							double max_value = x_acc[sycl::id<2>(i, (c * input_size) + max_idx)];

							for(std::size_t k = 1; k < pool_size; k++)
							{
								std::size_t candidate_idx = window_start + k;
								double value = x_acc[sycl::id<2>(i, (c * input_size) + candidate_idx)];
								if(value > max_value)
								{
									max_value = value;
									max_idx = candidate_idx;
								}
							}

							if(input_idx == max_idx)
							{
								sum += delta_acc[sycl::id<2>(i, (c * out_size) + out_pos)];
							}
						}
					}

					dx_acc[sycl::id<2>(i, (c * input_size) + input_idx)] = sum;
				});
			}).wait();
		}

		return dx;
	}

	math::matrix<double> avg_backward(const math::matrix<double>& delta,
									  std::size_t channels,
									  std::size_t input_size,
									  std::size_t pool_size,
									  std::size_t stride)
	{
		size_t batch_size = delta.shape().first;
		size_t delta_cols = delta.shape().second;

		std::size_t out_size = output_size(input_size, pool_size, stride);

		assert(delta_cols == channels * out_size);

		math::matrix<double> dx(batch_size, channels * input_size);

		{
			sycl::buffer<const double, 2> delta_buf(delta.elements.data(), sycl::range<2>(batch_size, delta_cols));
			sycl::buffer<double, 2> dx_buf(dx.elements.data(), sycl::range<2>(batch_size, channels * input_size));

			math::q.submit([&](sycl::handler& h) {
				auto delta_acc = delta_buf.template get_access<sycl::access::mode::read>(h);
				auto dx_acc = dx_buf.template get_access<sycl::access::mode::write>(h);

				h.parallel_for(sycl::range<3>(batch_size, channels, input_size), [=](sycl::id<3> idx) {
					std::size_t i = idx[0];
					std::size_t c = idx[1];
					std::size_t input_idx = idx[2];
					double sum = 0.0;

					for(std::size_t out_pos = 0; out_pos < out_size; out_pos++)
					{
						std::size_t window_start = out_pos * stride;
						if(input_idx >= window_start && input_idx < window_start + pool_size)
						{
							sum += delta_acc[sycl::id<2>(i, (c * out_size) + out_pos)] / pool_size;
						}
					}

					dx_acc[sycl::id<2>(i, (c * input_size) + input_idx)] = sum;
				});
			}).wait();
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
