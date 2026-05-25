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

	math::matrix<float> forward(const math::matrix<float>& x,
								 const math::matrix<float>& w,
								 const math::matrix<float>& b,
								 std::size_t stride,
								 std::size_t padding)
	{
		size_t batch_size = x.shape().first;
		size_t input_size = x.shape().second;
		size_t filters = w.shape().first;
		size_t kernel_size = w.shape().second;
		size_t b_rows = b.shape().first;
		size_t b_cols = b.shape().second;

		assert(b_rows == 1 && b_cols == filters);

		std::size_t out_size = output_size(input_size, kernel_size, stride, padding);
		math::matrix<float> z(batch_size, filters * out_size);

		const float* x_p = x.elements;
		const float* w_p = w.elements;
		const float* b_p = b.elements;
		float* z_p = z.elements;
		std::size_t is = input_size;
		std::size_t ks = kernel_size;
		std::size_t os = out_size;
		std::size_t st = stride;
		std::size_t pd = padding;

		math::q.parallel_for(sycl::range<3>(batch_size, filters, out_size), [=](sycl::id<3> idx) {
			std::size_t i = idx[0];
			std::size_t f = idx[1];
			std::size_t out_pos = idx[2];
			float sum = b_p[f];

			for(std::size_t k = 0; k < ks; k++)
			{
				std::size_t padded_idx = (out_pos * st) + k;
				if(padded_idx >= pd)
				{
					std::size_t input_idx = padded_idx - pd;
					if(input_idx < is)
					{
						sum += x_p[i * is + input_idx] * w_p[f * ks + k];
					}
				}
			}

			z_p[i * (filters * os) + (f * os) + out_pos] = sum;
		});

		return z;
	}

	math::matrix<float> input_derivative(const math::matrix<float>& delta,
										  const math::matrix<float>& w,
										  std::size_t input_size,
										  std::size_t stride,
										  std::size_t padding)
	{
		size_t batch_size = delta.shape().first;
		size_t delta_cols = delta.shape().second;
		size_t filters = w.shape().first;
		size_t kernel_size = w.shape().second;

		assert(delta_cols % filters == 0);

		std::size_t out_size = delta_cols / filters;
		math::matrix<float> dx(batch_size, input_size);

		const float* delta_p = delta.elements;
		const float* w_p = w.elements;
		float* dx_p = dx.elements;
		std::size_t is = input_size;
		std::size_t ks = kernel_size;
		std::size_t os = out_size;
		std::size_t st = stride;
		std::size_t pd = padding;
		std::size_t fs = filters;

		math::q.parallel_for(sycl::range<2>(batch_size, input_size), [=](sycl::id<2> idx) {
			std::size_t i = idx[0];
			std::size_t input_idx = idx[1];
			std::size_t padded_idx = input_idx + pd;
			float sum = 0.0;

			for(std::size_t f = 0; f < fs; f++)
			{
				for(std::size_t out_pos = 0; out_pos < os; out_pos++)
				{
					std::size_t window_start = out_pos * st;
					if(padded_idx >= window_start)
					{
						std::size_t k = padded_idx - window_start;
						if(k < ks)
						{
							sum += delta_p[i * (fs * os) + (f * os) + out_pos] *
								   w_p[f * ks + k];
						}
					}
				}
			}

			dx_p[i * is + input_idx] = sum;
		});

		return dx;
	}

	math::matrix<float> weight_derivative(const math::matrix<float>& x,
										   const math::matrix<float>& delta,
										   std::size_t filters,
										   std::size_t kernel_size,
										   std::size_t stride,
										   std::size_t padding)
	{
		size_t batch_size = x.shape().first;
		size_t input_size = x.shape().second;
		size_t delta_rows = delta.shape().first;
		size_t delta_cols = delta.shape().second;

		assert(batch_size == delta_rows);
		assert(delta_cols % filters == 0);

		std::size_t out_size = delta_cols / filters;
		math::matrix<float> dw(filters, kernel_size);

		const float* x_p = x.elements;
		const float* delta_p = delta.elements;
		float* dw_p = dw.elements;
		std::size_t bs = batch_size;
		std::size_t is = input_size;
		std::size_t ks = kernel_size;
		std::size_t os = out_size;
		std::size_t st = stride;
		std::size_t pd = padding;
		std::size_t fs = filters;

		math::q.parallel_for(sycl::range<2>(filters, kernel_size), [=](sycl::id<2> idx) {
			std::size_t f = idx[0];
			std::size_t k = idx[1];
			float sum = 0.0;

			for(std::size_t i = 0; i < bs; i++)
			{
				for(std::size_t out_pos = 0; out_pos < os; out_pos++)
				{
					std::size_t padded_idx = (out_pos * st) + k;
					if(padded_idx >= pd)
					{
						std::size_t input_idx = padded_idx - pd;
						if(input_idx < is)
						{
							sum += x_p[i * is + input_idx] *
								   delta_p[i * (fs * os) + (f * os) + out_pos];
						}
					}
				}
			}

			dw_p[f * ks + k] = sum;
		});

		return dw;
	}

	math::matrix<float> bias_derivative(const math::matrix<float>& delta,
										 std::size_t filters)
	{
		size_t batch_size = delta.shape().first;
		size_t delta_cols = delta.shape().second;

		assert(delta_cols % filters == 0);

		std::size_t out_size = delta_cols / filters;
		math::matrix<float> db(1, filters);

		const float* delta_p = delta.elements;
		float* db_p = db.elements;
		std::size_t bs = batch_size;
		std::size_t os = out_size;
		std::size_t fs = filters;

		math::q.parallel_for(sycl::range<1>(filters), [=](sycl::id<1> idx) {
			db_p[idx[0]] = 0.0f;
		});

		// Parallel reduction: every (batch, filter, out_pos) tuple atomically
		// adds into db[f]. Was previously `filters` threads (often 4) — now
		// it's batch_size*filters*os, which actually uses the GPU.
		math::q.parallel_for(sycl::range<3>(bs, fs, os), [=](sycl::id<3> idx) {
			std::size_t i = idx[0];
			std::size_t f = idx[1];
			std::size_t out_pos = idx[2];
			sycl::atomic_ref<float,
							 sycl::memory_order::relaxed,
							 sycl::memory_scope::device,
							 sycl::access::address_space::global_space> ref(db_p[f]);
			ref.fetch_add(delta_p[i * (fs * os) + (f * os) + out_pos]);
		});

		return db;
	}
};
