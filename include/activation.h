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

	math::matrix<float> relu(const math::matrix<float>& a)
	{
		auto [rows, cols] = a.shape();
		math::matrix<float> r(rows, cols);
		const float* a_ptr = a.elements;
		float* r_ptr = r.elements;
		std::size_t n = rows * cols;
		math::q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			float v = a_ptr[idx[0]];
			r_ptr[idx[0]] = v > 0.0 ? v : 0.0;
		});
		return r;
	}

	math::matrix<float> relu_derivative(const math::matrix<float>& a)
	{
		auto [rows, cols] = a.shape();
		math::matrix<float> r(rows, cols);
		const float* a_ptr = a.elements;
		float* r_ptr = r.elements;
		std::size_t n = rows * cols;
		math::q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			r_ptr[idx[0]] = (a_ptr[idx[0]] > 0.0) ? 1.0 : 0.0;
		});
		return r;
	}

	math::matrix<float> sigmoid(const math::matrix<float>& z)
	{
		auto [rows, cols] = z.shape();
		math::matrix<float> r(rows, cols);
		const float* z_ptr = z.elements;
		float* r_ptr = r.elements;
		std::size_t n = rows * cols;
		math::q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			r_ptr[idx[0]] = 1.0 / (1.0 + sycl::exp(-z_ptr[idx[0]]));
		});
		return r;
	}

	math::matrix<float> sigmoid_derivative(const math::matrix<float>& a)
	{
		auto [rows, cols] = a.shape();
		math::matrix<float> r(rows, cols);
		const float* a_ptr = a.elements;
		float* r_ptr = r.elements;
		std::size_t n = rows * cols;
		math::q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			float v = a_ptr[idx[0]];
			r_ptr[idx[0]] = v * (1.0 - v);
		});
		return r;
	}

	math::matrix<float> softmax(const math::matrix<float>& z)
	{
		auto [rows, cols] = z.shape();
		math::matrix<float> r(rows, cols);
		const float* z_ptr = z.elements;
		float* r_ptr = r.elements;
		std::size_t c = cols;
		math::q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
			std::size_t i = idx[0];
			float max_val = z_ptr[i * c];
			for(std::size_t k = 1; k < c; k++)
			{
				float v = z_ptr[i * c + k];
				if(v > max_val) max_val = v;
			}

			float sum = 0.0;
			for(std::size_t k = 0; k < c; k++)
			{
				sum += sycl::exp(z_ptr[i * c + k] - max_val);
			}

			for(std::size_t j = 0; j < c; j++)
			{
				r_ptr[i * c + j] = sycl::exp(z_ptr[i * c + j] - max_val) / sum;
			}
		});
		return r;
	}

	std::vector<size_t> argmax(const math::matrix<float>& z)
	{
		auto [rows, cols] = z.shape();
		std::vector<float> host(rows * cols);
		math::q.memcpy(host.data(), z.elements, rows * cols * sizeof(float)).wait();

		std::vector<size_t> indices(rows);
		for(size_t i = 0; i < rows; i++)
		{
			float max_val = host[i * cols];
			size_t max_idx = 0;
			for(size_t j = 1; j < cols; j++)
			{
				if(max_val < host[i * cols + j])
				{
					max_val = host[i * cols + j];
					max_idx = j;
				}
			}
			indices[i] = max_idx;
		}

		return indices;
	}
};
