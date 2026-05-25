#pragma once

#include <algorithm>
#include <cmath>
#include "math/matrix.h"

namespace loss
{
	enum class type {
		mse,
		bce,
		ce
	};

	float mse(const math::matrix<float>& y, const math::matrix<float>& Y)
	{
		auto [rows, cols] = y.shape();
		assert(cols == 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

		math::matrix<float> el(rows, 1);
		const float* y_p = y.elements;
		const float* Y_p = Y.elements;
		float* el_p = el.elements;

		math::q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
			std::size_t i = idx[0];
			float d = y_p[i] - Y_p[i];
			el_p[i] = d * d;
		});

		return math::reduce_sum(el);
	}

	math::matrix<float> mse_derivative(const math::matrix<float>& y, const math::matrix<float>& Y)
	{
		auto [rows, cols] = y.shape();
		assert(cols == 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

		math::matrix<float> grad(rows, cols);
		const float* y_p = y.elements;
		const float* Y_p = Y.elements;
		float* grad_p = grad.elements;
		float inv = 2.0 / static_cast<float>(rows);

		math::q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
			std::size_t i = idx[0];
			grad_p[i] = inv * (y_p[i] - Y_p[i]);
		});

		return grad;
	}

	float ce(const math::matrix<float>& y, const math::matrix<float>& Y)
	{
		auto [rows, cols] = y.shape();
		assert(cols > 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

		math::matrix<float> el(rows, 1);
		const float* y_p = y.elements;
		const float* Y_p = Y.elements;
		float* el_p = el.elements;
		const float epsilon = 1e-7f;
		std::size_t c = cols;

		math::q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
			std::size_t i = idx[0];
			float s = 0.0;
			for(std::size_t j = 0; j < c; j++)
			{
				if(Y_p[i * c + j] != 0.0)
				{
					float p = y_p[i * c + j];
					if(p < epsilon) p = epsilon;
					s = -sycl::log(p);
					break;
				}
			}
			el_p[i] = s;
		});

		return math::reduce_sum(el) / static_cast<float>(rows);
	}

	math::matrix<float> ce_derivative(const math::matrix<float>& y, const math::matrix<float>& Y)
	{
		return y - Y;
	}

	float bce(const math::matrix<float>& y, const math::matrix<float>& Y)
	{
		auto [rows, cols] = y.shape();
		assert(rows == Y.shape().first && cols == Y.shape().second);
		assert(cols == 1);

		math::matrix<float> el(rows, 1);
		const float* y_p = y.elements;
		const float* Y_p = Y.elements;
		float* el_p = el.elements;
		const float epsilon = 1e-7f;

		math::q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
			std::size_t i = idx[0];
			float pred = y_p[i];
			if(pred < epsilon) pred = epsilon;
			else if(pred > 1.0 - epsilon) pred = 1.0 - epsilon;
			float target = Y_p[i];
			el_p[i] = -(target * sycl::log(pred) + (1.0 - target) * sycl::log(1.0 - pred));
		});

		return math::reduce_sum(el) / static_cast<float>(rows);
	}

	math::matrix<float> bce_derivative(const math::matrix<float>& y, const math::matrix<float>& Y)
	{
		auto [rows, cols] = y.shape();
		assert(rows == Y.shape().first && cols == Y.shape().second);
		assert(cols == 1);

		math::matrix<float> grad(rows, cols);
		const float* y_p = y.elements;
		const float* Y_p = Y.elements;
		float* grad_p = grad.elements;
		const float epsilon = 1e-7f;
		float inv_rows = 1.0 / static_cast<float>(rows);

		math::q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
			std::size_t i = idx[0];
			float pred = y_p[i];
			if(pred < epsilon) pred = epsilon;
			else if(pred > 1.0 - epsilon) pred = 1.0 - epsilon;
			float target = Y_p[i];
			grad_p[i] = ((pred - target) / (pred * (1.0 - pred))) * inv_rows;
		});

		return grad;
	}
};
