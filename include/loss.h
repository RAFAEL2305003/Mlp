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

	double mse(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();

		assert(cols == 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

		double loss = 0.0;
		for(size_t i = 0; i < rows; i++)
		{
			loss += std::pow((y(i, 0) - Y(i, 0)), 2);
		}

		return loss;
	}

	math::matrix<double> mse_derivative(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();

		assert(cols == 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

		math::matrix<double> loss(rows, cols);
		std::cout << "Batch size = " << rows << "\n";
		for(size_t i = 0; i < rows; i++)
		{
			loss(i, 0) = (2.0 / rows) * (y(i, 0) - Y(i, 0));
		}

		return loss;
	}

	double ce(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		auto [rows, cols] = y.shape();

		assert(cols > 1);
		assert(rows == Y.shape().first && cols == Y.shape().second);

		const double epsilon = 1e-15;
		double loss = 0.0;
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				if(Y(i, j) != 0)
				{
					double p = std::max(y(i, j), epsilon);
					loss += -std::log(p);
					break;
				}
			}
		}

		return loss / rows;
	}

	math::matrix<double> ce_derivative(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		return y - Y;
	}

	double bce(const math::matrix<double>& y, const math::matrix<double>& Y)
	{
		size_t rows = y.shape().first;
		size_t cols = y.shape().second;

		assert(rows == Y.shape().first && cols == Y.shape().second);
		assert(cols == 1);

		const double epsilon = 1e-15;
		double loss = 0.0;

		{
			sycl::buffer<const double, 2> y_buf(y.elements.data(), sycl::range<2>(rows, cols));
			sycl::buffer<const double, 2> Y_buf(Y.elements.data(), sycl::range<2>(rows, cols));
			sycl::buffer<double, 1> loss_buf(&loss, sycl::range<1>(1));

			math::q.submit([&](sycl::handler& h) {
				auto y_acc = y_buf.template get_access<sycl::access::mode::read>(h);
				auto Y_acc = Y_buf.template get_access<sycl::access::mode::read>(h);
				auto loss_reduction = sycl::reduction(loss_buf, h, sycl::plus<double>());

				h.parallel_for(sycl::range<1>(rows), loss_reduction, [=](sycl::id<1> idx, auto& sum) {
					size_t i = idx[0];
					double pred = y_acc[sycl::id<2>(i, 0)];
					pred = std::min(std::max(pred, epsilon), 1.0 - epsilon);
					double target = Y_acc[sycl::id<2>(i, 0)];
					sum += -(target * sycl::log(pred) + (1.0 - target) * sycl::log(1.0 - pred));
				});
			}).wait();
		}
		return loss / rows;
	}

	math::matrix<double> bce_derivative(const math::matrix<double>& y,
										 const math::matrix<double>& Y)
	{
		size_t rows = y.shape().first;
		size_t cols = y.shape().second;

		assert(rows == Y.shape().first && cols == Y.shape().second);
		assert(cols == 1);

		math::matrix<double> grad(rows, cols);
		const double epsilon = 1e-15;

		{
			sycl::buffer<const double, 2> y_buf(y.elements.data(), sycl::range<2>(rows, cols));
			sycl::buffer<const double, 2> Y_buf(Y.elements.data(), sycl::range<2>(rows, cols));
			sycl::buffer<double, 2> grad_buf(grad.elements.data(), sycl::range<2>(rows, cols));

			math::q.submit([&](sycl::handler& h) {
				auto y_acc = y_buf.template get_access<sycl::access::mode::read>(h);
				auto Y_acc = Y_buf.template get_access<sycl::access::mode::read>(h);
				auto grad_acc = grad_buf.template get_access<sycl::access::mode::write>(h);

				h.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
					size_t i = idx[0];
					double pred = y_acc[sycl::id<2>(i, 0)];
					pred = std::min(std::max(pred, epsilon), 1.0 - epsilon);
					double target = Y_acc[sycl::id<2>(i, 0)];
					grad_acc[sycl::id<2>(i, 0)] = ((pred - target) / (pred * (1.0 - pred))) / rows;
				});
			}).wait();
		}
		return grad;
	}
};
