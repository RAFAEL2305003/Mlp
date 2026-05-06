#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <sycl/sycl.hpp>

namespace math
{
	inline sycl::queue q(sycl::gpu_selector_v);

	template<typename T>
	class matrix
	{
		private:
		std::size_t rows;
		std::size_t cols;

		public:
		std::vector<T> elements;

		matrix() : rows(0), cols(0) {}

		matrix(std::size_t r, std::size_t c) : rows(r), cols(c)
		{
			elements.resize(rows * cols);
		}

		matrix(std::size_t r, std::size_t c, double value) : rows(r), cols(c)
		{
			elements.resize(rows * cols);
			{
				sycl::buffer<T, 2> elements_buf(elements.data(), sycl::range<2>(rows, cols));
				q.submit([&](sycl::handler& h) {
					auto elements_acc = elements_buf.template get_access<sycl::access::mode::write>(h);
					h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx) {
						elements_acc[idx] = static_cast<T>(value);
					});
				}).wait();
			}
		}

		matrix(std::size_t r, std::size_t c, const std::vector<T>& v) : rows(r), cols(c)
		{
			assert(v.size() == rows * cols);
			elements.resize(rows * cols);
			elements = v;
		}

		matrix(const matrix& other) : rows(other.shape().first), cols(other.shape().second)
		{
			elements = other.elements;
		}

		matrix& operator=(const matrix& other)
		{
			if(this != &other)
			{
				rows = other.shape().first;
				cols = other.shape().second;
				elements = other.elements;
			}

			return *this;
		}

		std::pair<std::size_t, std::size_t> shape() { return std::pair<std::size_t, std::size_t>(rows, cols); }
		const std::pair<std::size_t, std::size_t> shape() const { return std::pair<std::size_t, std::size_t>(rows, cols); }

		T& operator()(std::size_t i, std::size_t j) { return elements[i * cols + j]; }
		const T& operator()(std::size_t i, std::size_t j) const { return elements[i * cols + j]; }

		matrix transpose()
		{
			matrix r(cols, rows);
			size_t r_count = rows;
			size_t c_count = cols;
			{
				sycl::buffer<T, 2> src_buf(elements.data(), sycl::range<2>(r_count, c_count));
				sycl::buffer<T, 2> dst_buf(r.elements.data(), sycl::range<2>(c_count, r_count));
				q.submit([&](sycl::handler& h) {
					auto src = src_buf.template get_access<sycl::access::mode::read>(h);
					auto dst = dst_buf.template get_access<sycl::access::mode::write>(h);
					h.parallel_for(sycl::range<2>(c_count, r_count), [=](sycl::id<2> idx) {
						size_t j = idx[0];
						size_t i = idx[1];
						dst[idx] = src[sycl::id<2>(i, j)];
					});
				}).wait();
			}
			return r;
		}

		void print()
		{
			for(std::size_t i = 0; i < rows; i++)
			{
				std::cout << "[";
				for(std::size_t j = 0; j < cols; j++)
				{
					std::cout << elements[i * cols + j];
					if(j != cols - 1)
					{
						std::cout << " ";
					}
				}
				std::cout << "]\n";
			}
			std::cout << "\n";
		}

		void print() const
		{
			for(std::size_t i = 0; i < rows; i++)
			{
				std::cout << "[";
				for(std::size_t j = 0; j < cols; j++)
				{
					std::cout << elements[i * cols + j];
					if(j != cols - 1)
					{
						std::cout << " ";
					}
				}
				std::cout << "]\n";
			}
			std::cout << "\n";
		}

	};

	template<typename T>
	matrix<T> operator*(double num, matrix<T>& a)
	{
		size_t rows = a.shape().first;
		size_t cols = a.shape().second;
		matrix<T> result(rows, cols);
		{
			sycl::buffer<T, 2> a_buf(a.elements.data(), sycl::range<2>(rows, cols));
			sycl::buffer<T, 2> r_buf(result.elements.data(), sycl::range<2>(rows, cols));
			q.submit([&](sycl::handler& h) {
				auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
				auto r_acc = r_buf.template get_access<sycl::access::mode::write>(h);
				h.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> idx) {
					r_acc[idx] = a_acc[idx] * num;
				});
			}).wait();
		}
		return result;
	}

	template<typename T, typename U>
	matrix<T> operator+(const matrix<T>& a, const matrix<U>& b)
	{
		size_t a_rows = a.shape().first;
		size_t a_cols = a.shape().second;
		size_t b_rows = b.shape().first;
		size_t b_cols = b.shape().second;
		matrix<T> c(a_rows, a_cols);
		{
			sycl::buffer<const T, 2> a_buf(a.elements.data(), sycl::range<2>(a_rows, a_cols));
			sycl::buffer<const U, 2> b_buf(b.elements.data(), sycl::range<2>(b_rows, b_cols));
			sycl::buffer<T, 2> c_buf(c.elements.data(), sycl::range<2>(a_rows, a_cols));
			q.submit([&](sycl::handler& h) {
				auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
				auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
				auto c_acc = c_buf.template get_access<sycl::access::mode::write>(h);
				h.parallel_for(sycl::range<2>(a_rows, a_cols), [=](sycl::id<2> idx) {
					size_t i = idx[0];
					size_t j = idx[1];
					size_t bi = (b_rows == 1) ? 0 : i;
					size_t bj = (b_cols == 1) ? 0 : j;
					c_acc[idx] = a_acc[idx] + b_acc[sycl::id<2>(bi, bj)];
				});
			}).wait();
		}
		return c;
	}

	template<typename T>
	matrix<T> operator-(const matrix<T>& a, const matrix<T>& b)
	{
		size_t a_rows = a.shape().first;
		size_t a_cols = a.shape().second;
		size_t b_rows = b.shape().first;
		size_t b_cols = b.shape().second;
		assert(a_rows == b_rows && a_cols == b_cols);
		matrix<T> c(a_rows, a_cols);
		{
			sycl::buffer<const T, 2> a_buf(a.elements.data(), sycl::range<2>(a_rows, a_cols));
			sycl::buffer<const T, 2> b_buf(b.elements.data(), sycl::range<2>(b_rows, b_cols));
			sycl::buffer<T, 2> c_buf(c.elements.data(), sycl::range<2>(a_rows, a_cols));
			q.submit([&](sycl::handler& h) {
				auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
				auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
				auto c_acc = c_buf.template get_access<sycl::access::mode::write>(h);
				h.parallel_for(sycl::range<2>(a_rows, a_cols), [=](sycl::id<2> idx) {
					c_acc[idx] = a_acc[idx] - b_acc[idx];
				});
			}).wait();
		}
		return c;
	}

	template<typename T, typename U>
	matrix<T> operator*(const matrix<U>& a, const matrix<T>& b)
	{
		size_t a_rows = a.shape().first;
		size_t a_cols = a.shape().second;
		size_t b_rows = b.shape().first;
		size_t b_cols = b.shape().second;
		assert(a_cols == b_rows);
		matrix<T> c(a_rows, b_cols);
		{
			sycl::buffer<const U, 2> a_buf(a.elements.data(), sycl::range<2>(a_rows, a_cols));
			sycl::buffer<const T, 2> b_buf(b.elements.data(), sycl::range<2>(b_rows, b_cols));
			sycl::buffer<T, 2> c_buf(c.elements.data(), sycl::range<2>(a_rows, b_cols));
			q.submit([&](sycl::handler& h) {
				auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
				auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
				auto c_acc = c_buf.template get_access<sycl::access::mode::write>(h);
				h.parallel_for(sycl::range<2>(a_rows, b_cols), [=](sycl::id<2> idx) {
					size_t i = idx[0];
					size_t j = idx[1];
					T total = 0;
					for(size_t k = 0; k < a_cols; k++)
					{
						total += a_acc[sycl::id<2>(i, k)] * b_acc[sycl::id<2>(k, j)];
					}
					c_acc[idx] = total;
				});
			}).wait();
		}
		return c;
	}

	template<typename T, typename U>
	matrix<T> hadamard(const matrix<U>& a, const matrix<T>& b)
	{
		size_t a_rows = a.shape().first;
		size_t a_cols = a.shape().second;
		size_t b_rows = b.shape().first;
		size_t b_cols = b.shape().second;
		assert(a_rows == b_rows && a_cols == b_cols);
		matrix<T> c(a_rows, a_cols);
		{
			sycl::buffer<const U, 2> a_buf(a.elements.data(), sycl::range<2>(a_rows, a_cols));
			sycl::buffer<const T, 2> b_buf(b.elements.data(), sycl::range<2>(b_rows, b_cols));
			sycl::buffer<T, 2> c_buf(c.elements.data(), sycl::range<2>(a_rows, a_cols));
			q.submit([&](sycl::handler& h) {
				auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
				auto b_acc = b_buf.template get_access<sycl::access::mode::read>(h);
				auto c_acc = c_buf.template get_access<sycl::access::mode::write>(h);
				h.parallel_for(sycl::range<2>(a_rows, a_cols), [=](sycl::id<2> idx) {
					c_acc[idx] = a_acc[idx] * b_acc[idx];
				});
			}).wait();
		}
		return c;
	}

	template<typename T>
	matrix<T> sum(const matrix<T>& a)
	{
		size_t rows = a.shape().first;
		size_t cols = a.shape().second;
		matrix<T> r(1, cols);
		{
			sycl::buffer<const T, 2> a_buf(a.elements.data(), sycl::range<2>(rows, cols));
			sycl::buffer<T, 2> r_buf(r.elements.data(), sycl::range<2>(1, cols));
			q.submit([&](sycl::handler& h) {
				auto a_acc = a_buf.template get_access<sycl::access::mode::read>(h);
				auto r_acc = r_buf.template get_access<sycl::access::mode::write>(h);
				h.parallel_for(sycl::range<2>(1, cols), [=](sycl::id<2> idx) {
					size_t j = idx[1];
					T total = 0;
					for(size_t i = 0; i < rows; i++)
					{
						total += a_acc[sycl::id<2>(i, j)];
					}
					r_acc[idx] = total;
				});
			}).wait();
		}
		return r;
	}

	template<typename T>
	math::matrix<T> concat(const math::matrix<T>& a, std::vector<T>& b)
	{
		auto [rows, cols] = a.shape();
		assert(rows == b.size());
		math::matrix<T> result(rows, cols + 1);
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				result(i, j) = a(i, j);	
			}
			result(i, cols) = b[i];
		}
		return result;
	}
};
