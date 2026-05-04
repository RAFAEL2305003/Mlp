#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <sycl/sycl.hpp>

namespace math
{
	sycl::queue q(sycl::gpu_selector_v);

	template<typename T>
	class matrix
	{
		private:
		std::size_t rows;
		std::size_t cols;

		public:
		T* elements;

		matrix() : rows(0), cols(0), elements(nullptr) {}

		matrix(std::size_t r, std::size_t c) : rows(r), cols(c)
		{
			elements = sycl::malloc_shared<T>(rows * cols, q);
		}

		matrix(std::size_t r, std::size_t c, double value) : rows(r), cols(c)
		{
			elements = sycl::malloc_shared<T>(rows * cols, q);
			q.fill(elements, static_cast<T>(value), rows * cols).wait();
		}

		matrix(std::size_t r, std::size_t c, const std::vector<T>& v) : rows(r), cols(c)
		{
			elements = sycl::malloc_shared<T>(rows * cols, q);
			q.memcpy(elements, v.data(), sizeof(T) * rows * cols).wait();
		}

		~matrix() { if(elements) sycl::free(elements, q); }

		matrix(const matrix& other) : rows(other.shape().first), cols(other.shape().second)
		{
			elements = sycl::malloc_shared<T>(rows * cols, q);
			q.memcpy(elements, other.elements, sizeof(T) * rows * cols).wait();
		}

		matrix& operator=(const matrix& other)
		{
			if(this != &other)
			{
				if(elements) { sycl::free(elements, q); }
				rows = other.shape().first;
				cols = other.shape().second;
				elements = sycl::malloc_shared<T>(rows * cols, q);
				q.memcpy(elements, other.elements, sizeof(T) * rows * cols).wait();
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
			T* src = elements;
			T* dst = r.elements;
			size_t r_count = rows;
			size_t c_count = cols;
			q.parallel_for(sycl::range<2>(c_count, r_count), [=](sycl::id<2> idx) {
				size_t j = idx[0];
				size_t i = idx[1];	
				dst[j * r_count + i] = src[i * c_count + j];
			}).wait();
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
		T* a_ptr = a.elements;
		T* r_ptr = result.elements;
		q.parallel_for(sycl::range<1>(rows * cols), [=](sycl::id<1> i) {
			r_ptr[i] = a_ptr[i] * num;
		}).wait();
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
		T* a_ptr = a.elements;
		T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		q.parallel_for(sycl::range<2>(a_rows, a_cols), [=](sycl::id<2> idx) {
			size_t i = idx[0];
			size_t j = idx[1];
			size_t bi = (b_rows == 1) ? 0 : i;
			size_t bj = (b_cols == 1) ? 0 : j;
			c_ptr[i * a_cols + j] = a_ptr[i * a_cols + j] + b_ptr[bi * b_cols + bj];
		}).wait();
		return c;
	}

	template<typename T>
	matrix<T> operator-(const matrix<T>& a, const matrix<T>& b)
	{
		size_t a_rows = a.shape().first;
		size_t a_cols = a.shape().second;
		size_t b_rows = b.shape().first;
		size_t b_cols = b.shape().second;
		matrix<T> c(a_rows, a_cols);
		T* a_ptr = a.elements;
		T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		q.parallel_for(sycl::range<2>(a_rows, a_cols), [=](sycl::id<2> idx) {
			size_t i = idx[0];
			size_t j = idx[1];
			c_ptr[i * a_cols + j] = a_ptr[i * a_cols + j] - b_ptr[i * a_cols + j];
		}).wait();
		return c;
	}

	template<typename T, typename U>
	matrix<T> operator*(const matrix<U>& a, const matrix<T>& b)
	{
		size_t a_rows = a.shape().first;
		size_t a_cols = a.shape().second;
		size_t b_cols = b.shape().second;
		matrix<T> c(a_rows, b_cols);
		T* a_ptr = a.elements;
		T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		q.parallel_for(sycl::range<2>(a_rows, b_cols), [=](sycl::id<2> idx) {
			size_t i = idx[0];
			size_t j = idx[1];
			T sum = 0;
			for(size_t k = 0; k < a_cols; k++) {
				sum += a_ptr[i * a_cols + k] * b_ptr[k * b_cols + j];
			}
			c_ptr[i * b_cols + j] = sum;
		}).wait();
		return c;
	}

	template<typename T, typename U>
	matrix<T> hadamard(const matrix<U>& a, const matrix<T>& b)
	{
		size_t a_rows = a.shape().first;
		size_t a_cols = a.shape().second;
		size_t b_rows = b.shape().first;
		size_t b_cols = b.shape().second;
		matrix<T> c(a_rows, a_cols);
		T* a_ptr = a.elements;
		T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		q.parallel_for(sycl::range<2>(a_rows, a_cols), [=](sycl::id<2> idx) {
			size_t i = idx[0];
			size_t j = idx[1];
			c_ptr[i * a_cols + j] = a_ptr[i * a_cols + j] * b_ptr[i * b_cols + j];
		}).wait();
		return c;
	}

	template<typename T>
	matrix<T> sum(const matrix<T>& a)
	{
		size_t rows = a.shape().first;
		size_t cols = a.shape().second;
		matrix<T> r(1, cols);
		T* a_ptr = a.elements;
		T* r_ptr = r.elements;
		q.parallel_for(sycl::range<1>(cols), [=](sycl::id<1> idx) {
			size_t j = idx;
			T total = 0;
			for(size_t i = 0; i < rows; i++) {
				total += a_ptr[i * cols + j];
			}
			r_ptr[j] = total;
		}).wait();
		return r;
	}

	template<typename T>
	math::matrix<T> concat(const math::matrix<T>& a, std::vector<T>& b)
	{
		auto [rows, cols] = a.shape();
		assert(rows == b.size());
		math::matrix<T> result(rows, cols + 1);
		T* a_ptr = a.elements;
		T* res_ptr = result.elements;
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

