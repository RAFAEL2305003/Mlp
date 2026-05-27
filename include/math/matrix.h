#pragma once

#include <iostream>
#include <vector>
#include <cassert>
#include <cstring>
#include <utility>
#include <sycl/sycl.hpp>

namespace math
{
	inline sycl::queue q{
		sycl::gpu_selector_v,
		sycl::property::queue::in_order{}
	};

	inline std::vector<void*> deferred_frees;

	inline void drain()
	{
		if(deferred_frees.empty())
		{
			q.wait();
			return;
		}
		q.wait();
		for(void* p : deferred_frees) sycl::free(p, q);
		deferred_frees.clear();
	}

	template<typename T>
	class matrix
	{
		private:
		std::size_t rows;
		std::size_t cols;

		void release()
		{
			if(elements)
			{
				deferred_frees.push_back(elements);
				elements = nullptr;
			}
		}

		public:
		T* elements;

		matrix() : rows(0), cols(0), elements(nullptr) {}

		matrix(std::size_t r, std::size_t c) : rows(r), cols(c), elements(nullptr)
		{
			if(rows * cols > 0)
			{
				elements = sycl::malloc_device<T>(rows * cols, q);
			}
		}

		matrix(std::size_t r, std::size_t c, float value) : rows(r), cols(c), elements(nullptr)
		{
			std::size_t n = rows * cols;
			if(n > 0)
			{
				elements = sycl::malloc_device<T>(n, q);
				T* p = elements;
				T v = static_cast<T>(value);
				q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
					p[idx[0]] = v;
				});
			}
		}

		matrix(std::size_t r, std::size_t c, const std::vector<T>& v) : rows(r), cols(c), elements(nullptr)
		{
			assert(v.size() == rows * cols);
			std::size_t n = rows * cols;
			if(n > 0)
			{
				elements = sycl::malloc_device<T>(n, q);
				q.memcpy(elements, v.data(), n * sizeof(T)).wait();
			}
		}

		~matrix() { release(); }

		matrix(const matrix& other) : rows(other.rows), cols(other.cols), elements(nullptr)
		{
			std::size_t n = rows * cols;
			if(n > 0)
			{
				elements = sycl::malloc_device<T>(n, q);
				if(other.elements)
				{
					q.memcpy(elements, other.elements, n * sizeof(T));
				}
			}
		}

		matrix(matrix&& other) noexcept : rows(other.rows), cols(other.cols), elements(other.elements)
		{
			other.rows = 0;
			other.cols = 0;
			other.elements = nullptr;
		}

		matrix& operator=(const matrix& other)
		{
			if(this != &other)
			{
				std::size_t new_size = other.rows * other.cols;
				std::size_t cur_size = rows * cols;
				if(elements != nullptr && cur_size == new_size)
				{
					rows = other.rows;
					cols = other.cols;
					if(other.elements)
					{
						q.memcpy(elements, other.elements, new_size * sizeof(T));
					}
				}
				else
				{
					release();
					rows = other.rows;
					cols = other.cols;
					if(new_size > 0)
					{
						elements = sycl::malloc_device<T>(new_size, q);
						if(other.elements)
						{
							q.memcpy(elements, other.elements, new_size * sizeof(T));
						}
					}
				}
			}
			return *this;
		}

		matrix& operator=(matrix&& other) noexcept
		{
			if(this != &other)
			{
				release();
				rows = other.rows;
				cols = other.cols;
				elements = other.elements;
				other.rows = 0;
				other.cols = 0;
				other.elements = nullptr;
			}
			return *this;
		}

		std::pair<std::size_t, std::size_t> shape() { return {rows, cols}; }
		const std::pair<std::size_t, std::size_t> shape() const { return {rows, cols}; }

		T& operator()(std::size_t i, std::size_t j) { return elements[i * cols + j]; }
		const T& operator()(std::size_t i, std::size_t j) const { return elements[i * cols + j]; }

		void ensure_shape(std::size_t r, std::size_t c)
		{
			std::size_t need = r * c;
			std::size_t have = rows * cols;
			if(need != have)
			{
				release();
				if(need > 0) elements = sycl::malloc_device<T>(need, q);
			}
			rows = r;
			cols = c;
		}

		matrix transpose() const
		{
			matrix r(cols, rows);
			const T* src = elements;
			T* dst = r.elements;
			std::size_t r_count = rows;
			std::size_t c_count = cols;
			q.parallel_for(sycl::range<2>(c_count, r_count), [=](sycl::id<2> idx) {
				std::size_t j = idx[0];
				std::size_t i = idx[1];
				dst[j * r_count + i] = src[i * c_count + j];
			});
			return r;
		}

		void print()
		{
			std::vector<T> host(rows * cols);
			if(elements && !host.empty())
			{
				q.memcpy(host.data(), elements, rows * cols * sizeof(T)).wait();
			}
			for(std::size_t i = 0; i < rows; i++)
			{
				std::cout << "[";
				for(std::size_t j = 0; j < cols; j++)
				{
					std::cout << host[i * cols + j];
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
			const_cast<matrix*>(this)->print();
		}
	};

	template<typename T>
	matrix<T> operator*(float num, const matrix<T>& a)
	{
		auto [rows, cols] = a.shape();
		matrix<T> r(rows, cols);
		const T* a_ptr = a.elements;
		T* r_ptr = r.elements;
		std::size_t n = rows * cols;
		q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			r_ptr[idx[0]] = a_ptr[idx[0]] * num;
		});
		return r;
	}

	template<typename T>
	matrix<T> operator+(const matrix<T>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		auto [b_rows, b_cols] = b.shape();
		matrix<T> c(a_rows, a_cols);
		const T* a_ptr = a.elements;
		const T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		bool b_row1 = (b_rows == 1);
		bool b_col1 = (b_cols == 1);
		std::size_t ac = a_cols;
		std::size_t bc = b_cols;
		q.parallel_for(sycl::range<2>(a_rows, a_cols), [=](sycl::id<2> idx) {
			std::size_t i = idx[0];
			std::size_t j = idx[1];
			std::size_t bi = b_row1 ? 0 : i;
			std::size_t bj = b_col1 ? 0 : j;
			c_ptr[i * ac + j] = a_ptr[i * ac + j] + b_ptr[bi * bc + bj];
		});
		return c;
	}

	template<typename T>
	matrix<T> operator-(const matrix<T>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		assert(a_rows == b.shape().first && a_cols == b.shape().second);
		matrix<T> c(a_rows, a_cols);
		const T* a_ptr = a.elements;
		const T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		std::size_t n = a_rows * a_cols;
		q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			c_ptr[idx[0]] = a_ptr[idx[0]] - b_ptr[idx[0]];
		});
		return c;
	}

	template<typename T>
	matrix<T> operator*(const matrix<T>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		auto [b_rows, b_cols] = b.shape();
		assert(a_cols == b_rows);
		matrix<T> c(a_rows, b_cols);
		const T* a_ptr = a.elements;
		const T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		std::size_t k_count = a_cols;
		std::size_t bc = b_cols;
		q.parallel_for(sycl::range<2>(a_rows, b_cols), [=](sycl::id<2> idx) {
			std::size_t i = idx[0];
			std::size_t j = idx[1];
			T total = 0;
			for(std::size_t k = 0; k < k_count; k++)
			{
				total += a_ptr[i * k_count + k] * b_ptr[k * bc + j];
			}
			c_ptr[i * bc + j] = total;
		});
		return c;
	}

	template<typename T>
	matrix<T> hadamard(const matrix<T>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		assert(a_rows == b.shape().first && a_cols == b.shape().second);
		matrix<T> c(a_rows, a_cols);
		const T* a_ptr = a.elements;
		const T* b_ptr = b.elements;
		T* c_ptr = c.elements;
		std::size_t n = a_rows * a_cols;
		q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			c_ptr[idx[0]] = a_ptr[idx[0]] * b_ptr[idx[0]];
		});
		return c;
	}

	template<typename T>
	matrix<T> sum(const matrix<T>& a)
	{
		auto [rows, cols] = a.shape();
		matrix<T> r(1, cols);
		const T* a_ptr = a.elements;
		T* r_ptr = r.elements;
		std::size_t r_count = rows;
		std::size_t c_count = cols;

		q.parallel_for(sycl::range<1>(c_count), [=](sycl::id<1> idx) {
			r_ptr[idx[0]] = T(0);
		});

		q.parallel_for(sycl::range<2>(r_count, c_count), [=](sycl::id<2> idx) {
			std::size_t i = idx[0];
			std::size_t j = idx[1];
			sycl::atomic_ref<T,
							 sycl::memory_order::relaxed,
							 sycl::memory_scope::device,
							 sycl::access::address_space::global_space> ref(r_ptr[j]);
			ref.fetch_add(a_ptr[i * c_count + j]);
		});
		return r;
	}

	template<typename T>
	void subtract_scaled_inplace(matrix<T>& target, float scale, const matrix<T>& other)
	{
		auto [tr, tc] = target.shape();
		auto [or_, oc] = other.shape();
		assert(tr == or_ && tc == oc);
		T* t_ptr = target.elements;
		const T* o_ptr = other.elements;
		std::size_t n = tr * tc;
		T s = static_cast<T>(scale);
		q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
			t_ptr[idx[0]] -= s * o_ptr[idx[0]];
		});
	}

	template<typename T>
	T reduce_sum(const matrix<T>& a)
	{
		auto [rows, cols] = a.shape();
		std::size_t n = rows * cols;
		T* dev_total = sycl::malloc_device<T>(1, q);
		q.memset(dev_total, 0, sizeof(T));
		const T* a_ptr = a.elements;
		q.parallel_for(sycl::range<1>(n),
					   sycl::reduction(dev_total, sycl::plus<T>()),
					   [=](sycl::id<1> idx, auto& red) {
			red += a_ptr[idx[0]];
		});
		T result;
		q.memcpy(&result, dev_total, sizeof(T)).wait();
		sycl::free(dev_total, q);
		return result;
	}

	template<typename T>
	math::matrix<T> concat(const math::matrix<T>& a, std::vector<T>& b)
	{
		auto [rows, cols] = a.shape();
		assert(rows == b.size());
		math::matrix<T> result(rows, cols + 1);

		T* col_buf = sycl::malloc_device<T>(rows, q);
		q.memcpy(col_buf, b.data(), rows * sizeof(T)).wait();

		const T* a_ptr = a.elements;
		T* r_ptr = result.elements;
		std::size_t cw = cols;
		std::size_t rw = cols + 1;
		q.parallel_for(sycl::range<2>(rows, rw), [=](sycl::id<2> idx) {
			std::size_t i = idx[0];
			std::size_t j = idx[1];
			r_ptr[i * rw + j] = (j < cw) ? a_ptr[i * cw + j] : col_buf[i];
		});
		q.wait();
		sycl::free(col_buf, q);
		return result;
	}
};
