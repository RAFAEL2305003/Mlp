#pragma once

#include <iostream>
#include <vector>
#include <cassert>

namespace math
{
	template<typename T>
	class matrix
	{
		private:
		std::size_t rows;
		std::size_t cols;
		std::vector<T> elements;

		public:
		matrix() : rows(0), cols(0)
		{
		}

		matrix(std::size_t r, std::size_t c) : rows(r), cols(c)
		{
			assert(rows >= 0 && cols >= 0);
			elements.resize(rows * cols);
		}

		matrix(std::size_t r, std::size_t c, double value) : rows(r), cols(c)
		{
			assert(rows >= 0 && cols >= 0);
			elements.resize(rows * cols, value);
		}

		matrix(std::size_t r, std::size_t c, std::vector<T> v) : rows(r), cols(c)
		{
			assert(rows >= 0 && cols >= 0);
			elements.resize(rows * cols);
			for(std::size_t i = 0; i < rows * cols; i++)
			{
				elements[i] = v[i];
			}
		}

		std::pair<std::size_t, std::size_t> shape()
		{
			return std::pair<std::size_t, std::size_t>(rows, cols);
		}

		const std::pair<std::size_t, std::size_t> shape() const
		{
			return std::pair<std::size_t, std::size_t>(rows, cols);
		}

		T& operator()(std::size_t i, std::size_t j)
		{
			assert(i < rows && j < cols);
			return elements[i * cols + j];
		}

		const T& operator()(std::size_t i, std::size_t j) const
		{
			assert(i < rows && j < cols);
			return elements[i * cols + j];
		}

		matrix transpose()
		{
			std::vector<double> new_elements(rows * cols);

			for(std::size_t i = 0; i < rows; i++)
			{
				for(std::size_t j = 0; j < cols; j++)
				{
					new_elements[j * rows + i] = elements[i * cols + j];
				}
			}

			matrix r(cols, rows, new_elements);
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
	matrix<T> identity(std::size_t n)
	{
		matrix<T> I(n, n);
		for(std::size_t i = 0; i < n; i++)
		{
			for(std::size_t j = 0; j < n; j++)
			{
				if(i == j)
				{
					I(i, j) = 1;
				}
			}
		}

		return I;
	}

	template<typename T>
	bool operator==(const matrix<T>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		auto [b_rows, b_cols] = b.shape();
		assert(a_rows == b_rows && a_cols == b_cols);

		for(std::size_t i = 0; i < a_rows; i++)
		{
			for(std::size_t j = 0; j < a_cols; j++)
			{
				if(a(i, j) != b(i, j))
				{
					return false;
				}
			}
		}

		return true;
	}

	template<typename T>
	matrix<T> operator*=(matrix<T>& a, double num)
	{
		auto [rows, cols] = a.shape();
		for(std::size_t i = 0; i < rows * cols; i++)
		{
			a(i, 0) = a(i, 0) * num;
		}

		return a;
	}

	template<typename T>
	matrix<T> operator/(const matrix<T>& a, double num)
	{
		auto [rows, cols] = a.shape();
		matrix<T> r(rows, cols);
		for(std::size_t i = 0; i < rows; i++)
		{
			for(std::size_t j = 0; j < cols; j++)
			{
				r(i, j) = a(i, j) / num;
			}
		}

		return r;
	}

	template<typename T>
	matrix<T> operator*(double num, const matrix<T>& a)
	{
		auto [rows, cols] = a.shape();
		matrix<T> r(rows, cols);
		for(std::size_t i = 0; i < rows; i++)
		{
			for(std::size_t j = 0; j < cols; j++)
			{
				r(i, j) = a(i, j) * num;
			}
		}

		return r;
	}

	template<typename T>
	matrix<T> operator*(matrix<T>& a, double num)
	{
		auto [rows, cols] = a.shape();
		for(std::size_t i = 0; i < rows; i++)
		{
			for(std::size_t j = 0; j < cols; j++)
			{
				a(i, j) = a(i, j) * num;
			}
		}

		return a;
	}

	template<typename T, typename U>
	matrix<T> operator+(const matrix<T>& a, const matrix<U>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		auto [b_rows, b_cols] = b.shape();
		matrix<T> c(a_rows, a_cols);
		for(std::size_t i = 0; i < a_rows; i++)
		{
			for(std::size_t j = 0; j < a_cols; j++)
			{
				size_t bi = (b_rows == 1) ? 0 : i;
				size_t bj = (b_cols == 1) ? 0 : j;
				c(i, j) = a(i, j) + b(bi, bj);
			}
		}

		return c;
	}

	template<typename T>
	matrix<T> operator-(const matrix<T>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		auto [b_rows, b_cols] = b.shape();
		assert(a_rows == b_rows && a_cols == b_cols);

		matrix<T> c(a_rows, a_cols);
		auto [rows, cols] = c.shape();
		for(std::size_t i = 0; i < rows; i++)
		{
			for(std::size_t j = 0; j < cols; j++)
			{
				c(i, j) = a(i, j) - b(i, j);
			}
		}

		return c;
	}

	template<typename T, typename U>
	matrix<T> operator*(const matrix<U>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		auto [b_rows, b_cols] = b.shape();
		assert(a_cols == b_rows);

		matrix<T> c(a_rows, b_cols);
		auto [rows, cols] = c.shape();
		for(std::size_t i = 0; i < rows; i++)
		{
			for(std::size_t j = 0; j < cols; j++)
			{
				double sum = 0;
				for(std::size_t k = 0; k < a_cols; k++)
				{
					sum += a(i, k) * b(k, j);
				}
				c(i, j) = sum;
			}
		}

		return c;
	}

	template<typename T, typename U>
	matrix<T> hadamard(const matrix<U>& a, const matrix<T>& b)
	{
		auto [a_rows, a_cols] = a.shape();
		auto [b_rows, b_cols] = b.shape();
		assert((a_rows == b_rows) && (a_cols == b_cols));

		matrix<T> c(a_rows, a_cols);
		auto [rows, cols] = c.shape();
		for(std::size_t i = 0; i < rows; i++)
		{
			for(std::size_t j = 0; j < cols; j++)
			{
				c(i, j) = a(i, j) * b(i, j);
			}
		}

		return c;
	}

	template<typename T>
	matrix<T> sum(const matrix<T>& a)
	{
		auto [rows, cols] = a.shape();
		matrix<T> r(1, cols);
		for(std::size_t j = 0; j < cols; j++)
		{
			T sum{0};
			for(std::size_t i = 0; i < rows; i++)
			{
				sum += a(i, j);
			}
			r(0, j) = sum;
		}

		return r;
	}

	template<typename T>
	math::matrix<T> concat(const math::matrix<T>& a, std::vector<T>& b)
	{
		assert(a.shape().first == b.size());
		
		auto [rows, cols] = a.shape();
		math::matrix<T> result(rows, cols + 1);
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				result(i, j) = a(i, j);	
				result(i, j + 1) = b[i];
			}
		}
		
		return result;
	}
};

