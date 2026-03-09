#pragma once

#include <iostream>
#include <vector>
#include <cassert>

template<typename T>
class Matrix
{
    private:
        std::size_t rows;
        std::size_t cols;
        std::vector<T> elements;

    public:
        /**
         * @brief Creates a Matrix
         *
         * To success, both parameters must be greater than or equal to 0
         * @param r Number of rows
         * @param c Number of columns
         */
        Matrix(std::size_t r, std::size_t c) : rows(r), cols(c)
        {
            assert(rows > 0 && cols > 0);
            elements.resize(rows * cols);
        }

        /**
         * @brief Creates a Matrix and fill it with vector data
         *
         * To success, both parameters must be greater than or equal to 0
         * @param r Number of rows
         * @param c Number of columns
         * @param v The vector to fill
         */
        Matrix(std::size_t r, std::size_t c, std::vector<T> v) : rows(r), cols(c)
        {
            assert(rows > 0 && cols > 0);
            elements.resize(rows * cols);
            for(std::size_t i = 0; i < rows * cols; i++)
            {
                elements[i] = v[i];
            }
        }

        /**
         * @brief Returns the dimensions of the Matrix
         *
         * Uses a tuple to return the dims
         * @return A mutable tuple with the dims
         */
        std::pair<std::size_t, std::size_t> shape()
        {
            return std::pair<std::size_t, std::size_t>(rows, cols);
        }

        /**
         * @brief Returns the dimensions of the Matrix
         *
         * Uses a tuple to return the dims
         * @return An immutable tuple with the dims
         */
        const std::pair<std::size_t, std::size_t> shape() const
        {
            return std::pair<std::size_t, std::size_t>(rows, cols);
        }

        /**
         * @brief Access and returns the item at i and j
         *
         * Returns the item if its dimensions(row and column) provided are valid
         * @param i Row index
         * @param j Column index
         * @return A reference to item itself
         */
        T& operator()(std::size_t i, std::size_t j)
        {
            assert(i < rows && j < cols);
            return elements[i * cols + j];
        }


        /**
         * @brief Access and returns the item at i and j
         *
         * Returns the item if its dimensions(row and column) provided are valid
         * @param i Row index
         * @param j Column index
         * @return A const reference to item itself
         */
        const T& operator()(std::size_t i, std::size_t j) const
        {
            assert(i < rows && j < cols);
            return elements[i * cols + j];
        }


        /**
         * @brief Transposes a Matrix
         *
         * Transposes a Matrix and return its reference
         *  FIX: Change this method to return a new matrix
         *  WARNING: If used wrong can broke the code
         *
         * @return Transposed Matrix
         */
        Matrix transpose()
        {
            std::vector<double> new_elements(rows * cols);

            for(std::size_t i = 0; i < rows; i++)
            {
                for(std::size_t j = 0; j < cols; j++)
                {
                    new_elements[j * rows + i] = elements[i * cols + j];
                }
            }

            elements = std::move(new_elements);
            std::swap(rows, cols);

            return *this;
        }

        /**
         * @brief Prints a Matrix
         *
         * Print (*this) Matrix
         */
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

/**
 * Multiply a Matrix for a scalar
 *
 * @param a The Matrix to operate
 * @param num The scalar
 * @return The Matrix after the operation
 */
template<typename T>
Matrix<T> operator*=(Matrix<T>& a, double num)
{
    auto [rows, cols] = a.shape();
    for(std::size_t i = 0; i < rows * cols; i++)
    {
        a(i, 0) = a(i, 0) * num;
    }

    return a;
}

template<typename T>
Matrix<T> operator*(double num, const Matrix<T>& a)
{
    auto [rows, cols] = a.shape();
    Matrix<T> r(rows, cols);
    for(std::size_t i = 0; i < rows; i++)
    {
        for(std::size_t j = 0; j < cols; j++)
        {
            r(i, j) = a(i, j) * num;
        }
    }

    return r;
}

/**
 * @brief Adds two Matrices
 *
 * Add two Matrices if both have the same dimensions
 * @param a First operand
 * @param b Second operand
 * @return The Matrix after add
 */
template<typename T, typename U>
Matrix<T> operator+(const Matrix<T>& a, const Matrix<U>& b)
{
    auto [a_rows, a_cols] = a.shape();
    auto [b_rows, b_cols] = b.shape();
    assert(a_rows == b_rows && a_cols == b_cols);

    Matrix<T> c(a_rows, a_cols);
    auto [rows, cols] = c.shape();
    for(std::size_t i = 0; i < rows; i++)
    {
        for(std::size_t j = 0; j < cols; j++)
        {
            c(i, j) = a(i, j) + b(i, j);
        }
    }

    return c;
}

/**
 * @brief Subtract two Matrices
 *
 * Subtract two Matrices if both have the same dimensions
 * @param a First operand
 * @param b Second operand
 * @return The Matrix after subtraction
 */
template<typename T>
Matrix<T> operator-(const Matrix<T>& a, const Matrix<T>& b)
{
    auto [a_rows, a_cols] = a.shape();
    auto [b_rows, b_cols] = b.shape();
    assert(a_rows == b_rows && a_cols == b_cols);

    Matrix<T> c(a_rows, a_cols);
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

/**
 * @brief Multiply two Matrices
 *
 * Multiply two Matrices if first's number of columns
 * and second's number of rows are equal
 * @param a First operand
 * @param b Second operand
 * @return The Matrix after operation
 */
template<typename T, typename U>
Matrix<T> hadamard(const Matrix<U>& a, const Matrix<T>& b)
{
    auto [a_rows, a_cols] = a.shape();
    auto [b_rows, b_cols] = b.shape();
    assert(a_cols == b_rows);

    Matrix<T> c(a_rows, b_cols);
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
Matrix<T> operator*(const Matrix<U>& a, const Matrix<T>& b)
{
    auto [a_rows, a_cols] = a.shape();
    auto [b_rows, b_cols] = b.shape();
    assert((a_rows == b_rows) && (a_cols == b_cols));

    Matrix<T> c(a_rows, a_cols);
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
