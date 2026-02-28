#pragma once

#include <random>
#include "matrix.h"

template<typename T>
class Random
{
    private:
        inline static std::random_device rd{};
        std::seed_seq ss;
        std::mt19937 mt;
        std::uniform_real_distribution<T> bzo;

    public:
        /**
         * @brief Creates the PRNG
         *
         * Creates the PRNG and an uniform distribution based on type of the
         * desired of random number
         * @param floor - the lowest value
         * @param ceiling - the highest value
         */
        Random(int floor, int ceiling) : ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()},
                    mt(ss), bzo(floor, ceiling) {}

        /**
         * @brief Generates a random number
         *
         * Generates a random number based on desired type
         */
        T genNumber()
        {
            return bzo(mt);
        }

        /**
         * @brief Fills a Matrix with random numbers
         *
         * Fills a Matrix of type T with random numbers
         * @param m - The Matrix to fill
         */
        void fillMatrixWithRand(Matrix<T>& m)
        {
            auto [rows, cols] = m.shape();
            for(std::size_t i = 0; i < rows; i++)
            {
                for(std::size_t j = 0; j < cols; j++)
                {
                    m(i, j) = genNumber();
                }
            }
        }
};