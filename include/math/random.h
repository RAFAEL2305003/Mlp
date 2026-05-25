#pragma once

#include <random>
#include "matrix.h"

namespace math
{
	template<typename T>
	class random
	{
		private:
		inline static std::random_device rd{};
		std::seed_seq ss;
		std::mt19937 mt;

		public:

		random() : ss{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()},
					mt(ss) {}

		T gen_number(T min, T max)
		{
			std::uniform_real_distribution<T> bzo(min, max);
			return bzo(mt);
		}
	};

    template<typename T>
	void fill_random(matrix<T>& m, random<T> &rng, T min, T max)
	{
		auto [rows, cols] = m.shape();
		for(std::size_t i = 0; i < rows; i++)
		{
			for(std::size_t j = 0; j < cols; j++)
			{
				m(i, j) = rng.gen_number(min, max);
			}
		}
	}
};
