#include "math/matrix.h"
#include "math/random.h"

namespace nn
{
	class linear_layer
	{
	private:
		math::matrix<float> x;
		math::matrix<float> w;
		math::matrix<float> b;
		math::matrix<float> z;

		math::matrix<float> dw;
		math::matrix<float> dx;
		math::matrix<float> db;

	public:
		std::size_t input_size;
		std::size_t output_size;

		linear_layer() : input_size(0), output_size(0) {}

		linear_layer(std::size_t input_size,
					 std::size_t output_size,
					 math::random<float>& rng)
			:
				  input_size(input_size),
				  output_size(output_size),
				w(input_size, output_size),
				  b(1, output_size, 0.0) { math::fill_random(w, rng, -1.0f, 1.0f); }

		math::matrix<float> forward(const math::matrix<float>& x)
		{
			this->x = x;
			z = x * w + b;
			return z;
		}

		math::matrix<float> backward(const math::matrix<float>& delta)
		{
			dw = x.transpose() * delta;
			db = math::sum(delta);
			dx = delta * w.transpose();
			return dx;
		}

		void update(float lr)
		{
			math::subtract_scaled_inplace(w, lr, dw);
			math::subtract_scaled_inplace(b, lr, db);
		}
	};
};