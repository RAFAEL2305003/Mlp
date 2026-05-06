#include "math/matrix.h"
#include "math/random.h"

namespace nn
{
	class linear_layer
	{
	private:
		math::matrix<double> x;
		math::matrix<double> w;
		math::matrix<double> b;
		math::matrix<double> z;

		math::matrix<double> dw;
		math::matrix<double> dx;
		math::matrix<double> db;

	public:
		std::size_t input_size;
		std::size_t output_size;

		linear_layer() : input_size(0), output_size(0) {}

		linear_layer(std::size_t input_size,
					 std::size_t output_size,
					 math::random<double>& rng)
			:
				  input_size(input_size),
				  output_size(output_size),
				w(input_size, output_size),
				  b(1, output_size) { math::fill_random(w, rng, -1.0, 1.0); }

		math::matrix<double> forward(const math::matrix<double>& x)
		{
			this->x = x;
			z = x * w + b;
			return z;
		}

		math::matrix<double> backward(const math::matrix<double>& delta)
		{
			dw = x.transpose() * delta;
			db = math::sum(delta);
			dx = delta * w.transpose();
			return dx;
		}

		void update(double lr)
		{
			w = w - (lr * dw);
			b = b - (lr * db);
		}
	};
};