#include "math/matrix.h"
#include "math/random.h"

namespace nn
{
	template<typename T>
	class linear_layer
	{
		private:
		math::matrix<T> x;
		math::matrix<T> w;
		math::matrix<T> b;
		math::matrix<T> z;

		math::matrix<T> dw;
		math::matrix<T> dx;
		math::matrix<T> db;

		public:
		std::size_t input_size;
		std::size_t output_size;

		linear_layer() : input_size(0), output_size(0) {}

		linear_layer(std::size_t input_size,
					 std::size_t output_size,
					 math::random<T>& rng)
			: 
			  	input_size(input_size),
			  	output_size(output_size),
				w(input_size, output_size),
              	b(1, output_size) { math::fill_random(w, rng, -1.0, 1.0); }
		
		math::matrix<T> forward(const math::matrix<T>& x)
		{
			this->x = x;
			z = x * w + b;
			return z;
		}

		math::matrix<T> backward(const math::matrix<T>& delta, double batch_size)
		{
			dw = x.transpose() * delta;
			dw = dw / batch_size;
			db = math::sum(delta);
			db = db /batch_size;
			dx = delta * w.transpose();
			dx = dx / batch_size;
			return dx;
		}

		void update(double lr)
		{
			w = w - (lr * dw);
			b = b - (lr * db);
		}
	};
};
