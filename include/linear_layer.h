#include "math/matrix.h"
#include "math/random.h"

namespace nn
{
	template<typename T, typename U, typename V>
	class linear_layer
	{
		private:
		math::matrix<T> x;
		math::matrix<V> w;
		math::matrix<V> b;
		math::matrix<V> z;
		math::matrix<U> a;
		std::size_t input_size;
		std::size_t hidden_size;
		std::size_t output_size;

		public:
		linear_layer<T, U, V>(std::size_t input_size,
							  std::size_t hidden_size,
							  std::size_t output_size,
							  const math::matrix<T>& x,
							  math::random<V>& rng)
			: 
			  	input_size(input_size),
			  	hidden_size(hidden_size),
			  	output_size(output_size),

				x(x), 
			  
				w(input_size, hidden_size),
              	b(1, hidden_size),
				
			  	z(x.shape().first, hidden_size),
			  	a(x.shape().first, hidden_size) { math::fill_random(w, rng, -1.0, 1.0); }

		
	};
};
				
