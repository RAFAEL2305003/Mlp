#include "math/matrix.h"
#include "math/random.h"
#include "activation.h"
#include "loss.h"

namespace nn
{
	enum class layer_type
	{
		input,
		hidden,
		output
	};

	template<typename T, typename V, typename U>
	class linear_layer
	{
		private:
		std::size_t input_size;
		std::size_t hidden_size;
		std::size_t output_size;

		math::matrix<T> x;
		math::matrix<V> w;
		math::matrix<V> b;
		math::matrix<V> z;
		math::matrix<U> a;
	
		layer_type type;	
		activation::type activation_type;
		
		public:
		linear_layer(std::size_t input_size,
				     std::size_t hidden_size,
					 std::size_t output_size,
					 const math::matrix<T>& x,
					 math::random<V>& rng,
					 layer_type type,
					 activation::type activation_type)
			: 
			  	input_size(input_size),
			  	hidden_size(hidden_size),
			  	output_size(output_size),

				x(x), 
			  
				w(input_size, hidden_size),
              	b(1, hidden_size),
				
			  	z(x.shape().first, hidden_size),
			  	a(x.shape().first, hidden_size),
				
				type(type),
				activation_type(activation_type) { math::fill_random(w, rng, -1.0, 1.0); }
		
		math::matrix<U> forward()
		{
			z = x * w + b;
			switch(activation_type)
			{
				case activation::type::relu:
					a = activation::relu(z);
				case activation::type::sigmoid:
					a = activation::sigmoid(z);
			}
			return a;
		}

		std::vector<V> backward(const math::matrix<V>& previous_delta)
		{
			switch(type)
			{
				case layer_type::output: {
					math::matrix dL_da = loss::mse_derivative(a);
					break;
				}
			}
		
			math::matrix<V> delta;	
			math::matrix<V> dz_dw;	
			math::matrix<V> dz_db;

			switch(activation_type)
			{
				case activation::type::relu: 
				{
					math::matrix da_dz = activation::relu_derivative(a);
					math::matrix<V> delta = hadamard(dL_da, da_dz);
					delta = hadamard(previous_delta, delta);
					dz_dw = a.tranpose() * delta;
					break;
				}
				case activation::type::sigmoid:
				{
					math::matrix da_dz = activation::sigmoid_derivative(a);
					math::matrix<V> delta = hadamard(dL_da, da_dz);
					delta = hadamard(previous_delta, delta);
					dz_dw = a.tranpose() * delta;
					break;
				}
			} 
			math::matrix<V> delta = hadamard(dL_da, da_dz);
			delta = hadamard(previous_delta, delta);
			dz_dw = a.tranpose() * delta;
		}
	};
};
				
