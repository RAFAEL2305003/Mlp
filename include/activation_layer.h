#include <stdexcept>
#include "math/matrix.h"
#include "activation.h"

namespace nn
{
	class activation_layer
	{
		private:
		activation::type act_type;
		math::matrix<float> a;
		
		public:
		activation_layer() {}

		activation_layer(activation::type act_type) : act_type(act_type) {}

		math::matrix<float> forward(const math::matrix<float>& z)
		{
			switch(act_type)
			{
				case activation::type::relu:
				{
					a = activation::relu(z);
					return a;
				}
				case activation::type::sigmoid:
				{
					a = activation::sigmoid(z);
					return a;
				}
				case activation::type::softmax:
				{
					a = activation::softmax(z);
					return a;
				}
				default:
				{
					throw std::runtime_error("Invalid activation type.");
				}
			}
		}

		// returns the delta of the current layer
		math::matrix<float> backward(const math::matrix<float>& delta_last)
		{
			switch(act_type)
			{
				case activation::type::relu:
				{
					return math::hadamard(delta_last, activation::relu_derivative(a));
				}
				case activation::type::sigmoid:
				{
					return math::hadamard(delta_last, activation::sigmoid_derivative(a));
				}
				case activation::type::softmax:
				{
					return delta_last;
				}
				default:
				{
					throw std::runtime_error("Invalid activation type.");
				}
			}
		}
	};
};
