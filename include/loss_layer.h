#include <stdexcept>
#include "math/matrix.h"
#include "loss.h"

namespace nn
{
	template<typename T>
	class loss_layer
	{
		private:
		loss::type loss_type;
		
		public:
		loss_layer() {}	

		loss_layer(loss::type loss_type) : loss_type(loss_type) {}	

		T forward(const math::matrix<T>& y, const math::matrix<T>& Y)
		{
			switch(loss_type)
			{
				case loss::type::mse:
				{
					return loss::mse(y, Y);
				}
				default:
				{
					throw std::runtime_error("Invalid loss type");
				}
			}
		}

		math::matrix<T> backward(const math::matrix<T>& y,const math::matrix<T>& Y)
		{
			switch(loss_type)
			{
				case loss::type::mse:
				{
					return loss::mse_derivative(y, Y); 
				}
				default:
				{
					throw std::runtime_error("Invalid loss type");
				}
			}
		}
	};
};
