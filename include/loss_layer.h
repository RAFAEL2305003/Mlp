#include <stdexcept>
#include "math/matrix.h"
#include "loss.h"

namespace nn
{
	class loss_layer
	{
		private:
		loss::type loss_type;
		
		public:
		loss_layer() {}	

		loss_layer(loss::type loss_type) : loss_type(loss_type) {}	

		double forward(const math::matrix<double>& y, const math::matrix<double>& Y)
		{
			switch(loss_type)
			{
				case loss::type::mse:
				{
					return loss::mse(y, Y);
				}
				case loss::type::bce:
				{
					return loss::bce(y, Y);
				}
				case loss::type::ce:
				{
					return loss::ce(y, Y);
				}
				default:
				{
					throw std::runtime_error("Invalid loss type");
				}
			}
		}

		math::matrix<double> backward(const math::matrix<double>& y,const math::matrix<double>& Y)
		{
			switch(loss_type)
			{
				case loss::type::mse:
				{
					return loss::mse_derivative(y, Y); 
				}
				case loss::type::bce:
				{
					return loss::bce_derivative(y, Y);
				}
				case loss::type::ce:
				{
					return loss::ce_derivative(y, Y);
				}	
				default:
				{
					throw std::runtime_error("Invalid loss type");
				}
			}
		}
	};
};
