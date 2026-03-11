#include "math/matrix.h"
#include "math/random.h"
#include "activation.h"
#include "loss.h"

// This MLP solves the XOR problem with a static number of layers and neurons per layer

namespace nn
{
	class mlp
	{
		private:
        math::matrix<double> X;
        math::matrix<int> Y;
        math::matrix<double> W1;
        math::matrix<double> W2;
        math::matrix<double> B1;
        math::matrix<double> B2;
        math::matrix<double> Z1;
        math::matrix<double> Z2;
        math::matrix<double> A;
        math::matrix<double> y;
        std::size_t input_size;
        std::size_t hidden_size;
        std::size_t output_size;
        std::size_t epochs;
        double lr;

        math::matrix<double> forward()
        {
            // forward from input to hidden
            Z1 = X * W1 + B1;
            A = activation::sigmoid(Z1);

            // forward from hidden to output
            Z2 = A * W2 + B2;
            y = activation::sigmoid(Z2);
            return y;
        }

        void backward()
        {
            // backward from output to hidden layer:
            // dMSE/dW2 = dMSE/dŶ * dŶ/dZ2 * dZ2/dW2
            math::matrix d_mse_y = loss::mse_derivative(Y, y);
            math::matrix d_y_z2 = activation::sigmoid_derivative(y);
            math::matrix delta = math::hadamard(d_mse_y, d_y_z2);
            math::matrix d_mse_w2 = A.transpose() * delta;
 
            // dMSE/dB2 = dMSE/dŶ * dŶ/dZ2 * dZ2/dB2
            auto [rows, cols] = Z2.shape();
            // math::matrix d_mse_b2 = math::matrix<double>(rows, cols,
            //     std::vector<double>(rows * cols, 1)).transpose() * delta;
			math::matrix d_mse_b2 = math::sum(delta);

            // backward from hidden to input
            // dMSE/dW1 = dMSE/dŶ * dŶ/dZ2 * dZ2/dA * dA/dZ1 * dZ1/dW1
            math::matrix d_a_z1 = activation::sigmoid_derivative(A);
            math::matrix delta_h = delta * W2.transpose();
            delta_h = math::hadamard(delta_h, d_a_z1);
            math::matrix d_mse_w1 = X.transpose() * delta_h;

            // dMSE/dB1 = dMSE/dŶ * dŶ/dZ2 * dZ2/dA * dA/dZ1 * dZ1/dB1
            auto [rows_, cols_] = Z1.shape();
            math::matrix d_z1_b1 = math::matrix<double>(rows_, cols_, std::vector<double>(rows_*cols_, 1));
            // math::matrix d_mse_b1 = math::matrix<double>(rows_, cols_, 
		    //					std::vector<double>(rows_*cols_, 1)).transpose() * delta_h;
			
            // std::vector<double> row(cols_);
            // for(size_t j = 0; j < cols_; j++)
            // {
            //     row[j] = d_mse_b1(0, j);
            // }
            // auto [b1_rows, b1_cols] = B1.shape();
            // d_mse_b1 = math::matrix<double>(b1_rows, b1_cols, row);
			math::matrix d_mse_b1 = math::sum(delta_h);

            // updating the weights
            update_weights(d_mse_w1, d_mse_b1, d_mse_w2, d_mse_b2);
        }
    
		void update_weights(const math::matrix<double>& d_mse_w1,
				 			const math::matrix<double>& d_mse_b1,
 							const math::matrix<double>& d_mse_w2,
							const math::matrix<double>& d_mse_b2)
    	{
			W1 = W1 - (lr * d_mse_w1);
			B1 = B1 - (lr * d_mse_b1);
			W2 = W2 - (lr * d_mse_w2);
			B2 = B2 - (lr * d_mse_b2);
		}

		public:
        mlp(std::size_t epochs, double lr, std::size_t input_size,
            std::size_t hidden_size, std::size_t output_size) :
            epochs(epochs), lr(lr), input_size(input_size),
            hidden_size(hidden_size), output_size(output_size),
            X(4, 2, {0, 0, 0, 1, 1, 0, 1, 1}),
            Y(4, 1, {0, 1, 1, 0}), W1(input_size, hidden_size),
            W2(hidden_size, output_size), B1(1, hidden_size),
            B2(1, output_size), Z1(4, 2), Z2(4, 1), A(4, 2),
            y(4, 1)
        {
            math::random<double> rng;
			double min = -1.0; 
			double max = 1.0;
            math::fill_random(W1, rng, min, max);
            math::fill_random(W2, rng, min, max);
        }

        void train()
        {
            for(std::size_t e = 0; e < epochs; e++)
            {
                forward();
                backward();
            }
        }

        void predict()
        {
            forward().print();
        }
	};
};
