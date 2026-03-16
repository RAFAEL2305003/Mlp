#include "../include/linear_layer.h"
#include "../include/activation_layer.h"
#include "../include/loss_layer.h"

class dense_layer
{
	private:
	size_t epochs;

	double learning_rate;

	size_t num_linear_layers;
	size_t num_activation_layers;

	math::matrix<double> inputs;
	math::matrix<double> outputs;

	std::vector<nn::linear_layer<double>> linear_layers;
	std::vector<nn::activation_layer<double>> activation_layers;
	nn::loss_layer<double> mse_layer;
	
	public:
	dense_layer(size_t epochs, 
				double learning_rate, 
				size_t num_linear_layers,
				size_t num_activation_layers,
				math::matrix<double> inputs,
				math::matrix<double> outputs)
		: 
			epochs(epochs),

			learning_rate(learning_rate),

			num_linear_layers(num_linear_layers),
			num_activation_layers(num_activation_layers),

			inputs(inputs),
			outputs(outputs)
		{
			math::random<double> rng;
		
			linear_layers.resize(num_linear_layers);
			linear_layers[0] = nn::linear_layer<double>(2, 8, rng);
			for(size_t i = 1; i < num_linear_layers - 1; i++)
			{
				linear_layers[i] = nn::linear_layer<double>(linear_layers[i - 1].output_size, 8, rng);
			}
			linear_layers[num_linear_layers - 1] = nn::linear_layer<double>(linear_layers[num_linear_layers - 2].output_size, 1, rng);

			activation_layers.resize(num_activation_layers);
			for(size_t i = 0; i < num_activation_layers; i++)
			{
				activation_layers[i] = nn::activation_layer<double>(activation::type::sigmoid);
			}

			mse_layer = nn::loss_layer<double>(loss::type::mse);
		}

	math::matrix<double> feedforward()
	{
		math::matrix fwd = linear_layers[0].forward(inputs);
		math::matrix act = activation_layers[0].forward(fwd);
		for(size_t i = 1; i < linear_layers.size(); i++)
		{
			fwd = linear_layers[i].forward(act);	
			act = activation_layers[i].forward(fwd);
		}

		return act;
	}

	void backward(const math::matrix<double>& Y)	
	{
		math::matrix dL_dy = mse_layer.backward(Y, outputs); // dL/dŷ
		math::matrix delta = activation_layers[num_activation_layers - 1].backward(dL_dy);  // delta = dL/dŷ * dŷ/dz (for output layer)
		math::matrix delta_last = linear_layers[num_activation_layers - 1].backward(delta); // delta_last = dL/dŷ * dŷ/dz * dz/da
		for(int i = num_linear_layers - 2; i >= 0; i--)
		{
			delta = activation_layers[i].backward(delta_last);	
			delta_last = linear_layers[i].backward(delta);
		}
	}

	void update()
	{
		for(size_t i = 0; i < linear_layers.size(); i++)
		{
			linear_layers[i].update(learning_rate);
		}
	}

	void train()
	{
		for(size_t e = 0; e < epochs; e++)
		{
			math::matrix fwd_pass = feedforward();
			backward(fwd_pass);			
			update();
		}
	}
};

int main()
{
	math::matrix<double> x = math::matrix<double>(4, 2, {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0});
	math::matrix<double> y = math::matrix<double>(4, 1, {0.0, 1.0, 1.0, 0.0});

	double lr = 0.1;
	size_t epochs = 10'000;
	dense_layer dense(epochs, lr, 3, 3, x, y);
	dense.train();	
	dense.feedforward().print();	
	return 0;
}

