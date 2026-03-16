#include <cmath>
#include "../include/linear_layer.h"
#include "../include/activation_layer.h"
#include "../include/loss_layer.h"
#include "../include/rapidcsv.h"

class dense_layer
{
	private:
	size_t epochs;

	double learning_rate;

	size_t num_layers;

	math::matrix<double> inputs;
	math::matrix<double> outputs;

	std::vector<nn::linear_layer<double>> linear_layers;
	std::vector<nn::activation_layer<double>> activation_layers;
	nn::loss_layer<double> mse_layer;
	
	public:
	dense_layer(size_t epochs, 
				double learning_rate, 
				size_t num_layers,
				math::matrix<double> inputs,
				math::matrix<double> outputs)
		: 
			epochs(epochs),

			learning_rate(learning_rate),

			num_layers(num_layers),

			inputs(inputs),
			outputs(outputs)
		{
			math::random<double> rng;
		
			linear_layers.resize(num_layers);
			activation_layers.resize(num_layers);

			linear_layers[0] = nn::linear_layer<double>(2, 4, rng);
			activation_layers[0] = nn::activation_layer<double>(activation::type::sigmoid);
			for(size_t i = 1; i < num_layers - 1; i++)
			{
				linear_layers[i] = nn::linear_layer<double>(linear_layers[i - 1].output_size, 4, rng);
				activation_layers[i] = nn::activation_layer<double>(activation::type::sigmoid);
			}
			linear_layers[num_layers - 1] = nn::linear_layer<double>(linear_layers[num_layers - 2].output_size, 1, rng);
			activation_layers[num_layers - 1] = nn::activation_layer<double>(activation::type::sigmoid);

			mse_layer = nn::loss_layer<double>(loss::type::mse);
		}

	math::matrix<double> feedforward()
	{
		math::matrix fwd = linear_layers[0].forward(inputs);
		math::matrix act = activation_layers[0].forward(fwd);
		for(size_t i = 1; i < num_layers; i++)
		{
			fwd = linear_layers[i].forward(act);	
			act = activation_layers[i].forward(fwd);
		}

		return act;
	}

	void backward(const math::matrix<double>& Y)	
	{
		math::matrix dL_dy = mse_layer.backward(Y, outputs);
		math::matrix delta = activation_layers[num_layers - 1].backward(dL_dy); 
		math::matrix delta_last = linear_layers[num_layers - 1].backward(delta);
		for(int i = num_layers - 2; i >= 0; i--)
		{
			delta = activation_layers[i].backward(delta_last);	
			delta_last = linear_layers[i].backward(delta);
		}
	}

	void update()
	{
		for(size_t i = 0; i < num_layers; i++)
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

	double accuracy(const math::matrix<double>& predicted)
	{
		auto [rows, cols] = predicted.shape();
		size_t correct = 0;
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				if(predicted(i, j) - outputs(i, j) == 0)
				{
					correct++;
				}
			}
		}		

		return static_cast<double> (correct) / rows;
	}
};

math::matrix<double> round(math::matrix<double>& predictions)
{
	auto [rows, cols] = predictions.shape();
	math::matrix<double> r(rows, cols);
	
	for(size_t i = 0; i < rows; i++)
	{
		for(size_t j = 0; j < cols; j++)
		{
			r(i, j) = std::round(predictions(i, j));
		}
	}
	
	return r;
}

int main()
{
	double lr = 0.1;
	size_t epochs = 5'000;
	size_t num_layers = 4;

	rapidcsv::Document train_dataset("two_moons.csv");

	// creating the matrix of attributes
	std::vector<double> attr1 = train_dataset.GetColumn<double>("x1");
	std::vector<double> attr2 = train_dataset.GetColumn<double>("x2");
	math::matrix<double> x(attr1.size(), 1, attr1);
	x = concat(x, attr2);

	// creating the matrix of labels
	std::vector<double> labels = train_dataset.GetColumn<double>("label");
	math::matrix<double> y(labels.size(), 1, labels); 

	dense_layer dense(epochs, lr, num_layers, x, y);
	dense.train();	

	math::matrix predictions = dense.feedforward();
	predictions = round(predictions);
	std::cout << "Accuracy = " << dense.accuracy(predictions) << "\n";

	return 0;
}

