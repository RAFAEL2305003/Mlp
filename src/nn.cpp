#include <cmath>
#include "../include/linear_layer.h"
#include "../include/activation_layer.h"
#include "../include/loss_layer.h"
#include "../include/rapidcsv.h"

// TODO: Parameterize the number of neurons in each layer

namespace nn {

	class counter
	{
		public:
			size_t idx;

			counter() : idx(0) {}
	};

	// get the columns at the interval [low high)
	math::matrix<double> get_cols(const math::matrix<double>& m, size_t low, size_t high)
	{
		auto [rows, cols] = m.shape();
		assert(low < high);
		assert(low <= cols && high <= cols);
	
		std::vector<double> es;
		for(size_t i = 0; i < rows; i++)
		{
		 	for(size_t j = low; j < high; j++)
		 	{
				es.push_back(m(i, j));
		 	}
		}
	
		math::matrix<double> r(rows, high - low, es);
	
		return r;
	}


	math::matrix<double> create_batches(const math::matrix<double>& dataset, size_t batch_size, counter& c)
	{
		auto [rows, cols] = dataset.shape();
		assert(c.idx < rows);
		math::matrix<double> batch(batch_size, cols);

		for(size_t i = c.idx; i < batch_size + c.idx; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				batch(i - c.idx, j) = dataset(i, j);
			}
		}

		c.idx += batch_size;
		return batch;
	}

class dense_layer
{
	private:
	size_t epochs;

	double learning_rate;

	size_t num_layers;

	math::matrix<double> dataset;
	// math::matrix<double> inputs;
	// math::matrix<double> outputs;

	std::vector<nn::linear_layer<double>> linear_layers;
	std::vector<nn::activation_layer<double>> activation_layers;
	nn::loss_layer<double> mse_layer;
	
	public:
	dense_layer(size_t epochs, 
				double learning_rate, 
				size_t num_layers,
				const math::matrix<double>& dataset)
		: 
			epochs(epochs),

			learning_rate(learning_rate),

			num_layers(num_layers),

			dataset(dataset)
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

	math::matrix<double> feedforward(const math::matrix<double>& x_batch)
	{
		math::matrix fwd = linear_layers[0].forward(x_batch);
		math::matrix act = activation_layers[0].forward(fwd);
		for(size_t i = 1; i < num_layers; i++)
		{
			fwd = linear_layers[i].forward(act);	
			act = activation_layers[i].forward(fwd);
		}

		return act;
	}

	void backward(const math::matrix<double>& Y, const math::matrix<double>& y_batch)	
	{
		math::matrix dL_dy = mse_layer.backward(Y, y_batch);
		math::matrix delta = activation_layers[num_layers - 1].backward(dL_dy); 
		math::matrix delta_last = linear_layers[num_layers - 1].backward(delta, 1);
		for(int i = num_layers - 2; i >= 0; i--)
		{
			delta = activation_layers[i].backward(delta_last);	
			delta_last = linear_layers[i].backward(delta, 1);
		}
	}

	void update()
	{
		for(size_t i = 0; i < num_layers; i++)
		{
			linear_layers[i].update(learning_rate);
		}
	}

	void train(size_t batch_size, counter& c)
	{
		auto [rows, cols] = dataset.shape();
		size_t num_batches = rows / batch_size;
		size_t old_batch_size = batch_size;

		size_t rmd = rows % batch_size;
		if(rmd != 0)
		{
			num_batches += 1;
		}

		for(size_t e = 0; e < epochs; e++)
		{
			for(size_t i = 0; i < num_batches; i++)
			{
				if(i == num_batches - 1 && rmd != 0)
				{
					batch_size = rmd;
				}

				math::matrix<double> batch = nn::create_batches(dataset, batch_size, c);
				math::matrix<double> x_batch = get_cols(batch, 0, 2);
				math::matrix<double> y_batch = get_cols(batch, 2, 3);

				math::matrix fwd_pass = feedforward(x_batch);
				backward(fwd_pass, y_batch);			
				update();
			}

			batch_size = old_batch_size;
			c.idx = 0;
		}
	}

	double accuracy(const math::matrix<double>& predicted)
	{
		auto [rows, cols] = predicted.shape();
		size_t correct = 0;
		math::matrix<double> outputs = get_cols(dataset, 2, 3);
		for(size_t i = 0; i < rows; i++)
		{
			for(size_t j = 0; j < cols; j++)
			{
				int pred = (predicted(i, j) > 0.5) ? 1 : 0;
				if(pred == outputs(i, j))
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

};

int main()
{
	double lr = 0.1;
	size_t epochs = 3'000;
	size_t num_layers = 5;
	size_t batch_size = 51;

	rapidcsv::Document train_dataset("two_moons.csv");

	// creating the matrix of attributes
	std::vector<double> attr1 = train_dataset.GetColumn<double>("x1");
	std::vector<double> attr2 = train_dataset.GetColumn<double>("x2");
	std::vector<double> label = train_dataset.GetColumn<double>("label");
	math::matrix<double> dataset(attr1.size(), 1, attr1);
	dataset = concat(dataset, attr2);
	dataset = concat(dataset, label);

	nn::counter c;
	nn::dense_layer dense(epochs, lr, num_layers, dataset);
	dense.train(batch_size, c);	

	math::matrix predictions = dense.feedforward(nn::get_cols(dataset, 0, 2));
	std::cout << "Accuracy = " << std::round(dense.accuracy(predictions) * 100) << "%" << "\n";

	return 0;
}

