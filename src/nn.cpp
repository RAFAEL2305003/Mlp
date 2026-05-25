#include <cmath>
#include "../include/linear_layer.h"
#include "../include/activation_layer.h"
#include "../include/conv1d_layer.h"
#include "../include/pooling_layer.h"
#include "../include/loss_layer.h"
#include "../include/rapidcsv.h"

namespace nn {

	class counter
	{
		public:
			size_t idx;

			counter() : idx(0) {}
	};

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

	size_t input_size;
	size_t output_size;

	math::matrix<double> dataset;

	std::vector<nn::conv1d_layer> conv;
	std::vector<nn::activation_layer> conv_activation;
	std::vector<nn::pooling_layer> pool;
	std::vector<nn::linear_layer> linear;
	std::vector<nn::activation_layer> activation;
	nn::loss_layer loss;
	
	public:
	dense_layer(size_t epochs, 
				double learning_rate, 
				std::vector<size_t> layers,
				std::vector<activation::type> activations,
				loss::type loss_type,
				const math::matrix<double>& dataset)
		: 
			epochs(epochs),

			learning_rate(learning_rate),

			num_layers(layers.size() - 1), // excluding input layer

			input_size(layers[0]),

			output_size(layers[layers.size() - 1]),

			dataset(dataset)
		{
			assert(layers.size() > 1);
			assert(activations.size() == num_layers);

			math::random<double> rng;

			conv.resize(0);
			conv_activation.resize(0);
			pool.resize(0);
			linear.resize(num_layers);
			activation.resize(num_layers);
		
			for(size_t i = 0; i < num_layers; i++)
			{
				linear[i] = nn::linear_layer(layers[i], layers[i + 1], rng);
				activation[i] = nn::activation_layer(activations[i]);
			}
			
			loss = nn::loss_layer(loss_type);
		}

	dense_layer(size_t epochs,
				double learning_rate,
				size_t input_size,
				std::vector<conv1d::config> conv_configs,
				std::vector<activation::type> conv_activations,
				std::vector<pooling::config> pool_configs,
				std::vector<size_t> layers,
				std::vector<activation::type> activations,
				loss::type loss_type,
				const math::matrix<double>& dataset)
		:
			epochs(epochs),

			learning_rate(learning_rate),

			num_layers(layers.size() - 1),

			input_size(input_size),

			output_size(layers[layers.size() - 1]),

			dataset(dataset)
		{
			assert(conv_configs.size() == conv_activations.size());
			assert(conv_configs.size() == pool_configs.size());
			assert(layers.size() > 1);
			assert(activations.size() == num_layers);

			math::random<double> rng;

			size_t current_input_size = this->input_size;

			conv.resize(conv_configs.size());
			conv_activation.resize(conv_configs.size());
			pool.resize(pool_configs.size());

			for(size_t i = 0; i < conv_configs.size(); i++)
			{
				conv[i] = nn::conv1d_layer(current_input_size,
										   conv_configs[i].filters,
										   conv_configs[i].kernel_size,
										   rng,
										   conv_configs[i].conv_type,
										   conv_configs[i].stride);
				conv_activation[i] = nn::activation_layer(conv_activations[i]);

				pool[i] = nn::pooling_layer(pool_configs[i].pool_type,
											conv[i].output_size,
											conv_configs[i].filters,
											pool_configs[i].pool_size,
											pool_configs[i].stride);

				current_input_size = conv_configs[i].filters * pool[i].output_size;
			}

			assert(layers[0] == current_input_size);

			linear.resize(num_layers);
			activation.resize(num_layers);

			for(size_t i = 0; i < num_layers; i++)
			{
				linear[i] = nn::linear_layer(layers[i], layers[i + 1], rng);
				activation[i] = nn::activation_layer(activations[i]);
			}

			loss = nn::loss_layer(loss_type);
		}

	math::matrix<double> feedforward(const math::matrix<double>& x_batch)
	{
		math::matrix<double> features = x_batch;
		for(size_t i = 0; i < conv.size(); i++)
		{
			features = conv[i].forward(features);
			features = conv_activation[i].forward(features);
			features = pool[i].forward(features);
		}

		math::matrix fwd = linear[0].forward(features);
		math::matrix act = activation[0].forward(fwd);
		for(size_t i = 1; i < num_layers; i++)
		{
			fwd = linear[i].forward(act);	
			act = activation[i].forward(fwd);
		}

		return act;
	}

	void backward(const math::matrix<double>& Y, const math::matrix<double>& y_batch)	
	{
		math::matrix dL_dy = loss.backward(Y, y_batch);
		math::matrix delta = activation[num_layers - 1].backward(dL_dy);
		math::matrix delta_last = linear[num_layers - 1].backward(delta);
		for(int i = num_layers - 2; i >= 0; i--)
		{
			delta = activation[i].backward(delta_last);	
			delta_last = linear[i].backward(delta);
		}

		for(size_t i = conv.size(); i-- > 0;)
		{
			delta_last = pool[i].backward(delta_last);
			delta_last = conv_activation[i].backward(delta_last);
			delta_last = conv[i].backward(delta_last);
		}
	}

	void update()
	{
		for(size_t i = 0; i < conv.size(); i++)
		{
			conv[i].update(learning_rate);
		}

		for(size_t i = 0; i < num_layers; i++)
		{
			linear[i].update(learning_rate);
		}
	}

	void train(size_t batch_size, counter& c)
	{
		auto [rows, cols] = dataset.shape();
		size_t num_batches = rows / batch_size;
		size_t old_batch_size = batch_size;

		size_t rmd = rows % batch_size;
		(rmd != 0) ? num_batches += 1 : num_batches;

		double total_loss = 0.0;
		for(size_t e = 0; e < epochs; e++)
		{
			total_loss = 0;
			for(size_t i = 0; i < num_batches; i++)
			{
				if(i == num_batches - 1 && rmd != 0)
				{
					batch_size = rmd;
				}

				math::matrix<double> batch = nn::create_batches(dataset, batch_size, c);
				math::matrix<double> x_batch = get_cols(batch, 0, input_size);
				math::matrix<double> y_batch = get_cols(batch, input_size, input_size + output_size);

				math::matrix fwd_pass = feedforward(x_batch);
				total_loss += loss.forward(fwd_pass, y_batch);
				backward(fwd_pass, y_batch);			
				update();
			}
			if(e % 5 == 0)
				std::cout << "epoch: " << e << ", loss: " << total_loss / num_batches << "\n";

			batch_size = old_batch_size;
			c.idx = 0;
		}
		std::cout << "epoch: " << epochs << ", loss: " << total_loss / num_batches << "\n";
	}

	double accuracy(const math::matrix<double>& predicted)
	{
		auto [rows, cols] = predicted.shape();
		math::matrix<double> y = get_cols(dataset, input_size, input_size + output_size);

		size_t correct = 0;
		if(cols == 1)
		{
			for(size_t i = 0; i < rows; i++)
			{
				double predicted_class = predicted(i, 0) >= 0.5 ? 1.0 : 0.0;
				if(predicted_class == y(i, 0))
				{
					correct++;
				}
			}
		}
		else
		{
			std::vector predicted_class = activation::argmax(predicted);
			std::vector actual_class = activation::argmax(y);
			for(size_t i = 0; i < rows; i++)
			{
				if(predicted_class[i] == actual_class[i])
				{
					correct++;
				}
			}
		}

		return static_cast<double> (correct) / rows;
	}

	std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<size_t>>
	confusion_counts(const math::matrix<double>& predicted)
	{
		auto [rows, cols] = predicted.shape();
		math::matrix<double> y = get_cols(dataset, input_size, input_size + output_size);
		size_t num_classes = cols == 1 ? 1 : cols;
		std::vector<size_t> tp(num_classes, 0);
		std::vector<size_t> fp(num_classes, 0);
		std::vector<size_t> fn(num_classes, 0);

		if(cols == 1)
		{
			for(size_t i = 0; i < rows; i++)
			{
				bool p = predicted(i, 0) >= 0.5;
				bool a = y(i, 0) >= 0.5;
				if(p && a) tp[0]++;
				else if(p && !a) fp[0]++;
				else if(!p && a) fn[0]++;
			}
		}
		else
		{
			std::vector predicted_class = activation::argmax(predicted);
			std::vector actual_class = activation::argmax(y);
			for(size_t i = 0; i < rows; i++)
			{
				size_t p = predicted_class[i];
				size_t a = actual_class[i];
				if(p == a)
				{
					tp[p]++;
				}
				else
				{
					fp[p]++;
					fn[a]++;
				}
			}
		}

		return {tp, fp, fn};
	}

	double precision(const math::matrix<double>& predicted)
	{
		auto [tp, fp, fn] = confusion_counts(predicted);
		size_t num_classes = tp.size();
		double sum = 0.0;
		for(size_t c = 0; c < num_classes; c++)
		{
			size_t denom = tp[c] + fp[c];
			if(denom > 0)
			{
				sum += static_cast<double>(tp[c]) / static_cast<double>(denom);
			}
		}
		return sum / static_cast<double>(num_classes);
	}

	double recall(const math::matrix<double>& predicted)
	{
		auto [tp, fp, fn] = confusion_counts(predicted);
		size_t num_classes = tp.size();
		double sum = 0.0;
		for(size_t c = 0; c < num_classes; c++)
		{
			size_t denom = tp[c] + fn[c];
			if(denom > 0)
			{
				sum += static_cast<double>(tp[c]) / static_cast<double>(denom);
			}
		}
		return sum / static_cast<double>(num_classes);
	}

	double f1_score(const math::matrix<double>& predicted)
	{
		auto [tp, fp, fn] = confusion_counts(predicted);
		size_t num_classes = tp.size();
		double sum = 0.0;
		for(size_t c = 0; c < num_classes; c++)
		{
			double p_denom = static_cast<double>(tp[c] + fp[c]);
			double r_denom = static_cast<double>(tp[c] + fn[c]);
			double p = p_denom > 0.0 ? static_cast<double>(tp[c]) / p_denom : 0.0;
			double r = r_denom > 0.0 ? static_cast<double>(tp[c]) / r_denom : 0.0;
			if(p + r > 0.0)
			{
				sum += 2.0 * p * r / (p + r);
			}
		}
		return sum / static_cast<double>(num_classes);
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

math::matrix<double> read_csv(std::string filename)
{
    rapidcsv::Document attr_dataset(filename);
    std::vector<std::string> attrs = attr_dataset.GetColumnNames();
    assert(!attrs.empty());

    std::vector<double> attr = attr_dataset.GetColumn<double>(attrs[0]);
    math::matrix<double> dataset(attr.size(), 1, attr);
    for(size_t i = 1; i < attrs.size(); i++)
    {
        attr = attr_dataset.GetColumn<double>(attrs[i]);
        dataset = math::concat(dataset, attr);
    }

    return dataset;
}


int main()
{
	// todo: add this hyperparams to a json file
	double lr = 0.01;
	size_t epochs = 100;
	size_t batch_size = 4096;
	size_t input_size = 16;

	std::vector<conv1d::config> conv_configs = {
		{4, 3, conv1d::type::valid, 1},
		{4, 3, conv1d::type::valid, 1}
	};
	std::vector<activation::type> conv_activations = {
		activation::type::relu,
		activation::type::relu
	};
	std::vector<pooling::config> pool_configs = {
		{pooling::type::max, 2, 2},
		{pooling::type::max, 2, 2}
	};

	std::string filename = "can_ids.csv";
	math::matrix<double> dataset = read_csv(filename);
	size_t dataset_cols = dataset.shape().second;
	assert(dataset_cols > input_size);
	size_t output_size = dataset_cols - input_size;

	std::vector<size_t> layers = {52, 16, 8, 4, output_size};

	std::vector<activation::type> activations = {
		activation::type::relu,
		activation::type::relu,
		activation::type::relu,
		output_size == 1 ? activation::type::sigmoid : activation::type::softmax
	};

	loss::type loss = output_size == 1 ? loss::type::bce : loss::type::ce;

	nn::counter c;
	nn::dense_layer dense(epochs,
						  lr,
						  input_size,
						  conv_configs,
						  conv_activations,
						  pool_configs,
						  layers,
						  activations,
						  loss,
						  dataset);
 	dense.train(batch_size, c);

	math::matrix predictions = dense.feedforward(nn::get_cols(dataset, 0, input_size));
	std::cout << "Accuracy  = " << std::round(dense.accuracy(predictions) * 100) << "%" << "\n";
	std::cout << "Precision = " << std::round(dense.precision(predictions) * 100) << "%" << "\n";
	std::cout << "F1-score  = " << std::round(dense.f1_score(predictions) * 100) << "%" << "\n";

	return 0;
}
