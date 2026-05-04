#include <cmath>
#include "../include/linear_layer.h"
#include "../include/activation_layer.h"
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

	// todo: shuffle before create a batch
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

			dataset(dataset)
		{
			math::random<double> rng;

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
		math::matrix fwd = linear[0].forward(x_batch);
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
	}

	void update()
	{
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
		if(rmd != 0)
		{
			num_batches += 1;
		}
		for(size_t e = 0; e < epochs; e++)
		{
			double total_loss = 0.0;

			for(size_t i = 0; i < num_batches; i++)
			{
				if(i == num_batches - 1 && rmd != 0)
				{
					batch_size = rmd;
				}

				math::matrix<double> batch = nn::create_batches(dataset, batch_size, c);
				math::matrix<double> x_batch = get_cols(batch, 0, 32);
				math::matrix<double> y_batch = get_cols(batch, 32, 33);

				math::matrix fwd_pass = feedforward(x_batch);
				total_loss += loss.forward(fwd_pass, y_batch);
				backward(fwd_pass, y_batch);			
				update();
			}
			if(e % 100 == 0)
				std::cout << "epoch: " << e << ", loss: " << total_loss / num_batches << "\n";

			batch_size = old_batch_size;
			c.idx = 0;
		}
	}

	double accuracy(const math::matrix<double>& predicted)
	{
		size_t rows = predicted.shape().first;

		std::vector actual = activation::argmax(get_cols(dataset, 32, 33));

		size_t correct = 0;
		for(size_t i = 0; i < rows; i++)
		{
			if(predicted(i, 0) == actual[i])
			{
				correct++;
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

/*math::matrix<double> read_csv(std::string filename)
{

	rapidcsv::Document train_dataset(filename);
	std::vector<double> attr1 = train_dataset.GetColumn<double>("SepalLengthCm");
	std::vector<double> attr2 = train_dataset.GetColumn<double>("SepalWidthCm");
	std::vector<double> attr3 = train_dataset.GetColumn<double>("PetalLengthCm");
	std::vector<double> attr4 = train_dataset.GetColumn<double>("PetalWidthCm");

	std::vector<double> label1 = train_dataset.GetColumn<double>("Species_Iris-setosa");
	std::vector<double> label2 = train_dataset.GetColumn<double>("Species_Iris-versicolor");
	std::vector<double> label3 = train_dataset.GetColumn<double>("Species_Iris-virginica");
	math::matrix<double> dataset(attr1.size(), 1, attr1);
	dataset = math::concat(dataset, attr2);
	dataset = math::concat(dataset, attr3);
	dataset = math::concat(dataset, attr4);
	dataset = math::concat(dataset, label1);
	dataset = math::concat(dataset, label2);
	dataset = math::concat(dataset, label3);

	return dataset; 
}*/

math::matrix<double> read_csv(std::string filename)
{
    rapidcsv::Document attr_dataset(filename);
    std::vector<std::string> attrs = {
        "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted",
        "num_root", "num_file_creations", "num_shells", "num_access_files",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"
    };

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
	size_t epochs = 1'000;
	size_t batch_size = 50'000;

	std::vector<size_t> layers = {32, 16, 8, 4, 1};

	std::vector<activation::type> activations = {activation::type::relu, activation::type::relu, activation::type::relu, activation::type::sigmoid};

	std::string filename = "nsl_kdd.csv";
	math::matrix<double> dataset = read_csv(filename);

	nn::counter c;
	nn::dense_layer dense(epochs, lr, layers, activations, loss::type::bce, dataset);
 	dense.train(batch_size, c);	

	math::matrix predictions = dense.feedforward(nn::get_cols(dataset, 0, 32));
	std::cout << "Accuracy = " << std::round(dense.accuracy(predictions) * 100) << "%" << "\n";

	return 0;
}

