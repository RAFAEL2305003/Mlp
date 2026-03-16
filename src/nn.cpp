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

			linear_layers[0] = nn::linear_layer<double>(4, 8, rng);
			activation_layers[0] = nn::activation_layer<double>(activation::type::sigmoid);
			for(size_t i = 1; i < num_layers - 1; i++)
			{
				linear_layers[i] = nn::linear_layer<double>(linear_layers[i - 1].output_size, 8, rng);
				activation_layers[i] = nn::activation_layer<double>(activation::type::sigmoid);
			}
			linear_layers[num_layers - 1] = nn::linear_layer<double>(linear_layers[num_layers - 2].output_size, 3, rng);
			activation_layers[num_layers - 1] = nn::activation_layer<double>(activation::type::softmax);

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
};

int main()
{
	double lr = 0.1;
	size_t epochs = 10'000;
	size_t num_layers = 4;

	rapidcsv::Document train_dataset("IrisTest.csv");

	// creating the matrix of attributes
	std::vector<double> sepal_l = train_dataset.GetColumn<double>("SepalLengthCm");
	std::vector<double> sepal_w = train_dataset.GetColumn<double>("SepalWidthCm");
	std::vector<double> petal_l = train_dataset.GetColumn<double>("PetalLengthCm");
	std::vector<double> petal_w = train_dataset.GetColumn<double>("PetalWidthCm");
	math::matrix<double> x(sepal_l.size(), 1, sepal_l);
	x = concat(x, sepal_w);
	x = concat(x, petal_l);
	x = concat(x, petal_w);

	// creating the matrix of labels
	std::vector<double> y_setosa = train_dataset.GetColumn<double>("Species_Iris-setosa");
	std::vector<double> y_versicolor = train_dataset.GetColumn<double>("Species_Iris-versicolor");
	std::vector<double> y_virginica = train_dataset.GetColumn<double>("Species_Iris-virginica");
	math::matrix<double> y(y_setosa.size(), 1, y_setosa); 
	y = concat(y, y_versicolor);
	y = concat(y, y_virginica);

	dense_layer dense(epochs, lr, num_layers, x, y);
	dense.train();	

	math::matrix predictions = dense.feedforward();
	predictions.print();

	return 0;
}

