#include <vector>
#include "../include/linear_layer.h"
#include "../include/activation_layer.h"
#include "../include/loss_layer.h"
#include "../include/rapidcsv.h"

std::vector<double> xor_truth_table = {
	0, 0, 0, 0,
	0, 0, 1, 1,
	0, 1, 0, 1,
	0, 1, 1, 0,
	1, 0, 0, 1,
	1, 0, 1, 0,
	1, 1, 0, 0,
	1, 1, 1, 1,
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

class counter
{
	public:
		size_t idx;

		counter() : idx(0) {}
};

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

void xor_train(const math::matrix<double>& dataset, 
		double lr, 
		size_t epochs,
	       	size_t batch_size, 
		counter& c, 
		nn::linear_layer<double>& z1,
	       	nn::activation_layer<double>& a1,
	       	nn::linear_layer<double>& z2,
	       	nn::activation_layer<double>& a2,
	       	nn::loss_layer<double>& l)
{
	auto [rows, cols] = dataset.shape();
	size_t rmd = rows % batch_size;
	size_t num_batches = rows / batch_size;

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

			math::matrix<double> batch = create_batches(dataset, batch_size, c);
			math::matrix<double> x_batch = get_cols(batch, 0, 2);
			math::matrix<double> y_batch = get_cols(batch, 2, 3);
			// forward pass
			math::matrix<double> zh = z1.forward(x_batch);
			math::matrix<double> ah = a1.forward(zh);
			math::matrix<double> zo = z2.forward(ah);
			math::matrix<double> y = a2.forward(zo);
			// backward pass
			math::matrix<double> dy_L = l.backward(y, y_batch);
			math::matrix<double> delta_o = a2.backward(dy_L);
			math::matrix<double> dxo = z2.backward(delta_o, static_cast<double>(batch_size));
			math::matrix<double> delta = a1.backward(dxo);
			math::matrix<double> dxi = z1.backward(delta, static_cast<double>(batch_size));
			// update parameters
			z1.update(lr);
			z2.update(lr);
		}
		c.idx = 0;
	}
}

math::matrix<double> predict(const math::matrix<double>& x,
	               	      nn::linear_layer<double>& z1,
	       		      nn::activation_layer<double>& a1,
			      nn::linear_layer<double>& z2, 
			      nn::activation_layer<double>& a2)
{
	math::matrix<double> y = z1.forward(x);
	y = a1.forward(y);
	y = z2.forward(y);
	y = a2.forward(y);
	return y;
}

double accuracy(const math::matrix<double>& predicted, const math::matrix<double>& outputs)
{
	auto [rows, cols] = predicted.shape();
	size_t correct = 0;
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

int main()
{
	math::random<double> rng;

	rapidcsv::Document train_dataset("two_moons.csv");
	std::vector<double> attr1 = train_dataset.GetColumn<double>("x1");
	std::vector<double> attr2 = train_dataset.GetColumn<double>("x2");
	std::vector<double> label = train_dataset.GetColumn<double>("label");
	math::matrix<double> dataset(attr1.size(), 1, attr1);
	dataset = concat(dataset, attr2);
	dataset = concat(dataset, label); 
	
	counter c;
	size_t epochs = 100'000;
	double lr = 0.1;
	size_t batch_size = 100;

	nn::linear_layer<double> z1(2, 4, rng);
	nn::activation_layer<double> a1(activation::type::sigmoid);
	nn::linear_layer<double> z2(4, 1, rng);
	nn::activation_layer<double> a2(activation::type::sigmoid);
	nn::loss_layer<double> l(loss::type::mse);

	xor_train(dataset, lr, epochs, batch_size, c, z1, a1, z2, a2, l);

	math::matrix<double> predicted = predict(get_cols(dataset, 0, 2), z1, a1, z2, a2);
	std::cout << "Accuracy = " << std::round(accuracy(predicted, get_cols(dataset, 2, 3)) * 100) << "%" << "\n";

	return 0;
}
