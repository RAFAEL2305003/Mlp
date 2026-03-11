#include "../include/mlp.h"

int main()
{
	double lr = 0.1;
	size_t epochs = 10'000;
	size_t input_size = 2;
	size_t hidden_size = 8;
	size_t output_size = 1;

    nn::mlp neural_net(epochs, lr, input_size, hidden_size, output_size);
    neural_net.train();
    neural_net.predict();

    return 0;
}
