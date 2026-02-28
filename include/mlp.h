#include "matrix/matrix.h"
#include "matrix/random.h"
#include "activation.h"
#include "loss.h"

class Mlp
{
    private:
        Matrix<int> X;
        Matrix<int> Y;
        Matrix<double> W1;
        Matrix<double> W2;
        Matrix<int> b1;
        Matrix<int> b2;
        Matrix<double> z1;
        Matrix<double> z2;
        Matrix<double> a1;
        Matrix<double> a2;
        std::size_t instances_size;
        std::size_t input_size;
        std::size_t hidden_size;
        std::size_t output_size;
        std::size_t epochs;
        double lr;

        void forward_hidden(const Matrix<int>& x)
        {
            z1 = (W1 * x) + b1;
            a1 = ReLU(z1);
        }

        void forward_output(const Matrix<double>& a)
        {
            z2 = (W2 * a) + b2;
            a2 = ReLU(z2);
        }

        void backward_output(const Matrix<int>& y)
        {
            double d_mse = MSE_derivative(Y, y);
        }

    public:
        Mlp(std::size_t epochs, double lr, std::size_t input_size,
            std::size_t hidden_size, std::size_t output_size) :
            epochs(epochs), lr(lr), input_size(input_size), instances_size(4),
            hidden_size(hidden_size), output_size(output_size),
            X(4, 2, {0, 0, 0, 1, 1, 0, 1, 1}), Y(4, 1, {0, 1, 1, 0}),
            W1(input_size, 2), W2(output_size, hidden_size), b1(hidden_size, 1),
            b2(output_size, 1), z1(hidden_size, 1), z2(1, 1), a1(hidden_size, 1),
            a2(1, 1)
        {
            Random<double> r(-1, 1);
            r.fillMatrixWithRand(W1);
            r.fillMatrixWithRand(W2);
            b1(0, 0) = 0;
            b1(1, 0) = 0;
            b2(0, 0) = 0;
        }

        void forward()
        {
            for(size_t e = 0; e < 5; e++)
            {
                std::cout << e << "'th epoch:\n";
                for(size_t i = 0; i < instances_size; i++)
                {
                    std::vector<int> row(input_size);
                    for(size_t j = 0; j < input_size; j++)
                    {
                        row[j] = X(i, j);
                    }
                    Matrix<int> x(input_size, 1, row);
                    forward_hidden(x);
                    forward_output(a1);
                    // std::cout << "forward for instance:\n";
                    // x.print();
                    // std::cout << "\n";
                    // a2.print();
                }
                // backward_output(a2);
            }
        }
};