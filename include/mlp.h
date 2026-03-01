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
        std::size_t input_size;
        std::size_t hidden_size;
        std::size_t output_size;
        std::size_t epochs;
        double lr;

        void forward()
        {
            // forward from input to hidden
            z1 = (X * W1);
            auto [rows, cols] = z1.shape();
            for(std::size_t i = 0; i < rows; i++)
            {
                for(std::size_t j = 0; j < cols; j++)
                {
                    z1(i, j) += b1(0, j);
                }
            }
            a1 = ReLU(z1);
            /**
             * The result for z1 consists in a matrix with 4 rows and 2 cols
             *
             * An output example:
             * [0                0] -> in each row is the output of each neuron
             * [-0.792214 0.555185]    in hidden layer for each sample. So
             * [-0.520921 0.545377]    for sample (0,0) the output of each
             * [-1.31313   1.10056]    neuron is 0 (for neuron 1 and 2).
             */

            // forward from hidden to output
            z2 = (z1 * W2);
            auto [rows_, cols_] = z2.shape();
            for(std::size_t i = 0; i < rows_; i++)
            {
                for(std::size_t j = 0; j < cols_; j++)
                {
                    z2(i, j) += b2(0, j);
                }
            }
            a2 = ReLU(z2);
            /**
             * The result for z2 consists in a matrix with 4 rows and 1 cols
             *
             * An output example:
             * [0        ] -> This is the output of the model for each sample.
             * [0        ]    So for [0, 0] sample, the model returns 0 as output
             * [0.135342 ]    and so on.
             * [0        ]
             */
        }

        void backward()
        {
        }

    public:
        Mlp(std::size_t epochs, double lr, std::size_t input_size,
            std::size_t hidden_size, std::size_t output_size) :
            epochs(epochs), lr(lr), input_size(input_size),
            hidden_size(hidden_size), output_size(output_size),
            X(input_size, hidden_size, {0, 0, 0, 1, 1, 0, 1, 1}),
            Y(input_size, output_size, {0, 1, 1, 0}), W1(2, hidden_size),
            W2(hidden_size, output_size), b1(1, hidden_size),
            b2(output_size, 1), z1(hidden_size, 1), z2(1, 1), a1(hidden_size, 1),
            a2(output_size, 1)
        {
            Random<double> r(-1, 1);
            r.fillMatrixWithRand(W1);
            r.fillMatrixWithRand(W2);
            b1(0, 0) = 0;
            b1(0, 1) = 0;
            b2(0, 0) = 0;
        }

        void train()
        {
            for(std::size_t e = 0; e < 1; e++)
            {
                forward();
                backward();
            }
        }
};