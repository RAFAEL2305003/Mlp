#include "matrix/matrix.h"
#include "matrix/random.h"
#include "activation.h"
#include "loss.h"

// This MLP solves the XOR problem with a static number of layers and neurons per layer

class Mlp
{
    private:
        Matrix<double> X;
        Matrix<int> Y;
        Matrix<double> W1;
        Matrix<double> W2;
        Matrix<double> B1;
        Matrix<double> B2;
        Matrix<double> Z1;
        Matrix<double> Z2;
        Matrix<double> A;
        Matrix<double> y;
        std::size_t input_size;
        std::size_t hidden_size;
        std::size_t output_size;
        std::size_t epochs;
        double lr;

        Matrix<double> forward()
        {
            // forward from input to hidden
            Z1 = hadamard(X, W1);
            auto [rows, cols] = Z1.shape();
            for(std::size_t i = 0; i < rows; i++)
            {
                Z1(i, 0) += B1(0, 0);
                Z1(i, 1) += B1(0, 1);
            }
            A = sigmoid(Z1);
            /**
             * The result for Z1 consists in a matrix with 4 rows and 2 cols
             *
             * An output example:
             * [0                0] -> in each row is the output of each neuron
             * [-0.792214 0.555185]    in hidden layer for each sample. So
             * [-0.520921 0.545377]    for sample (0,0) the output of each
             * [-1.31313   1.10056]    neuron is 0 (for neuron 1 and 2).
             */

            // forward from hidden to output
            Z2 = hadamard(A, W2);
            auto [rows_, cols_] = Z2.shape();
            for(std::size_t i = 0; i < rows_; i++)
            {
                Z2(i, 0) += B2(0, 0);
            }
            y = sigmoid(Z2);
            /**
             * The result for Z2 consists in a matrix with 4 rows and 1 cols
             *
             * An output example:
             * [0        ] -> This is the output of the model for each sample.
             * [0        ]    So for [0, 0] sample, the model returns 0 as output
             * [0.135342 ]    and so on.
             * [0        ]
             */
            return y;
        }

        void backward()
        {
            Matrix d_mse_y = mse_derivative(Y, y);
            // d_mse_y.print();
            Matrix d_y_z2 = sigmoid_derivative(y);
            // d_y_z2.print();
            Matrix delta = d_mse_y * d_y_z2;
            // delta.print();
            Matrix d_z2_w2 = A;
            Matrix d_mse_w2 = hadamard(d_z2_w2.transpose(), delta);
            // d_mse_w2.print();
            auto [rows, cols] = Z2.shape();
            Matrix d_z2_b2 = Matrix<double>(rows, cols, {1, 1, 1, 1});
            Matrix d_mse_b2 = hadamard(d_z2_b2.transpose(), delta);
            // d_mse_b2.print();

            Matrix d_z2_a = W2;
            // d_z2_a.print();
            Matrix d_a_z1 = sigmoid_derivative(A);
            // d_a_z1.print();
            Matrix delta_h = hadamard(delta, d_z2_a.transpose());
            // delta_h.print();
            delta_h = delta_h * d_a_z1;
            // delta_h.print();
            Matrix d_z1_w1 = X;
            // d_z1_w1.print();
            Matrix d_mse_w1 = hadamard(d_z1_w1.transpose(), delta_h);
            // d_mse_w1.print();
            auto [rows_, cols_] = Z1.shape();
            Matrix d_z1_b1 = Matrix<double>(rows_, cols_, {1, 1, 1, 1, 1, 1, 1, 1});
            // d_z1_b1.print();
            Matrix d_mse_b1 = hadamard(d_z1_b1.transpose(), delta_h);
            // d_mse_b1.print();
            std::vector<double> row(cols_);
            for(size_t j = 0; j < cols_; j++)
            {
                row[j] = d_mse_b1(0, j);
            }
            auto [b1_rows, b1_cols] = B1.shape();
            d_mse_b1 = Matrix<double>(b1_rows, b1_cols, row);
            // d_mse_b1.print();
            update_wheights(d_mse_w1, d_mse_b1, d_mse_w2, d_mse_b2);
        }
    
    void update_wheights(const Matrix<double>& d_mse_w1, const Matrix<double>& d_mse_b1, const Matrix<double>& d_mse_w2, const Matrix<double>& d_mse_b2)
    {
        W1 = W1 - (lr * d_mse_w1);
        B1 = B1 - (lr * d_mse_b1);
        W2 = W2 - (lr * d_mse_w2);
        B2 = B2 - (lr * d_mse_b2);
    }

    public:
        Mlp(std::size_t epochs, double lr, std::size_t input_size,
            std::size_t hidden_size, std::size_t output_size) :
            epochs(epochs), lr(lr), input_size(input_size),
            hidden_size(hidden_size), output_size(output_size),
            X(4, 2, {0, 0, 0, 1, 1, 0, 1, 1}),
            Y(4, 1, {0, 1, 1, 0}), W1(input_size, hidden_size),
            W2(hidden_size, output_size), B1(1, hidden_size),
            B2(1, output_size), Z1(4, 2), Z2(4, 1), A(4, 2),
            y(4, 1)
        {
            Random<double> r(-1, 1);
            r.fillMatrixWithRand(W1);
            r.fillMatrixWithRand(W2);
            B1(0, 0) = 0;
            B1(0, 1) = 0;
            B2(0, 0) = 0;
        }

        void train()
        {
            for(std::size_t e = 0; e < epochs; e++)
            {
                forward();
                backward();
            }
        }

        void predict()
        {
            forward().print();
        }
};
