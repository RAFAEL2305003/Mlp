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

        Matrix<double> broadcast(const Matrix<double>& Z, const Matrix<double>& B)
        {
            auto [n_rows, n_neurons] = B.shape();
            auto [rows, cols] = Z.shape();
            Matrix b = Matrix<double>(rows, cols);

            for(size_t i = 0; i < rows; i++)
            {
                for(size_t j = 0; j < n_neurons; j++)
                {
                    b(i, j) = B(0, j);
                }
            }

            return b;
        }

        Matrix<double> forward()
        {
            // forward from input to hidden
            Z1 = hadamard(X, W1);
            Matrix b1 = broadcast(Z1, B1);
            Z1 = Z1 + b1;
            A = sigmoid(Z1);

            // forward from hidden to output
            Z2 = hadamard(A, W2);
            Matrix b2 = broadcast(Z2, B2);
            Z2 = Z2 + b2;
            y = sigmoid(Z2);

            return y;
        }

        void backward()
        {
            // backward from output to hidden
            Matrix d_mse_y = mse_derivative(Y, y);
            Matrix d_y_z2 = sigmoid_derivative(y);
            Matrix delta = d_mse_y * d_y_z2;
            Matrix d_z2_w2 = A;
            Matrix d_mse_w2 = hadamard(d_z2_w2.transpose(), delta);
            auto [rows, cols] = Z2.shape();
            Matrix d_z2_b2 = Matrix<double>(rows, cols, {1, 1, 1, 1});
            Matrix d_mse_b2 = hadamard(d_z2_b2.transpose(), delta);

            // backward from hidden to input
            Matrix d_z2_a = W2;
            Matrix d_a_z1 = sigmoid_derivative(A);
            Matrix delta_h = hadamard(delta, d_z2_a.transpose());
            delta_h = delta_h * d_a_z1;
            Matrix d_z1_w1 = X;
            Matrix d_mse_w1 = hadamard(d_z1_w1.transpose(), delta_h);
            auto [rows_, cols_] = Z1.shape();
            Matrix d_z1_b1 = Matrix<double>(rows_, cols_, {1, 1, 1, 1, 1, 1, 1, 1});
            Matrix d_mse_b1 = hadamard(d_z1_b1.transpose(), delta_h);

            std::vector<double> row(cols_);
            for(size_t j = 0; j < cols_; j++)
            {
                row[j] = d_mse_b1(0, j);
            }
            auto [b1_rows, b1_cols] = B1.shape();
            d_mse_b1 = Matrix<double>(b1_rows, b1_cols, row);

            // updating the weights
            update_weights(d_mse_w1, d_mse_b1, d_mse_w2, d_mse_b2);
        }
    
    void update_weights(const Matrix<double>& d_mse_w1, const Matrix<double>& d_mse_b1, const Matrix<double>& d_mse_w2, const Matrix<double>& d_mse_b2)
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
