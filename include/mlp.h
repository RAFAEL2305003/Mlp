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

    public:
        Mlp(std::size_t epochs, double lr, std::size_t input_size,
            std::size_t hidden_size, std::size_t output_size) :
            epochs(epochs), lr(lr), input_size(input_size),
            hidden_size(hidden_size), output_size(output_size),
            X(input_size, hidden_size, {0, 0, 0, 1, 1, 0, 1, 1}),
            Y(input_size, output_size, {0, 1, 1, 0}), W1(2, hidden_size),
            W2(output_size, hidden_size), b1(hidden_size, 1),
            b2(output_size, 1), z1(hidden_size, 1), z2(1, 1), a1(hidden_size, 1),
            a2(output_size, 1)
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
            for(size_t e = 0; e < 1; e++)
            {
                z1 = (X * W1);
            }
        }
};