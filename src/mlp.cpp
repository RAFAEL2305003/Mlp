#include "../include/mlp.h"

int main()
{
    Mlp mlp(100'000, 0.1, 2, 2, 1);
    mlp.train();
    mlp.predict();
    return 0;
}
