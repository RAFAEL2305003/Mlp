#include "../include/mlp.h"

int main()
{
    Mlp mlp(10'000, 0.1, 2, 8, 1);
    mlp.train();
    mlp.predict();
    return 0;
}
