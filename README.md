# Mlp

Small C++ framework for training feed-forward (MLP) and 1D convolutional
+ MLP networks on the CPU. Models are described in JSON and trained on a
CSV.

For the parallel (CPU/GPU SYCL) version, switch to branch `sycl`.

Dataset used:
https://drive.google.com/file/d/1eCARI1taMWfzpsOZW6amGl8m5cqXFW-V/view?usp=sharing

## Build

Requires a C++17 compiler.

```bash
g++ -O3 -std=c++17 src/nn.cpp -o nn
```

## Usage

```bash
./nn models/mlp_01.json
```

The program loads the JSON, reads the CSV referenced in `dataset`,
normalizes the first `input_size` columns, trains for `epochs` epochs on
mini-batches of `batch_size`, and at the end prints accuracy, precision
and F1-score.

## JSON format

Required fields:

| Field           | Type     | Description                                                                |
| --------------- | -------- | -------------------------------------------------------------------------- |
| `learning_rate` | float    | SGD learning rate.                                                         |
| `epochs`        | int      | Number of epochs.                                                          |
| `batch_size`    | int      | Mini-batch size.                                                           |
| `input_size`    | int      | Number of input columns (first `input_size` columns of the CSV).           |
| `dataset`       | string   | CSV path. Columns after `input_size` are treated as the label.             |
| `layers`        | int[]    | Size of each dense layer, including input and output.                      |
| `activations`   | string[] | Activations of the dense layers. Length = `len(layers) - 1`.               |
| `loss`          | string   | Loss function.                                                             |
| `conv`          | array    | Optional, list of conv1d layers.                                           |
| `pool`          | array    | Optional, list of pooling layers (same length as `conv`).                  |

## What is available

### Activations (field `activations` and `conv[].activation`)

- `relu`
- `sigmoid`
- `softmax`

### Loss (field `loss`)

- `mse` — mean squared error
- `bce` — binary cross-entropy
- `ce`  — categorical cross-entropy

### 1D convolution (field `conv[].type`)

- `valid` — no padding
- `same`  — output has the same size as the input
- `full`  — full padding

Each `conv` item has: `filters`, `kernel_size`, `type`, `stride`,
`activation`.

### Pooling (field `pool[].type`)

- `max`
- `avg`

Each `pool` item has: `type`, `pool_size`, `stride`.

## Size formulas (to make `layers[0]` match the conv block output)

For each layer `i`:

```
conv_out_i = (input_i - kernel_size + 2*padding) / stride + 1
pool_out_i = (conv_out_i - pool_size) / pool_stride + 1
input_{i+1} = pool_out_i           // the next conv sees one channel per filter
```

The flatten passed into the first dense layer is
`last_filters * last_pool_out`, and this value must be exactly
`layers[0]` (otherwise the constructor trips an assert).

The size of `activations` must be `len(layers) - 1`. The last entry of
`activations` is the output activation.

## Architecture examples

### Plain MLP

```json
{
  "learning_rate": 0.01,
  "epochs": 100,
  "batch_size": 4096,
  "input_size": 16,
  "dataset": "can_ids.csv",
  "conv": [],
  "pool": [],
  "layers": [16, 32, 16, 8, 5],
  "activations": ["relu", "relu", "relu", "softmax"],
  "loss": "ce"
}
```

16 input features, three ReLU hidden layers (32, 16, 8) and a softmax
over 5 classes, trained with cross-entropy.

### Shallow CNN + linear classifier

```json
{
  "learning_rate": 0.01,
  "epochs": 15,
  "batch_size": 2048,
  "input_size": 16,
  "dataset": "can_ids.csv",
  "conv": [
    {"filters": 4, "kernel_size": 2, "type": "valid", "stride": 2, "activation": "relu"}
  ],
  "pool": [
    {"type": "max", "pool_size": 2, "stride": 2}
  ],
  "layers": [16, 10, 5],
  "activations": ["relu", "softmax"],
  "loss": "ce"
}
```

`conv valid` 16→8, `pool` 8→4, flatten = `4 filters * 4 = 16` = `layers[0]`.

### Deep CNN

```json
{
  "learning_rate": 0.005,
  "epochs": 20,
  "batch_size": 2048,
  "input_size": 16,
  "dataset": "can_ids.csv",
  "conv": [
    {"filters": 4, "kernel_size": 3, "type": "valid", "stride": 1, "activation": "relu"},
    {"filters": 4, "kernel_size": 3, "type": "valid", "stride": 1, "activation": "relu"},
    {"filters": 4, "kernel_size": 3, "type": "valid", "stride": 1, "activation": "relu"}
  ],
  "pool": [
    {"type": "max", "pool_size": 2, "stride": 2},
    {"type": "max", "pool_size": 2, "stride": 2},
    {"type": "max", "pool_size": 2, "stride": 2}
  ],
  "layers": [100, 32, 8, 5],
  "activations": ["relu", "relu", "softmax"],
  "loss": "ce"
}
```

### Binary classification with `sigmoid` + `bce`

```json
{
  "learning_rate": 0.01,
  "epochs": 50,
  "batch_size": 1024,
  "input_size": 16,
  "dataset": "can_ids.csv",
  "conv": [],
  "pool": [],
  "layers": [16, 16, 1],
  "activations": ["relu", "sigmoid"],
  "loss": "bce"
}
```

## Structure

- [src/nn.cpp](src/nn.cpp) — entry point and training/prediction loop.
- [include/](include/) — layers (`linear_layer`, `conv1d_layer`,
  `pooling_layer`, `activation_layer`, `loss_layer`), matrix utilities
  in [include/math/](include/math/), JSON and CSV parsers.
- [models/](models/) — sample JSONs.
