# PureNN

A lightweight, header-only neural network library implemented in pure C++ from scratch. PureNN provides a clean, educational implementation of feedforward neural networks with numerical gradient computation.

## Features

- **Header-only library**: No compilation required, just include and use
- **Pure C++**: Zero external dependencies, uses only standard library
- **Multiple training modes**: Full-batch and mini-batch gradient descent
- **Multiple output formats**: Classification, regression, and raw predictions

## Installation

```bash
git clone https://github.com/Gmvz3090/PureNN
cd PureNN
make install
```

## Quick Start

```cpp
#include <purenn>
#include <iostream>

int main() {
    using namespace purenn;
    
    // Create training data
    std::vector<DataPoint> data = {
        {{0.1, 0.2}, {0.0, 1.0}},  // Binary classification
        {{0.8, 0.9}, {1.0, 0.0}}
    };
    
    // Create network: 2 inputs, 3 hidden neurons, 2 outputs
    NeuralNetwork nn({2, 3, 2}, data, 0.2, 4);
    nn.randomInit();
    nn.trainnetworkbatching(100);
    
    // Make prediction
    int result = nn.classify({0.75, 0.85});
    std::cout << "Predicted class: " << result << std::endl;
    
    return 0;
}
```

## API Reference

### Core Structures

#### `struct Neuron`

Represents a single neuron with value and bias.

**Members:**
- `double value` - Current activation value of the neuron
- `double bias` - Bias term added to weighted sum

**Methods:**
- `void activateSigm()` - Applies sigmoid activation function: `σ(x) = 1/(1 + e^(-x))`

---

#### `struct Layer`

Represents a layer of neurons with weighted connections.

**Members:**
- `std::vector<Neuron> Neurons` - Collection of neurons in this layer
- `std::vector<std::vector<double>> weights` - Weight matrix [input][output]
- `int outNodes` - Number of output neurons
- `int inNodes` - Number of input connections

**Constructors:**
- `Layer()` - Default constructor, creates empty layer
- `Layer(int in, int out)` - Creates layer with specified input/output dimensions

**Methods:**
- `void setup(int in, int out)` - Configure layer dimensions after construction
- `void CalcLayer(const Layer& PrevLayer)` - Forward propagation with sigmoid activation

---

#### `struct OutputLayer`

Specialized output layer with softmax activation for classification tasks.

**Members:**
- `std::vector<Neuron> Neurons` - Collection of output neurons
- `std::vector<std::vector<double>> weights` - Weight matrix [input][output]
- `int outNodes` - Number of output neurons
- `int inNodes` - Number of input connections

**Constructors:**
- `OutputLayer()` - Default constructor, creates empty output layer
- `OutputLayer(int in, int out)` - Creates output layer with specified dimensions

**Methods:**
- `void setup(int in, int out)` - Configure layer dimensions after construction
- `void CalcLayer(const Layer& PrevLayer)` - Forward propagation with softmax activation
- `double NodeCost(double expected, double actual)` - Computes squared error for single node
- `double Loss(const std::vector<double>& expected)` - Computes total loss for layer

---

#### `struct DataPoint`

Container for training data with input features and expected outputs.

**Members:**
- `std::vector<double> inputs` - Input feature vector
- `std::vector<double> expected` - Expected output vector

**Constructor:**
- `DataPoint(const std::vector<double>& in, const std::vector<double>& exp)` - Creates data point with input features and expected output vector

---

#### `struct NeuralNetwork`

Main neural network class coordinating all layers and training.

**Members:**
- `Layer input` - Input layer
- `std::vector<Layer> hidden` - Hidden layers
- `OutputLayer output` - Output layer
- `std::vector<DataPoint> train` - Training dataset
- `double learn` - Learning rate
- `int batchsize` - Batch size for mini-batch training

**Constructor:**
- `NeuralNetwork(const std::vector<int>& structure, const std::vector<DataPoint>& training_data, double learnrate, int batch_size)`
  - `structure`: Network topology (e.g., {2, 3, 2} = 2 inputs, 3 hidden, 2 outputs)
  - `training_data`: Training dataset
  - `learnrate`: Learning rate for gradient descent
  - `batch_size`: Size of mini-batches

### Training Methods

- `void trainnetwork(int epochs)` - Full-batch gradient descent training
- `void trainnetworkbatching(int epochs)` - Mini-batch gradient descent training
- `void randomInit(double min_val = -1.0, double max_val = 1.0)` - Initialize weights and biases randomly

### Prediction Methods

- `std::vector<double> predict(const DataPoint& dp)` - Get raw output probabilities
- `std::vector<double> predict(const std::vector<double>& inputs)` - Predict from raw input vector
- `int classify(const DataPoint& dp)` - Get predicted class index
- `int classify(const std::vector<double>& inputs)` - Classify from raw input vector

### Evaluation Methods

- `double pointloss(const DataPoint& dp)` - Compute loss for single data point
- `double getLoss()` - Compute average loss over entire training set
- `double getBatchLoss(int start_idx, int batch_size)` - Compute loss for batch

### Utility Methods

- `void reset()` - Reset all neuron values to zero
- `void setInputs(const DataPoint& dp)` - Set input layer values from data point

### Internal Methods

- `double getGradient(double& weight)` - Compute gradient using finite differences
- `double getBatchGradient(double& weight, int start_idx, int batch_size)` - Batch gradient computation
- `void adjust()` - Update all weights and biases (full-batch)
- `void adjustBatch(int start_idx, int batch_size)` - Update weights and biases (mini-batch)

## Usage Examples

### Binary Classification

```cpp
std::vector<DataPoint> data = {
    {{0.1, 0.2}, {0.0, 1.0}},  // Class 1
    {{0.8, 0.9}, {1.0, 0.0}}   // Class 0
};

NeuralNetwork nn({2, 3, 2}, data, 0.2, 4);
nn.randomInit();
nn.trainnetworkbatching(50);

int result = nn.classify({0.5, 0.5});
```

### Multi-class Classification

```cpp
std::vector<DataPoint> data = {
    {{0.1, 0.2}, {1.0, 0.0, 0.0}},  // Class 0
    {{0.5, 0.5}, {0.0, 1.0, 0.0}},  // Class 1
    {{0.8, 0.9}, {0.0, 0.0, 1.0}}   // Class 2
};

NeuralNetwork nn({2, 4, 3}, data, 0.1, 2);
nn.randomInit();
nn.trainnetworkbatching(100);

int predicted_class = nn.classify({0.6, 0.7});
```

### Regression

```cpp
std::vector<DataPoint> data = {
    {{1.0, 2.0}, {0.5}},
    {{2.0, 3.0}, {1.0}},
    {{3.0, 4.0}, {1.5}}
};

NeuralNetwork nn({2, 4, 1}, data, 0.01, 1);
nn.randomInit();
nn.trainnetworkbatching(200);

auto result = nn.predict({2.5, 3.5});
```

## Network Architecture

- **Input Layer**: Receives input features, no activation function
- **Hidden Layers**: Use sigmoid activation function
- **Output Layer**: Uses softmax activation for multi-class classification

## Algorithm Details

- **Forward Propagation**: Standard weighted sum → activation pattern
- **Gradient Computation**: Numerical gradients using finite differences (h = 0.001)
- **Optimization**: Gradient descent with configurable learning rate
- **Loss Function**: Squared error loss
- **Batch Training**: Supports both full-batch and mini-batch training

## Compilation

```bash
g++ -std=c++11 your_program.cpp -o your_program
```

## Uninstall

```bash
make uninstall
```

## Files Structure

```
PureNN/
├── purenn          # Header-only library
├── Makefile        # Installation script
├── README.md       # This documentation
├── example.cpp     # Usage example
└── .gitignore      # Git ignore rules
```

# Future improvements

1. Adding Backpropagation
2. Additional Activation Functions
3. Saving Models
4. Advanced Optimizers
5. Multi-Threading
6. Overfitting Prevention
7. Progress Tracking (Visual Changes)