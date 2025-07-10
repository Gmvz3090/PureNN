# PureNN

![Version](https://img.shields.io/badge/version-1.2.2-blue.svg)
![C++](https://img.shields.io/badge/C++-11%2B-brightgreen.svg)
![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20windows%20%7C%20macOS-lightgrey.svg)
![Dependencies](https://img.shields.io/badge/dependencies-none-orange.svg)
![Header Only](https://img.shields.io/badge/header--only-yes-yellow.svg)

A lightweight, header-only neural network library implemented in pure C++ from scratch.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Structures](#core-structures)
  - [struct Neuron](#struct-neuron)
  - [struct Layer](#struct-layer)
  - [struct OutputLayer](#struct-outputlayer)
  - [struct DataPoint](#struct-datapoint)
  - [struct NeuralNetwork](#struct-neuralnetwork)
- [API Reference](#api-reference)
  - [Training Methods](#training-methods)
  - [Prediction Methods](#prediction-methods)
  - [Evaluation Methods](#evaluation-methods)
  - [Utility Methods](#utility-methods)
- [Usage Examples](#usage-examples)
  - [Binary Classification with Backpropagation](#binary-classification-with-backpropagation)
  - [Multi-class Classification with SoftMax](#multi-class-classification-with-softmax)
  - [Deep Network with Multiple Activation Functions](#deep-network-with-multiple-activation-functions)
  - [Regression with Linear Output](#regression-with-linear-output)
- [Training Pipeline](#training-pipeline)
- [Network Architecture](#network-architecture)
- [Activation Functions](#activation-functions)
  - [Available Activation Functions](#available-activation-functions)
  - [Type-Safe Activation Selection](#type-safe-activation-selection)
  - [Activation Function Selection Guide](#activation-function-selection-guide)
  - [Performance Considerations](#performance-considerations)
  - [Common Activation Patterns](#common-activation-patterns)
- [Backpropagation vs Finite Differences](#backpropagation-vs-finite-differences)
- [Performance Optimization](#performance-optimization)
- [Uninstall](#uninstall)
- [Files Structure](#files-structure)
- [Future Improvements](#future-improvements)

## Features

- **Header-only library**: No compilation required, just include and use
- **Pure C++**: Zero external dependencies, uses only standard library
- **Multiple batch modes**: Full-batch and mini-batch gradient descent
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
    
    // Create training data (binary classification)
    std::vector<DataPoint> data = {
        {0.1, 0.2, false},  // Convenience constructor for binary data
        {0.3, 0.1, false},
        {0.8, 0.9, true},
        {0.7, 0.8, true}
    };
    
    // Create network: 2 inputs, 8 hidden neurons, 2 outputs
    // With backpropagation enabled (default)
    NeuralNetwork nn({2, 8, 2}, data, 0.1, 2, {"ReLU", "Sigmoid"});
    nn.randomInit();
    nn.trainnetworkbatching(100);
    
    // Make predictions
    bool result = nn.classifyBinary({0.75, 0.85});
    std::cout << "Binary result: " << (result ? "true" : "false") << std::endl;
    
    int class_result = nn.classify({0.75, 0.85});
    std::cout << "Class prediction: " << class_result << std::endl;
    
    return 0;
}
```

## Core Structures

### `struct Neuron`

Represents a single neuron with value, bias, and gradient for backpropagation.

**Members:**
- `double value` - Current activation value of the neuron
- `double bias` - Bias term added to weighted sum
- `double gradient` - Gradient for backpropagation (new in v2.0)

---

### `struct Layer`

Represents a layer of neurons with weighted connections and type-safe activation functions.

**Members:**
- `std::vector<Neuron> Neurons` - Collection of neurons in this layer
- `std::vector<std::vector<double>> weights` - Weight matrix [input][output]
- `std::vector<std::vector<double>> weightGradients` - Weight gradients for backpropagation
- `int outNodes` - Number of output neurons
- `int inNodes` - Number of input connections
- `ActivationType activationType` - Type-safe activation function enum

**Constructors:**
- `Layer()` - Default constructor, creates empty layer with ReLU activation
- `Layer(int in, int out, ActivationType activation = ActivationType::ReLU)` - Type-safe constructor
- `Layer(int in, int out, const std::string& activation = "ReLU")` - String-based constructor (backward compatible)

**Methods:**
- `void setup(int in, int out, ActivationType activation = ActivationType::ReLU)` - Configure layer with type safety
- `void setup(int in, int out, const std::string& activation = "ReLU")` - String-based setup
- `void CalcLayer(const Layer& PrevLayer)` - Forward propagation with the layer's activation function
- `void Activate()` - Apply the layer's activation function to all neurons
- `double activationDerivative(double x)` - Compute activation derivative for backpropagation

---

### `struct OutputLayer`

Specialized output layer inheriting from Layer with loss computation capabilities.

**Inherits:** All Layer functionality plus:

**Methods:**
- `double NodeCost(double expected, double actual)` - Computes squared error for single node
- `double Loss(const std::vector<double>& expected)` - Computes total loss for layer
- `void calculateOutputGradients(const std::vector<double>& expected)` - Compute output gradients for backpropagation

---

### `struct DataPoint`

Container for training data with input features and expected outputs.

**Members:**
- `std::vector<double> inputs` - Input feature vector
- `std::vector<double> expected` - Expected output vector

**Constructors:**
- `DataPoint(const std::vector<double>& in, const std::vector<double>& exp)` - Standard constructor
- `DataPoint(double in1, double in2, bool classification)` - Convenience constructor for binary classification

---

### `struct NeuralNetwork`

Main neural network class with efficient backpropagation support.

**Members:**
- `Layer input` - Input layer (with Linear activation)
- `std::vector<Layer> hidden` - Hidden layers
- `OutputLayer output` - Output layer (inherits from Layer)
- `std::vector<DataPoint> train` - Training dataset
- `double learn` - Learning rate
- `int batchsize` - Batch size for mini-batch training
- `bool useBackprop` - Enable/disable backpropagation

**Constructor:**
- `NeuralNetwork(const std::vector<int>& structure, const std::vector<DataPoint>& training_data, double learnrate, int batch_size, const std::vector<std::string>& activations = {}, bool use_backprop = true)`
  - `structure`: Network topology (e.g., {2, 8, 2} = 2 inputs, 8 hidden, 2 outputs)
  - `training_data`: Training dataset
  - `learnrate`: Learning rate for gradient descent
  - `batch_size`: Size of mini-batches
  - `activations`: Activation functions for each layer (optional)
  - `use_backprop`: Enable backpropagation (default: true)

## API Reference

### Training Methods

- `void trainnetwork(int epochs)` - Full-batch training with backpropagation
- `void trainnetworkbatching(int epochs)` - Mini-batch training with backpropagation
- `void randomInit(double min_val = -1.0, double max_val = 1.0)` - Initialize weights and biases randomly
- `void setBackpropagation(bool enable)` - Enable/disable backpropagation during runtime

### Prediction Methods

- `std::vector<double> predict(const DataPoint& dp)` - Get raw output values
- `std::vector<double> predict(const std::vector<double>& inputs)` - Predict from raw input vector
- `int classify(const DataPoint& dp)` - Get predicted class index (argmax)
- `int classify(const std::vector<double>& inputs)` - Classify from raw input vector
- `bool classifyBinary(const std::vector<double>& inputs)` - Binary classification convenience method

### Evaluation Methods

- `double pointloss(const DataPoint& dp)` - Compute loss for single data point
- `double getLoss()` - Compute average loss over entire training set
- `double getBatchLoss(int start_idx, int batch_size)` - Compute loss for batch

### Utility Methods

- `void reset()` - Reset all neuron values and gradients to zero
- `void setInputs(const DataPoint& dp)` - Set input layer values from data point
- `void forwardPass(const DataPoint& dp)` - Perform forward propagation
- `void backwardPass(const DataPoint& dp)` - Perform backpropagation (compute gradients)
- `void updateWeights()` - Update weights using computed gradients

## Usage Examples

### Binary Classification with Backpropagation

```cpp
std::vector<DataPoint> data = {
    {0.1, 0.2, false},  // Automatically converts to {0.1, 0.2} -> {0.0, 1.0}
    {0.3, 0.1, false},
    {0.8, 0.9, true},   // Automatically converts to {0.8, 0.9} -> {1.0, 0.0}
    {0.7, 0.8, true}
};

// Network with backpropagation enabled (default)
NeuralNetwork nn({2, 8, 2}, data, 0.1, 2, {"ReLU", "Sigmoid"});
nn.randomInit(-0.5, 0.5);  // Smaller initialization range
nn.trainnetworkbatching(100);

bool result = nn.classifyBinary({0.5, 0.5});
std::cout << "Binary prediction: " << (result ? "positive" : "negative") << std::endl;
```

### Multi-class Classification with SoftMax

```cpp
std::vector<DataPoint> data = {
    {{0.1, 0.2}, {1.0, 0.0, 0.0}},  // Class 0
    {{0.5, 0.5}, {0.0, 1.0, 0.0}},  // Class 1
    {{0.8, 0.9}, {0.0, 0.0, 1.0}}   // Class 2
};

// Network with SoftMax output for multi-class classification
NeuralNetwork nn({2, 16, 8, 3}, data, 0.01, 1, {"ReLU", "ReLU", "SoftMax"});
nn.randomInit(-0.3, 0.3);
nn.trainnetworkbatching(200);

int predicted_class = nn.classify({0.6, 0.7});
auto probabilities = nn.predict({0.6, 0.7});
```

### Deep Network with Multiple Activation Functions

```cpp
std::vector<DataPoint> data = {
    {{1.0, 2.0, 3.0}, {0.5, 1.0}},
    {{2.0, 3.0, 4.0}, {1.0, 0.5}},
    {{3.0, 4.0, 5.0}, {1.5, 0.8}}
};

// Deep network with modern activation functions
NeuralNetwork nn({3, 32, 16, 8, 2}, data, 0.001, 1, {"SiLU", "ReLU", "SiLU", "Sigmoid"});
nn.randomInit(-0.2, 0.2);
nn.trainnetworkbatching(300);

auto result = nn.predict({2.5, 3.5, 4.5});
```

### Regression with Linear Output

```cpp
std::vector<DataPoint> data = {
    {{1.0, 2.0}, {3.5}},    // Single output regression
    {{2.0, 3.0}, {5.2}},
    {{3.0, 4.0}, {7.1}}
};

// Regression network with Linear output activation
NeuralNetwork nn({2, 10, 5, 1}, data, 0.001, 1, {"ReLU", "ReLU", "Linear"});
nn.randomInit(-0.1, 0.1);
nn.trainnetworkbatching(500);

auto result = nn.predict({2.5, 3.5});
double predicted_value = result[0];
```

## Training Pipeline

1. **Data Preparation**: Create `DataPoint` objects with normalized inputs
2. **Network Design**: Choose architecture and activation functions
3. **Initialization**: Use `randomInit()` with appropriate range
4. **Training**: Use `trainnetworkbatching()` with backpropagation
5. **Evaluation**: Test predictions and monitor loss convergence

See the comprehensive [Pipeline Guide](pipeline-guide) for detailed workflow.

## Network Architecture

- **Input Layer**: Receives input features, uses Linear activation (no transformation)
- **Hidden Layers**: Configurable activation functions (default: ReLU)
- **Output Layer**: Inherits from Layer, configurable activation (default: Sigmoid)

The inheritance structure ensures code reuse while maintaining specialized functionality.

## Activation Functions

### Available Activation Functions

#### ReLU (Rectified Linear Unit)
- **Formula**: `f(x) = max(0, x)`
- **Derivative**: `f'(x) = x > 0 ? 1 : 0`
- **Best for**: Hidden layers in deep networks
- **Usage**: `ActivationType::ReLU` or `"ReLU"`

#### Sigmoid
- **Formula**: `f(x) = 1/(1 + e^(-x))`
- **Derivative**: `f'(x) = f(x)(1 - f(x))`
- **Best for**: Binary classification output layers
- **Usage**: `ActivationType::Sigmoid` or `"Sigmoid"`

#### SoftMax
- **Formula**: `f(x_i) = e^(x_i) / Σ(e^(x_j))` for all j
- **Best for**: Multi-class classification output layers
- **Usage**: `ActivationType::SoftMax` or `"SoftMax"`

#### TanH (Hyperbolic Tangent)
- **Formula**: `f(x) = tanh(x)`
- **Derivative**: `f'(x) = 1 - f(x)²`
- **Best for**: Hidden layers, zero-centered outputs
- **Usage**: `ActivationType::TanH` or `"TanH"`

#### SoftPlus
- **Formula**: `f(x) = log(1 + e^x)`
- **Derivative**: `f'(x) = 1/(1 + e^(-x))`
- **Best for**: Smooth alternative to ReLU
- **Usage**: `ActivationType::SoftPlus` or `"SoftPlus"`

#### SiLU (Sigmoid Linear Unit / Swish)
- **Formula**: `f(x) = x / (1 + e^(-x))`
- **Derivative**: `f'(x) = σ(x)(1 + x(1 - σ(x)))`
- **Best for**: Modern hidden layers, replacement for ReLU
- **Usage**: `ActivationType::SiLU` or `"SiLU"`

#### Linear
- **Formula**: `f(x) = x`
- **Derivative**: `f'(x) = 1`
- **Best for**: Regression output layers, input layers
- **Usage**: `ActivationType::Linear` or `"Linear"`

### Type-Safe Activation Selection

```cpp
// Type-safe enum usage (recommended)
Layer hidden_layer(10, 5, ActivationType::ReLU);

// String-based usage (backward compatible)
Layer hidden_layer(10, 5, "ReLU");

// Compile-time error prevention
Layer layer(10, 5, ActivationType::InvalidType);  // Compilation error!
```

### Activation Function Selection Guide

#### For Hidden Layers:
- **ReLU**: Default choice, fast and effective
- **SiLU**: Modern alternative with better gradient flow
- **TanH**: Zero-centered outputs, good for certain architectures

#### For Output Layers:
- **Sigmoid**: Binary classification
- **SoftMax**: Multi-class classification
- **Linear**: Regression tasks
- **TanH**: Bounded regression (-1, 1)

### Performance Considerations

- **Training Speed**: Linear > ReLU > TanH ≈ Sigmoid > SoftPlus > SiLU > SoftMax
- **Gradient Flow**: SiLU > ReLU > SoftPlus > TanH > Sigmoid
- **Numerical Stability**: All functions use numerically stable implementations

### Common Activation Patterns

```cpp
// Modern Classification
{"SiLU", "ReLU", "SoftMax"}

// Traditional Classification  
{"ReLU", "ReLU", "Sigmoid"}

// Regression
{"ReLU", "ReLU", "Linear"}
{"TanH", "TanH", "TanH"}

// Hybrid Approaches
{"SiLU", "ReLU", "Sigmoid"}
```

## Backpropagation vs Finite Differences

### Backpropagation (Default)
- **Speed**: 100-1000x faster for larger networks
- **Accuracy**: Exact analytical gradients
- **Scalability**: Handles thousands of parameters efficiently
- **Memory**: Efficient gradient computation

### Finite Differences (Legacy)
- **Speed**: Slower, O(n) gradient computations per parameter
- **Accuracy**: Numerical approximation with potential errors
- **Use cases**: Debugging, very small networks, educational purposes

### Switching Between Methods

```cpp
// Enable backpropagation (default)
NeuralNetwork nn(structure, data, lr, bs, activations, true);

// Use finite differences
NeuralNetwork nn(structure, data, lr, bs, activations, false);

// Change during runtime
nn.setBackpropagation(false);
```

## Performance Optimization

1. **Use smaller initialization ranges**: `nn.randomInit(-0.3, 0.3)` often works better than default
2. **Choose appropriate learning rates**: Start with 0.01-0.1, adjust based on convergence
3. **Batch size selection**: Powers of 2 (16, 32, 64) often work well
4. **Activation function selection**: SiLU often outperforms ReLU in hidden layers
5. **Architecture design**: Start simple, add complexity only if needed

## Uninstall

```bash
make uninstall
```

## Files Structure

```
PureNN/
├── purenn          # Enhanced header-only library v2.0
├── Makefile        # Installation script
├── README.md       # This updated documentation
├── example.cpp     # Usage example with backpropagation
└── .gitignore      # Git ignore rules
```

## Future Improvements

1. **Model Serialization** - Save/load trained networks
2. **Advanced Optimizers** - Adam, RMSprop, momentum
3. **Regularization** - L1/L2 regularization, dropout
4. **Batch Normalization** - Improved training stability
5. **Multi-Threading** - Parallel batch processing
6. **GPU Acceleration** - CUDA support for large networks
7. **Validation Split** - Automatic train/validation separation
8. **Early Stopping** - Prevent overfitting
9. **Learning Rate Scheduling** - Adaptive learning rates

---
