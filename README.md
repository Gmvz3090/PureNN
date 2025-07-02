# PureNN

![Version](https://img.shields.io/badge/version-1.21-blue.svg)
![C++](https://img.shields.io/badge/C++-11%2B-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
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
  - [Prediction Methods](#prediction-methods)
  - [Evaluation Methods](#evaluation-methods)
  - [Utility Methods](#utility-methods)
  - [Internal Methods](#internal-methods)
- [Usage Examples](#usage-examples)
  - [Binary Classification with Custom Activations](#binary-classification-with-custom-activations)
  - [Multi-class Classification with SoftMax](#multi-class-classification-with-softmax)
  - [Deep Network with Multiple Activation Functions](#deep-network-with-multiple-activation-functions)
  - [Using Default Activations](#using-default-activations)
- [Network Architecture](#network-architecture)
- [Uninstall](#uninstall)
- [Files Structure](#files-structure)
- [Activation Functions](#activation-functions)
  - [Available Activation Functions](#available-activation-functions)
  - [Activation Function Selection Guide](#activation-function-selection-guide)
  - [Performance Considerations](#performance-considerations)
  - [Common Activation Patterns](#common-activation-patterns)
- [Future Improvements](#future-improvements)

## Features

- **Header-only library**: No compilation required, just include and use
- **Pure C++**: Zero external dependencies, uses only standard library
- **Multiple activation functions**: ReLU, Sigmoid, SoftMax, TanH, SoftPlus, SiLU
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
    // With custom activation functions: ReLU for hidden, Sigmoid for output
    NeuralNetwork nn({2, 3, 2}, data, 0.2, 4, {"ReLU", "Sigmoid"});
    nn.randomInit();
    nn.trainnetworkbatching(100);
    
    // Make prediction
    int result = nn.classify({0.75, 0.85});
    std::cout << "Predicted class: " << result << std::endl;
    
    return 0;
}
```

## Core Structures

### `struct Neuron`

Represents a single neuron with value and bias.

**Members:**
- `double value` - Current activation value of the neuron
- `double bias` - Bias term added to weighted sum

---

### `struct Layer`

Represents a layer of neurons with weighted connections and configurable activation function.

**Members:**
- `std::vector<Neuron> Neurons` - Collection of neurons in this layer
- `std::vector<std::vector<double>> weights` - Weight matrix [input][output]
- `int outNodes` - Number of output neurons
- `int inNodes` - Number of input connections
- `std::string activationMethod` - Activation function for this layer

**Constructors:**
- `Layer()` - Default constructor, creates empty layer with ReLU activation
- `Layer(int in, int out, const std::string& activation = "ReLU")` - Creates layer with specified dimensions and activation function

**Methods:**
- `void setup(int in, int out, const std::string& activation = "ReLU")` - Configure layer dimensions and activation after construction
- `void CalcLayer(const Layer& PrevLayer)` - Forward propagation with the layer's activation function
- `void Activate()` - Apply the layer's activation function to all neurons

---

### `struct OutputLayer`

Specialized output layer with configurable activation function, typically used for classification or regression tasks.

**Members:**
- `std::vector<Neuron> Neurons` - Collection of output neurons
- `std::vector<std::vector<double>> weights` - Weight matrix [input][output]
- `int outNodes` - Number of output neurons
- `int inNodes` - Number of input connections
- `std::string activationMethod` - Activation function for this layer

**Constructors:**
- `OutputLayer()` - Default constructor, creates empty output layer with Sigmoid activation
- `OutputLayer(int in, int out, const std::string& activation = "Sigmoid")` - Creates output layer with specified dimensions and activation

**Methods:**
- `void setup(int in, int out, const std::string& activation = "Sigmoid")` - Configure layer dimensions and activation after construction
- `void CalcLayer(const Layer& PrevLayer)` - Forward propagation with the layer's activation function
- `void Activate()` - Apply the layer's activation function to all neurons
- `double NodeCost(double expected, double actual)` - Computes squared error for single node
- `double Loss(const std::vector<double>& expected)` - Computes total loss for layer

---

### `struct DataPoint`

Container for training data with input features and expected outputs.

**Members:**
- `std::vector<double> inputs` - Input feature vector
- `std::vector<double> expected` - Expected output vector

**Constructor:**
- `DataPoint(const std::vector<double>& in, const std::vector<double>& exp)` - Creates data point with input features and expected output vector

---

### `struct NeuralNetwork`

Main neural network class coordinating all layers and training.

**Members:**
- `Layer input` - Input layer
- `std::vector<Layer> hidden` - Hidden layers
- `OutputLayer output` - Output layer
- `std::vector<DataPoint> train` - Training dataset
- `double learn` - Learning rate
- `int batchsize` - Batch size for mini-batch training

**Constructor:**
- `NeuralNetwork(const std::vector<int>& structure, const std::vector<DataPoint>& training_data, double learnrate, int batch_size, const std::vector<std::string>& activations = {})`
  - `structure`: Network topology (e.g., {2, 3, 2} = 2 inputs, 3 hidden, 2 outputs)
  - `training_data`: Training dataset
  - `learnrate`: Learning rate for gradient descent
  - `batch_size`: Size of mini-batches
  - `activations`: Activation functions for each layer (optional, defaults to ReLU for hidden, Sigmoid for output)

## API Reference

**Training Methods:**
- `void trainnetwork(int epochs)` - Full-batch gradient descent training
- `void trainnetworkbatching(int epochs)` - Mini-batch gradient descent training
- `void randomInit(double min_val = -1.0, double max_val = 1.0)` - Initialize weights and biases randomly

### Prediction Methods

- `std::vector<double> predict(const DataPoint& dp)` - Get raw output values
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

### Binary Classification with Custom Activations

```cpp
std::vector<DataPoint> data = {
    {{0.1, 0.2}, {0.0, 1.0}},  // Class 1
    {{0.8, 0.9}, {1.0, 0.0}}   // Class 0
};

// Network with ReLU hidden layer and Sigmoid output
NeuralNetwork nn({2, 3, 2}, data, 0.2, 4, {"ReLU", "Sigmoid"});
nn.randomInit();
nn.trainnetworkbatching(50);

int result = nn.classify({0.5, 0.5});
```

### Multi-class Classification with SoftMax

```cpp
std::vector<DataPoint> data = {
    {{0.1, 0.2}, {1.0, 0.0, 0.0}},  // Class 0
    {{0.5, 0.5}, {0.0, 1.0, 0.0}},  // Class 1
    {{0.8, 0.9}, {0.0, 0.0, 1.0}}   // Class 2
};

// Network with TanH hidden layer and SoftMax output for multi-class classification
NeuralNetwork nn({2, 4, 3}, data, 0.1, 2, {"TanH", "SoftMax"});
nn.randomInit();
nn.trainnetworkbatching(100);

int predicted_class = nn.classify({0.6, 0.7});
```

### Deep Network with Multiple Activation Functions

```cpp
std::vector<DataPoint> data = {
    {{1.0, 2.0, 3.0}, {0.5, 1.0}},
    {{2.0, 3.0, 4.0}, {1.0, 0.5}},
    {{3.0, 4.0, 5.0}, {1.5, 0.8}}
};

// Deep network: 3 inputs, two hidden layers (8, 4 neurons), 2 outputs
// ReLU -> SiLU -> Sigmoid activation sequence
NeuralNetwork nn({3, 8, 4, 2}, data, 0.01, 1, {"ReLU", "SiLU", "Sigmoid"});
nn.randomInit();
nn.trainnetworkbatching(200);

auto result = nn.predict({2.5, 3.5, 4.5});
```

### Using Default Activations

```cpp
// If no activations specified, defaults to ReLU for hidden layers and Sigmoid for output
NeuralNetwork nn({2, 3, 2}, data, 0.2, 4);
nn.randomInit();
nn.trainnetworkbatching(100);
```

## Network Architecture

- **Input Layer**: Receives input features, no activation function
- **Hidden Layers**: Configurable activation functions (default: ReLU)
- **Output Layer**: Configurable activation function (default: Sigmoid)

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

## Activation Functions

PureNN supports multiple activation functions that can be configured per layer to optimize performance for different types of neural network tasks.

### Available Activation Functions

#### ReLU (Rectified Linear Unit)
- **Formula**: `f(x) = max(0, x)`
- **Best for**: Hidden layers in deep networks, general-purpose activation
- **Usage**: `"ReLU"`

#### Sigmoid
- **Formula**: `f(x) = 1/(1 + e^(-x))`
- **Best for**: Binary classification output layers, gates in LSTM networks
- **Usage**: `"Sigmoid"`

#### SoftMax
- **Formula**: `f(x_i) = e^(x_i) / Σ(e^(x_j))` for all j
- **Best for**: Multi-class classification output layers
- **Usage**: `"SoftMax"`

#### TanH (Hyperbolic Tangent)
- **Formula**: `f(x) = (2/(1 + e^(-2x))) - 1`
- **Best for**: Hidden layers, especially when you need zero-centered activations
- **Usage**: `"TanH"`

#### SoftPlus
- **Formula**: `f(x) = log(1 + e^x)`
- **Best for**: Hidden layers when smooth activation is preferred over ReLU
- **Usage**: `"SoftPlus"`

#### SiLU (Sigmoid Linear Unit / Swish)
- **Formula**: `f(x) = x / (1 + e^(-x))`
- **Best for**: Hidden layers in modern deep networks, replacement for ReLU
- **Usage**: `"SiLU"`

### Activation Function Selection Guide

#### For Hidden Layers:
- **ReLU**: Default choice, fast and effective for most cases
- **SiLU**: Modern alternative to ReLU with potentially better performance
- **TanH**: When you need zero-centered outputs
- **SoftPlus**: When you need smooth, always-positive activations

#### For Output Layers:
- **Sigmoid**: Binary classification (outputs probability for positive class)
- **SoftMax**: Multi-class classification (outputs probability distribution)
- **ReLU**: Regression with non-negative outputs
- **TanH**: Regression with outputs in range (-1, 1)
- **None/Linear**: Standard regression (though not explicitly supported, achieved by not applying activation)

### Performance Considerations

- **Speed**: ReLU > Sigmoid ≈ TanH > SoftPlus > SiLU > SoftMax
- **Memory**: All functions have similar memory requirements
- **Gradient Flow**: SiLU > ReLU > SoftPlus > TanH > Sigmoid
- **Numerical Stability**: All implementations use standard library functions for stability

### Common Activation Patterns

```cpp
// Classification Networks
{"ReLU", "ReLU", "SoftMax"}     // Multi-class classification
{"ReLU", "ReLU", "Sigmoid"}     // Binary classification
{"SiLU", "SiLU", "SoftMax"}     // Modern multi-class classification

// Regression Networks
{"ReLU", "ReLU", "ReLU"}        // Non-negative regression
{"TanH", "TanH", "TanH"}        // Bounded regression
{"SiLU", "SiLU", "SoftPlus"}    // Modern regression

// Mixed Networks
{"ReLU", "SiLU", "Sigmoid"}     // Hybrid approach
{"TanH", "ReLU", "SoftMax"}     // Zero-centered to positive classification
```

## Future Improvements

1. Adding Backpropagation
2. Saving Models
3. Advanced Optimizers
4. Multi-Threading
5. Overfitting Prevention
6. Progress Tracking (Visual Changes)
7. GPU Acceleration
8. Batch Normalization

---

**PureNN v1.21** - Made with ❤️ for the C++ community