# Inside the Backward Pass
### Building a 2–2–1 Neural Network from Scratch

## Project Overview

This project explores how neural networks learn by implementing a small neural network library from scratch using **NumPy and Matplotlib**, without relying on PyTorch, TensorFlow, or automatic differentiation libraries.

The main objective is to understand how gradients propagate through neural network layers, implement backpropagation using the chain rule, and verify its correctness using numerical gradient checking.

The project also investigates whether a small **2–2–1 neural network** can learn nonlinear classification patterns that a linear model cannot represent.

This project was completed as part of the **Calculus for AI & ML internship training capstone**.

## Project Phases

### Phase 1: Handwritten Calculations

Manually trace a neural network consisting of:

- 2 input neurons
- 2 hidden neurons with sigmoid activation
- 1 output neuron
- Mean Squared Error (MSE) loss

Calculate the forward-pass values and backpropagation gradients using the chain rule.

The handwritten calculations are included in the project report.

### Phase 2: Neural Network Implementation

Implement the following components from scratch:

| Component | Description |
|---|---|
| Linear | Computes weighted sums and gradients for weights, biases, and inputs |
| Sigmoid | Sigmoid activation and derivative |
| ReLU | Rectified Linear Unit activation and derivative |
| Tanh | Hyperbolic tangent activation and derivative |
| MSELoss | Mean Squared Error loss and gradient |
| BCELoss | Binary Cross-Entropy loss and gradient |
| Network | Manages forward and backward propagation through layers |

The implementation is compared against the fixed numerical example from Phase 1.

### Phase 3: Numerical Gradient Checking

Validate the manually implemented backpropagation gradients using central finite differences.

For each weight and bias, compare the analytical gradient from backpropagation with a numerical approximation.

**Results:**

- Parameters checked: 9
- Required relative error: below `1e-6`
- Maximum observed relative error: `3.14e-11`
- Result: **9/9 gradients passed**

This verification was performed using the original 2–2–1 network with MSE loss.

### Phase 4: XOR Classification

Train the 2–2–1 neural network on the XOR dataset and compare its performance with a linear classifier.

Both models use binary cross-entropy loss and full-batch gradient descent.

**Training configuration:**

- Dataset: 4 XOR combinations
- Learning rate: 1.0
- Epochs: 3,000
- Hidden activation: Sigmoid
- Output activation: Sigmoid

**Results:**

| Model | Training Accuracy | Final BCE Loss |
|---|---|---|
| Linear classifier | 50% | 0.693147 |
| 2–2–1 neural network | 100% | 0.006915 |

The neural network correctly classified all four XOR examples, while the trained linear baseline achieved 50% accuracy.

This experiment demonstrates the importance of nonlinear activation functions in learning patterns that cannot be perfectly separated by a single straight line.

*Accuracy is measured on the four training examples, not on an independent test set.*

### Phase 5: Two-Moons Classification (Optional Extension)

Test the same 2–2–1 network architecture on a more challenging nonlinear dataset generated using NumPy.

**Experiment configuration:**

- Total samples: 200
- Training samples: 160
- Testing samples: 40
- Gaussian noise standard deviation: 0.08
- Learning rate: 1.0
- Epochs: 3,000
- Architecture: 2–2–1

**Results:**

| Metric | Result |
|---|---|
| Training accuracy | 90% |
| Testing accuracy | 85% |
| Training BCE loss | 0.235785 |
| Testing BCE loss | 0.350437 |

The network learned a useful decision boundary but did not closely follow the curved shapes of the two moons.

This suggests potential limitations in the representational capacity of a network with only two hidden neurons. However, optimization, initialization, and dataset noise may also influence the results.

## Technologies Used

- **Python** — Implementation language
- **NumPy** — Matrix operations, dataset generation, and neural network calculations
- **Matplotlib** — Visualizations and training graphs
- **Google Colab** — Development and execution environment

No machine learning frameworks or automatic differentiation libraries were used.

## How to Run

1. Open the `.ipynb` notebook from this repository in Google Colab.
2. Ensure the runtime is set to Python 3. A CPU runtime is sufficient.
3. Run all notebook cells sequentially using **Runtime → Run all**.
4. Review the printed forward-pass values, backward-pass gradients, and gradient-check results.
5. Examine the XOR and two-moons classification results and visualizations.

The notebook uses NumPy random generators with fixed seeds for reproducibility.

### Requirements

```text
numpy
matplotlib
```

Both libraries are normally available in Google Colab.

For local execution, install them using:

```bash
pip install numpy matplotlib
```

## Visualizations

The notebook includes:

- XOR dataset visualization
- XOR training loss curve
- Learned XOR decision boundary
- Linear classifier versus neural network comparison
- Two-moons dataset visualization
- Two-moons training loss
- Two-moons classification decision boundary

These visualizations illustrate the effect of nonlinear activation functions and the behavior of a small neural network during training.

## Key Takeaways

1. Backpropagation applies the chain rule backward through the network to compute gradients.
2. Numerical gradient checking provides an independent way to verify analytical gradients.
3. Gradient descent updates the network's parameters to reduce prediction error.
4. Nonlinear activation functions allow neural networks to learn patterns that a single linear classifier cannot perfectly represent.
5. A 2–2–1 network can solve the four-point XOR training problem, but its performance on more complex datasets may be limited.

## Project Files

The repository contains:

- **Jupyter Notebook (`.ipynb`)** — Neural network implementation, gradient checking, training experiments, and visualizations.
- **Project Report (`.pdf`)** — Explanation of the implementation, experimental results, conclusions, and handwritten forward/backward calculations.
  
## Conclusion

This project demonstrates the mathematical foundations of neural network learning through a fully manual implementation.

By combining handwritten calculations, custom backpropagation, numerical gradient checking, and nonlinear classification experiments, the project provides a practical understanding of how neural networks compute gradients and learn from data.

The focus is not on building a large or highly optimized model, but on understanding and verifying the mechanisms that make neural network training possible.
