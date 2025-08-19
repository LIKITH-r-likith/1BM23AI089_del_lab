import math
import matplotlib.pyplot as plt
import numpy as np

def linear(x):
    return x
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def tanh(x):
    return math.tanh(x)
def relu(x):
    return max(0, x)

def softmax(x_vals):
    exp_vals = np.exp(x_vals - np.max(x_vals))
    return exp_vals / np.sum(exp_vals)

def perceptron(input_value, weight, bias, activation_function):
    weighted_sum = input_value * weight + bias
    return activation_function(weighted_sum)

def plot_activation_functions(input_vector, softmax_vals):
    x_vals = np.linspace(-10, 10, 400)
    y_linear = [linear(x) for x in x_vals]
    
    y_sigmoid = [sigmoid(x) for x in x_vals]
    y_tanh = [tanh(x) for x in x_vals]
    y_relu = [relu(x) for x in x_vals]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_linear, label='Linear', linestyle='--')
    
    plt.plot(x_vals, y_sigmoid, label='Sigmoid', color='green')
    plt.plot(x_vals, y_tanh, label='Tanh', color='red')
    plt.plot(x_vals, y_relu, label='ReLU', color='blue')

    labels = ['Linear', 'Sigmoid', 'Tanh', 'ReLU']
    plt.bar(labels, softmax_vals, alpha=0.5, color=['blue', 'green', 'red', 'purple'], label='Softmax', width=0.2, align='center')
    plt.title('Activation Functions and Softmax Output')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    print("Simple Perceptron Model with Different Activation Functions\n")
    input_value = float(input("Enter input value: "))
    weight = float(input("Enter weight: "))
    bias = float(input("Enter bias: "))
    output_linear = perceptron(input_value, weight, bias, linear)
    
    output_sigmoid = perceptron(input_value, weight, bias, sigmoid)
    output_tanh = perceptron(input_value, weight, bias, tanh)
    output_relu = perceptron(input_value, weight, bias, relu)
    print("\nOutput using different activation functions")
    
    print(f"Linear activation output: {output_linear}")
    print(f"Sigmoid activation output: {output_sigmoid}")
    print(f"Tanh activation output: {output_tanh}")
    print(f"ReLU activation output: {output_relu}")
    input_vector = np.array([output_linear, output_sigmoid, output_tanh, output_relu])
    output_softmax = softmax(input_vector)
    
    print("\nSoftmax output (probabilities):")
    print(f"Softmax outputs: {output_softmax}")
    plot_activation_functions(input_vector, output_softmax)

if __name__ == "__main__":
    main()
