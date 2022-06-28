import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
module_path = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(module_path)

import matplotlib as mplt
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from tqdm import tqdm

import perceptron.data_util as data_util
from perceptron.model import Model
from perceptron.layer import Layer
from perceptron.dropout import Dropout
from perceptron.activation import Sigmoid, LeakyRelu
from perceptron.decay import LinearDecay
from perceptron.loss import BinaryCrossentropy


def flatten(nested):
    """Helper function for flattening targets list."""
    unnested = []
    for element in nested:
        unnested += element
    return unnested


if __name__ == "__main__":
    plt.style.use("fivethirtyeight")

    # Load data
    features, targets = make_circles(n_samples=1000, noise=0.1)
    features = features.tolist()
    targets = targets.reshape(-1, 1).tolist()

    # Define colors for each class
    correct_color_0 = "#009600"
    correct_color_1 = "#960000"
    incorrect_color_0 = "#ddffdd"
    incorrect_color_1 = "#ffdddd"

    # Split data per class for visualization
    f0 = []  # Features of class 0
    f1 = []  # Features of class 1
    for feature, target in zip(features, flatten(targets)):
        if target == 0:
            f0.append(feature)
        if target == 1:
            f1.append(feature)

    x0, y0 = data_util.transpose(f0)
    x1, y1 = data_util.transpose(f1)

    # Prepare figure to plot
    fig = plt.figure("", figsize=(20, 8))

    # Plot preview of the dataset
    preview_ax = fig.add_subplot(121, title="preview")
    preview_ax.tick_params(labelsize=10)
    preview_ax.scatter(x0, y0, s=25, color=[correct_color_0] * len(f0))
    preview_ax.scatter(x1, y1, s=25, color=[correct_color_1] * len(f1))

    # Prepare model for training
    def model_factory():
        model = Model()
        model.add(Layer(input_size=2, layer_size=5))
        model.add(LeakyRelu())
        model.add(Layer(input_size=5, layer_size=5))
        model.add(LeakyRelu())
        model.add(Layer(input_size=5, layer_size=1))
        model.add(Sigmoid())

        return model

    model = model_factory()
    model.compile("GD")

    epochs = 50
    base_lr = 0.01
    batch_size = 1
    decay = LinearDecay(base_lr, epochs)
    loss_function = BinaryCrossentropy()

    # Enable interactive mode
    plt.ion()

    # Prepare plots for training visualization
    training_ax = fig.add_subplot(122, title="training")
    training_ax.tick_params(labelsize=10)

    result_colors_0 = [incorrect_color_0] * len(f0)
    result_colors_1 = [incorrect_color_1] * len(f1)
    class_0_plot = training_ax.scatter(x0, y0, s=25, color=result_colors_0)
    class_1_plot = training_ax.scatter(x1, y1, s=25, color=result_colors_1)

    # Training loop
    progress = tqdm(
        range(epochs),
        unit="epochs",
        bar_format="Training: {percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt}{postfix}",
    )

    for epoch in progress:
        tinputs, ttargets = data_util.shuffle(features, targets)
        learning_rate = decay(epoch)

        sample_counter = 1
        for tinput, ttarget in zip(tinputs, ttargets):

            # Forward pass through the model
            outputs = model.forward_pass(tinput)

            # Loss derivative for sample
            dloss = loss_function.derivative(outputs, ttarget)

            # Backward pass through the model
            model.backprop(dloss)

            # Perform the update
            is_batch = sample_counter % batch_size == 0
            is_last = sample_counter == len(tinputs)

            if is_batch or is_last:
                model.update(learning_rate, batch_size)

            sample_counter += 1

        # Update visualization
        result_colors_0 = [
            correct_color_0 if model.predict(sample)[0] <= 0.5 else incorrect_color_0
            for sample in f0
        ]
        class_0_plot.set_color(result_colors_0)

        result_colors_1 = [
            incorrect_color_1 if model.predict(sample)[0] <= 0.5 else correct_color_1
            for sample in f1
        ]
        class_1_plot.set_color(result_colors_1)

        fig.canvas.draw()
        plt.pause(0.0001)

    plt.ioff()
    plt.show()
