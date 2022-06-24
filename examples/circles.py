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

plt.style.use("fivethirtyeight")

# Load data
features, targets = make_circles(n_samples=1000, noise=0.1)
features = features.tolist()
targets = targets.reshape(-1, 1).tolist()

# Helper function to flatten targets list
def flatten(nested):
    unnested = []
    for element in nested:
        unnested += element
    return unnested


# Define colors for each class
class_0_correct_color = "#009600"
class_1_correct_color = "#960000"
class_0_incorrect_color = "#ddffdd"
class_1_incorrect_color = "#ffdddd"

# Split data per class for visualization
class_0 = []
class_1 = []
for feature, target in zip(features, flatten(targets)):
    if target == 0:
        class_0.append(feature)
    if target == 1:
        class_1.append(feature)

class_0_x, class_0_y = data_util.transpose(class_0)
class_1_x, class_1_y = data_util.transpose(class_1)

fig = plt.figure("", figsize=(20, 8))

# Plot preview of the dataset
preview_ax = fig.add_subplot(121, title="preview")
preview_ax.tick_params(labelsize=10)
preview_ax.scatter(
    class_0_x, class_0_y, s=15, color=[class_0_correct_color] * len(class_0)
)
preview_ax.scatter(
    class_1_x, class_1_y, s=15, color=[class_1_correct_color] * len(class_1)
)


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

# Prepare visualizations
class_0_result_colors = [class_0_incorrect_color] * len(class_0)
class_1_result_colors = [class_1_incorrect_color] * len(class_1)

plt.ion()

training_ax = fig.add_subplot(122, title="training")
training_ax.tick_params(labelsize=10)

class_0_plot = training_ax.scatter(
    class_0_x, class_0_y, s=15, color=class_0_result_colors
)
class_1_plot = training_ax.scatter(
    class_1_x, class_1_y, s=15, color=class_1_result_colors
)

progress = tqdm(
    range(epochs),
    unit="epochs",
    bar_format="Training: {percentage:3.0f}% |{bar:40}| {n_fmt}/{total_fmt}{postfix}",
)

# Training loop
for epoch in progress:
    tinputs, ttargets = data_util.shuffle(features, targets)
    learning_rate = decay(epoch)

    sample_counter = 1
    for tinput, ttarget in zip(tinputs, ttargets):

        # Forward pass through the model
        state = tinput
        for layer in model.layers:
            if isinstance(layer, Layer):
                state = model.optimizer.forward_pass(layer, state)
            elif isinstance(layer, Dropout):
                state = layer.forward_pass(state, training=True)
            else:
                state = layer.forward_pass(state)
        outputs = state

        # Loss derivative for sample
        dloss = loss_function.derivative(outputs, ttarget)

        # Backward pass through the model
        dstate = dloss

        for layer in reversed(model.layers):
            dstate = layer.backprop(dstate)

        # Perform the update
        is_batch = sample_counter % batch_size == 0
        is_last = sample_counter == len(tinputs)

        if is_batch or is_last:
            for layer in model.layers:
                if isinstance(layer, Layer):
                    model.optimizer.update(layer, learning_rate, batch_size)

        sample_counter += 1

    # Update visualization
    class_0_result_colors = []
    for sample in class_0:
        if model.predict(sample)[0] <= 0.5:
            class_0_result_colors.append(class_0_correct_color)
        else:
            class_0_result_colors.append(class_0_incorrect_color)
    class_0_plot.set_color(class_0_result_colors)

    class_1_result_colors = []
    for sample in class_1:
        if model.predict(sample)[0] > 0.5:
            class_1_result_colors.append(class_1_correct_color)
        else:
            class_1_result_colors.append(class_1_incorrect_color)
    class_1_plot.set_color(class_1_result_colors)

    fig.canvas.draw()
    plt.pause(0.0001)

plt.ioff()
plt.show()
