"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.


class Network(minitorch.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.layer1_weights = RParam(hidden_size, 2)
        self.layer1_biases = RParam(hidden_size)

        self.layer2_weights = RParam(hidden_size, hidden_size)
        self.layer2_biases = RParam(hidden_size)

        self.layer3_weights = RParam(1, hidden_size)
        self.layer3_bias = RParam(1)

    def forward(self, x):
        N = x.shape[0]

        x_exp = x.view(N, 1, 2)
        weights1_exp = self.layer1_weights.value.view(1, self.hidden_size, 2)
        z1 = (x_exp * weights1_exp).sum(2)
        z1 = z1.view(N, self.hidden_size)
        z1 += self.layer1_biases.value.view(1, self.hidden_size)
        a1 = z1.relu()

        a1_exp = a1.view(N, 1, self.hidden_size)
        weights2_exp = self.layer2_weights.value.view(1, self.hidden_size, self.hidden_size)
        z2 = (a1_exp * weights2_exp).sum(2)
        z2 = z2.view(N, self.hidden_size)
        z2 += self.layer2_biases.value.view(1, self.hidden_size)
        a2 = z2.relu()

        a2_exp = a2.view(N, 1, self.hidden_size)
        weights3_exp = self.layer3_weights.value.view(1, 1, self.hidden_size)
        z3 = (a2_exp * weights3_exp).sum(2)
        z3 = z3.view(N)
        z3 += self.layer3_bias.value
        output = z3.sigmoid()
        return output.view(N, 1)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    for dataset in minitorch.datasets:
        print(f"-------------- Training on Dataset: {dataset} ---------------")
        data = minitorch.datasets[dataset](PTS)
        TensorTrain(HIDDEN).train(data, RATE)
        print(f"------------------ Finished Training ---------------------")
