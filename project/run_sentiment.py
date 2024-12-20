import os.path

print("Setting environment variables...")

os.environ['HOME'] = 'C:\\Users\\addas'
os.environ['EMBEDDINGS_ROOT'] = 'C:\\Users\\addas\\.embeddings'

print("Importing libraries...")

import embeddings
import random
from tqdm import tqdm
import numpy as np
import minitorch
from minitorch import Tensor, TensorBackend
from datasets import load_dataset

BACKEND = minitorch.TensorBackend(minitorch.FastOps)

print("Backend initialized.")

def RParam(*shape):
    r = 0.1 * (minitorch.rand(shape, backend=BACKEND) - 0.5)
    return minitorch.Parameter(r)


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        batch, in_size = x.shape
        return (
            x.view(batch, in_size) @ self.weights.value.view(in_size, self.out_size)
        ).view(batch, self.out_size) + self.bias.value


class Conv1d(minitorch.Module):
    def __init__(self, in_channels, out_channels, kernel_width):
        super().__init__()
        self.weights = RParam(out_channels, in_channels, kernel_width)
        self.bias = RParam(1, out_channels, 1)

    def forward(self, input):
        # # TODO: Implement for Task 4.5.
        # raise NotImplementedError("Need to implement for Task 4.5")
        x = input.permute(0, 2, 1)
        out = minitorch.conv1d(x, self.weights.value)
        out = out + self.bias.value
        return out

class CNNSentimentKim(minitorch.Module):
    """
    Implement a CNN for Sentiment classification based on Y. Kim 2014.

    This model should implement the following procedure:

    1. Apply a 1d convolution with input_channels=embedding_dim
        feature_map_size=100 output channels and [3, 4, 5]-sized kernels
        followed by a non-linear activation function (the paper uses tanh, we apply a ReLu)
    2. Apply max-over-time across each feature map
    3. Apply a Linear to size C (number of classes) followed by a ReLU and Dropout with rate 25%
    4. Apply a sigmoid over the class dimension.
    """

    def __init__(
        self,
        feature_map_size=100,
        embedding_size=50,
        filter_sizes=[3, 4, 5],
        dropout=0.25,
    ):
        super().__init__()
        self.convs = [
            Conv1d(embedding_size, feature_map_size, fs) for fs in filter_sizes
        ]

        # TODO: Implement for Task 4.5.
        # raise NotImplementedError("Need to implement for Task 4.5")

        self.conv1 = Conv1d(embedding_size, feature_map_size, filter_sizes[0])
        self.conv2 = Conv1d(embedding_size, feature_map_size, filter_sizes[1])
        self.conv3 = Conv1d(embedding_size, feature_map_size, filter_sizes[2])

        self.dropout_rate = dropout
        self.linear = Linear(feature_map_size * len(filter_sizes), 1)

    def forward(self, embeddings):
        """
        embeddings tensor: [batch x sentence length x embedding dim]
        """
        # TODO: Implement for Task 4.5.
        # raise NotImplementedError("Need to implement for Task 4.5")
        feature_maps = []
        for conv in self.convs:
            c_out = conv(embeddings)
            c_out = c_out.relu()
            c_pooled = minitorch.max(c_out, dim=2)
            c_pooled = c_pooled.view(c_pooled.shape[0], c_pooled.shape[1])
            feature_maps.append(c_pooled)

        batch_size = feature_maps[0].shape[0]

        cat_features = cat(feature_maps, dim=1)

        out = self.linear(cat_features)
        out = minitorch.dropout(out, p=self.dropout_rate)
        out = out.sigmoid()
        return out.view(out.shape[0])

# Evaluation helper methods
def get_predictions_array(y_true, model_output):
    predictions_array = []
    for j, logit in enumerate(model_output.to_numpy()):
        true_label = y_true[j]
        if logit > 0.5:
            predicted_label = 1.0
        else:
            predicted_label = 0
        predictions_array.append((true_label, predicted_label, logit))
    return predictions_array


def cat(tensors: list[Tensor], dim: int) -> Tensor:
    if not tensors:
        raise ValueError("Tensor list is empty")

    num_dims = len(tensors[0].shape)
    for tensor in tensors:
        if len(tensor.shape) != num_dims:
            raise ValueError("All tensors must have the same number of dimensions")

    for d in range(num_dims):
        if d != dim:
            size = tensors[0].shape[d]
            for tensor in tensors[1:]:
                if tensor.shape[d] != size:
                    raise ValueError(f"Size mismatch at dimension {d}")

    new_shape = list(tensors[0].shape)
    new_shape[dim] = sum(tensor.shape[dim] for tensor in tensors)
    new_shape = tuple(new_shape)

    np_arrays = [tensor.to_numpy() for tensor in tensors]
    concatenated_np = np.concatenate(np_arrays, axis=dim)

    new_storage = concatenated_np.flatten().tolist()
    return Tensor.make(new_storage, new_shape, backend=tensors[0].backend)

def get_accuracy(predictions_array):
    correct = 0
    for y_true, y_pred, logit in predictions_array:
        if y_true == y_pred:
            correct += 1
    return correct / len(predictions_array)


best_val = 0.0

def default_log_fn(
    epoch,
    train_loss,
    losses,
    train_predictions,
    train_accuracy,
    validation_predictions,
    validation_accuracy,
):
    global best_val
    current_train_acc = train_accuracy[-1] if train_accuracy else 0.0
    current_val_acc = validation_accuracy[-1] if validation_accuracy else 0.0
    best_val = max(best_val, current_val_acc)

    log_message = (
        f"Epoch {epoch}, "
        f"loss {train_loss:.4f}, "
        f"train accuracy: {current_train_acc*100:.2f}%, "
        f"validation accuracy: {current_val_acc*100:.2f}%\n"
        f"Best Valid accuracy: {best_val*100:.2f}%\n"
    )

    print(log_message.strip())

    with open("sentiment.txt", "a", encoding="utf-8") as log_file:
        log_file.write(log_message)

class SentenceSentimentTrain:
    def __init__(self, model):
        self.model = model

    def train(
        self,
        data_train,
        learning_rate,
        batch_size=10,
        max_epochs=500,
        data_val=None,
        log_fn=default_log_fn,
    ):
        print("Starting training...")
        model = self.model
        (X_train, y_train) = data_train
        n_training_samples = len(X_train)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []
        train_accuracy = []
        validation_accuracy = []
        for epoch in range(1, max_epochs + 1):
            total_loss = 0.0

            model.train()
            train_predictions = []
            batch_size = min(batch_size, n_training_samples)
            for batch_num, example_num in enumerate(
                    tqdm(range(0, n_training_samples, batch_size), desc=f"Epoch {epoch}")
                ):
                    y = minitorch.tensor(
                        y_train[example_num : example_num + batch_size], backend=BACKEND
                    )
                    x = minitorch.tensor(
                        X_train[example_num : example_num + batch_size], backend=BACKEND
                    )
                    x.requires_grad_(True)
                    y.requires_grad_(True)
                    # Forward
                    out = model.forward(x)
                    prob = (out * y) + (out - 1.0) * (y - 1.0)
                    loss = -(prob.log() / y.shape[0]).sum()
                    loss.view(1).backward()

                    # Save train predictions
                    train_predictions += get_predictions_array(y, out)
                    total_loss += loss[0]

                    # Update
                    optim.step()

                    if batch_num % 10 == 0:
                        print(f"  Batch {batch_num}: Current loss {loss[0]:.4f}")

            # Evaluate on validation set at the end of the epoch
            validation_predictions = []
            if data_val is not None:
                (X_val, y_val) = data_val
                model.eval()
                y = minitorch.tensor(
                    y_val,
                    backend=BACKEND,
                )
                x = minitorch.tensor(
                    X_val,
                    backend=BACKEND,
                )
                out = model.forward(x)
                validation_predictions += get_predictions_array(y, out)
                val_acc = get_accuracy(validation_predictions)
                validation_accuracy.append(val_acc)
                model.train()

            train_accuracy.append(get_accuracy(train_predictions))
            losses.append(total_loss)
            log_fn(
                epoch,
                total_loss,
                losses,
                train_predictions,
                train_accuracy,
                validation_predictions,
                validation_accuracy,
            )
            total_loss = 0.0

            if val_acc > 0.7:
                print(f"Best validation accuracy exceeded 70% at epoch {epoch}. Stopping training.")
                break


def encode_sentences(
    dataset, N, max_sentence_len, embeddings_lookup, unk_embedding, unks
):
    Xs = []
    ys = []
    for sentence in tqdm(dataset["sentence"][:N], desc="Encoding sentences"):
        sentence_embedding = [[0] * embeddings_lookup.d_emb] * max_sentence_len
        for i, w in enumerate(sentence.split()):
            sentence_embedding[i] = [0] * embeddings_lookup.d_emb
            if w in embeddings_lookup:
                sentence_embedding[i][:] = embeddings_lookup.emb(w)
            else:
                unks.add(w)
                sentence_embedding[i][:] = unk_embedding
        Xs.append(sentence_embedding)
    ys = dataset["label"][:N]
    return Xs, ys


def encode_sentiment_data(dataset, pretrained_embeddings, N_train, N_val=0):
    print("Determining maximum sentence length...")

    max_sentence_len = 0
    for sentence in dataset["train"]["sentence"] + dataset["validation"]["sentence"]:
        max_sentence_len = max(max_sentence_len, len(sentence.split()))
    print(f"Maximum sentence length: {max_sentence_len}")

    unks = set()
    unk_embedding = [
        0.1 * (random.random() - 0.5) for i in range(pretrained_embeddings.d_emb)
    ]
    print("Encoding training data...")
    X_train, y_train = encode_sentences(
        dataset["train"],
        N_train,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print("Encoding validation data...")
    X_val, y_val = encode_sentences(
        dataset["validation"],
        N_val,
        max_sentence_len,
        pretrained_embeddings,
        unk_embedding,
        unks,
    )
    print(f"missing pre-trained embedding for {len(unks)} unknown words")

    return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
    with open("sentiment.txt", "w", encoding="utf-8") as log_file:
        log_file.write("Training Logs\n\n")

    train_size = 450
    validation_size = 100
    learning_rate = 0.01
    max_epochs = 250

    (X_train, y_train), (X_val, y_val) = encode_sentiment_data(
        load_dataset("glue", "sst2"),
        embeddings.GloveEmbedding("wikipedia_gigaword", d_emb=50, show_progress=True),
        train_size,
        validation_size,
    )
    model_trainer = SentenceSentimentTrain(
        CNNSentimentKim(feature_map_size=100, filter_sizes=[3, 4, 5], dropout=0.25)
    )
    model_trainer.train(
        (X_train, y_train),
        learning_rate,
        max_epochs=max_epochs,
        data_val=(X_val, y_val),
    )
