import torch

from ml.models.lstm import LSTMConfig, LSTMSignClassifier


def test_forward_pass_output_shape():
    config = LSTMConfig(input_size=222, num_classes=5, hidden_size=16, num_layers=1)
    model = LSTMSignClassifier(config)

    batch = torch.randn(4, 32, 222)  # (batch, seq_len, input_size)
    logits = model(batch)

    assert logits.shape == (4, 5)


def test_forward_pass_with_multiple_layers_and_dropout():
    config = LSTMConfig(input_size=10, num_classes=3, hidden_size=8, num_layers=2, dropout=0.5)
    model = LSTMSignClassifier(config)

    logits = model(torch.randn(2, 16, 10))

    assert logits.shape == (2, 3)


def test_different_sequence_lengths_produce_same_output_shape():
    config = LSTMConfig(input_size=6, num_classes=4, hidden_size=8, num_layers=1)
    model = LSTMSignClassifier(config)

    short = model(torch.randn(3, 5, 6))
    long = model(torch.randn(3, 40, 6))

    assert short.shape == long.shape == (3, 4)
