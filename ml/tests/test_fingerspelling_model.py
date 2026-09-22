import torch

from ml.fingerspelling.model import INPUT_SIZE, FingerspellingMLP


def test_forward_pass_shape():
    model = FingerspellingMLP(num_classes=26, hidden_size=32)
    x = torch.zeros((4, INPUT_SIZE), dtype=torch.float32)

    logits = model(x)

    assert logits.shape == (4, 26)


def test_can_learn_a_trivially_separable_toy_problem():
    """Sanity check, same spirit as ml/tests/test_train.py's tiny checkpoint
    run: not a claim about real accuracy, just that the model/training loop
    plumbing actually reduces loss on data it CAN separate."""
    torch.manual_seed(0)
    model = FingerspellingMLP(num_classes=2, hidden_size=16)
    x = torch.cat([torch.zeros((20, INPUT_SIZE)), torch.ones((20, INPUT_SIZE))])
    y = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    initial_loss = criterion(model(x), y).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    final_loss = criterion(model(x), y).item()

    assert final_loss < initial_loss * 0.1
