"""One CPU update on synthetic audio and video tokens."""
import torch
from reed.adaptation import OnlineAdapters

torch.manual_seed(1)
tokens = [torch.randn(4, 6, 16), torch.randn(4, 8, 16)]
head = torch.nn.Linear(16, 5).requires_grad_(False)
adapters = OnlineAdapters(torch.zeros(2, 16), torch.linspace(.1, 1, 100), .9, classes=5)
optimizer = torch.optim.Adam(adapters.parameters(), lr=1e-4)
predictions = adapters.step(lambda x: head(x.mean(1)), tokens, tokens,
                            noisy=(False, True), changed=False, optimizer=optimizer)
print("Predicted classes:", predictions.argmax(1).tolist())
