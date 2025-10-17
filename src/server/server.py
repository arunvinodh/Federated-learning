import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

sys.path.append(str(Path(__file__).resolve().parent.parent))

import flwr as fl
from models.cnn_lstm import CNNDPLSTM

# Optional: Custom strategy with verbose logging
class VerboseFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        print(f"[Server] Round {rnd} - Received {len(results)} results")
        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        print(f"[Server] Round {rnd} - Evaluation results from {len(results)} clients")
        return super().aggregate_evaluate(rnd, results, failures)

strategy = VerboseFedAvg(
    min_available_clients=2,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

# Start server (yes, it's deprecated — but still works)
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
