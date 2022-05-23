def linear_decay(base_rate: float, current_epoch: int, total_epochs: int) -> float:
    return base_rate * (1.0 - (current_epoch / total_epochs))
