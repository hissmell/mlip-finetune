from mlip_finetune import Trainer, NaiveStrategy
from mlip_finetune.configs import load_config

# Load configuration
config = load_config("configs/naive_finetune.yaml")

# Create and run trainer
trainer = Trainer(config)
trainer.setup()
trainer.fit()

# Evaluate
metrics = trainer.evaluate()
print(f"Validation loss: {metrics['loss']:.4f}")