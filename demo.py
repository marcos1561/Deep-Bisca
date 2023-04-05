import models as models
from observation import History, Observation

train_cfg = models.Dnn.TrainConfig(
    train_episodes=2000, batch_size=64,
    steps_to_update_target_model=30, 
    steps_to_update_p2_model= 26 * 20,
    steps_to_train_model=9, 
    min_replay_size=1_000, #min_replay_size=1000,
)

bellmans_cfg = models.Dnn.BellmansEqConfig(
    learning_rate=0.2, discount_factor=0.418,
)

neural_network_cfg = models.Dnn.NeuralNetworkConfig(
    learning_rate=0.001*2,
)

exploration = models.Exploration(
    min_epsilon=0.01,
    x_percent_at_k_min=0.2,
    train_episodes=train_cfg.train_episodes,
)

model = models.Dnn(train_cfg, bellmans_cfg, neural_network_cfg, exploration,
    observation_system=History(), name="history_6_v2", load_model_name="history_5", from_checkpoint=True)

model.train()