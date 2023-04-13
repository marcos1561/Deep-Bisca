import models as models
from observation import History, Observation

train_cfg = models.Dnn.TrainConfig(
    train_episodes=6000, batch_size=64,
    steps_to_update_target_model=30, 
    steps_to_update_p2_model= 26 * 20,
    steps_to_train_model=6, 
    min_replay_size=100, #min_replay_size=1000,
)

bellmans_cfg = models.Dnn.BellmansEqConfig(
    learning_rate=0.2, discount_factor=0.98,
)

neural_network_cfg = models.Dnn.NeuralNetworkConfig(
    learning_rate=0.001/2, tau=0.001/2
)

exploration = models.Exploration(
    min_epsilon=0.01,
    x_percent_at_k_min=0.3,
    train_episodes=train_cfg.train_episodes,
)


model = models.Dnn(train_cfg, bellmans_cfg, neural_network_cfg, exploration,
    observation_system=History(), name="model_v2", load_model_name="model_v1", from_checkpoint=False)

model.train()