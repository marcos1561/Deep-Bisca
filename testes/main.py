from tensorflow.keras.models import load_model
from numpy.random import random 

model = load_model("saved_models/bianca_v1")

teste = model.predict(random(47).reshape(1, 47))
print(teste)
input()

