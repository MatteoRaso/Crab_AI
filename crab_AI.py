import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
df = df.drop("id", axis=1)
df = df.drop("Sex", axis=1)

y = df["Age"]
X = df.drop("Age", axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(20, activation="relu"))
model.add(tf.keras.layers.Dense(1))

model.compile(loss="MAE", optimizer="adam", metrics=["MAE"])

history = model.fit(X, y, epochs=200, validation_split=0.2, batch_size=512)

df = pd.read_csv("test.csv")
ID = df["id"]
df = df.drop("id", axis=1)
df = df.drop("Sex", axis=1)

output = model.predict(df).flatten()
submission = pd.DataFrame(data={"id": ID, "Age": output})
submission.to_csv("submission.csv", index=False)

plt.plot(history.history["MAE"], label="loss")
plt.plot(history.history["val_MAE"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

