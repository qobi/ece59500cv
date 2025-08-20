import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class classifier(Model):
  def __init__(self):
    super(classifier, self).__init__()
    self.d = Dense(1)

  def call(self, x):
    return self.d(x)

def loss(points, labels, model):
    input = tf.constant(points, dtype = tf.float32)
    target = tf.constant(labels, dtype = tf.int32)
    output = classifier(input)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = target,
                                                   logits = output)
    return loss

def initialize():
    global model, optimizer
    model = classifier()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

@tf.function
def step():
    model.train()
    input = torch.tensor(points, dtype = torch.float, requires_grad = True)
    target = torch.tensor(labels, dtype = torch.long)
    output = model(input)
    loss = torch.nn.functional.cross_entropy(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train():
    initialize()
    for i in range(100): step()

def classify(point, model):
    model.eval()
    input = torch.tensor([point], dtype = torch.float)
    output = model(input)
    _, prediction = output.data.max(1)
    return prediction[0]
