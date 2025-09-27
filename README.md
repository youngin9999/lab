20250928

I performed the mnist prediction with new activation
I thought the sigmoid function would be better because a biological neuron is often described as “on/off” (0 or 1).
But sigmoid is hard to train due to gradient vanishing.
So I tried an activation of sigmoid(x) + f(x), where f(x) = a·x and a decays to 0 during training.
The training ran successfully, but the final accuracy was about the same as with ReLU.
I’m not sure why, and I suspect the activation function isn’t a big factor for this setup.


act(x) = sigmoid(x) + a·x
a = (1/10000)^(1/epoch) and finist setting is 0
