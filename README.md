20250928

I performed the mnist prediction with new activation
I thought the sigmoid function would be better because a biological neuron is often described as “on/off” (0 or 1).
But sigmoid is hard to train due to gradient vanishing.
So I tried an activation of sigmoid(x) + f(x), where f(x) = a·x and a decays to 0 during training.
The training ran successfully, but the final accuracy was about the same as with ReLU.
I’m not sure why, and I suspect the activation function isn’t a big factor for this setup.



<img width="703" height="70" alt="image" src="https://github.com/user-attachments/assets/d9665996-f419-41c3-b8b8-e3b023165da4" />


\[
a \;=\; \left(\frac{1}{10000}\right)^{\tfrac{\text{present epoch}}{\text{total epoch}}}
\]
