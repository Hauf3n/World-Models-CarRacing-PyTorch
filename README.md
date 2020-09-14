# World Models CarRacing PyTorch

 Implementation of [World Models](https://arxiv.org/abs/1803.10122) paper by David Ha, Jürgen Schmidhuber.<br />
 Applied to the gym CarRacing-v0 environment. Model-based reinforcement learning.
 
 # Tasks
 1 -  Collect data. Let the agent act in the env without optimization. I added a random track tile start to enrich the data.<br /><br />
 2 -  Train VAE and save the encodings.<br /><br />
 3 -  Train Mixture-Density-Network-RNN (MDN-RNN) with the VAE encodings. Occasionally kill some gradients because of potential inf loss.<br /><br />
 4 -  Train the Controller network which has the input: MDN-RNN hidden state + VAE encoding. Use an evolutionary algorithm (CMA-ES) for optimization.<br /><br />
 5 -  (Try) dream environment where the MDN-RNN generates the VAE encodings and let the VAE decoder output the encoding. No inputs from the real environment.<br />
 
 # Results
 
 ![runs](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/runs.gif)
 [Youtube](https://www.youtube.com/watch?v=CAA_a5qtD34)<br />
 The major improvements in regards to model free RL methods (watch CarRacing environment on Youtube)
 are that the Controller network gets an oberservation encoding and a time dependent representation from the RNN. 
 Because of that you are able to solve the task by just applying an evolutionary algorithm on top of that.
 
 # Evolutionary training
 
 ![es](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/es.png)
 - (small) population size: 16, rollouts: 18
 
 # VAE representations
 
 ![vae](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/vae.gif)
 
 # Dream env (random policy here)
 
 ![dream](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/dream_r.gif)
 
 - working poorly, mdn-rnn only gives a good prediction when you insert a sequence of 3+ env steps.
 One step prediction doesnt seem to work properly ... Maybe overfitting, i don't know.
