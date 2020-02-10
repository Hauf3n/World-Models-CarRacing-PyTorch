# World Models CarRacing PyTorch

 Implementation of [World Models](https://arxiv.org/abs/1803.10122) paper by David Ha, Jürgen Schmidhuber.<br />
 Applied to the gym CarRacing-v0 environment.
 
 # Results
 
 ![runs](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/runs.gif)
 
 # Evolutionary training
 
 ![es](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/es.png)
 - pop size:16, rollouts:18
 
 # VAE representations
 
 ![vae](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/vae.gif)
 
 # Dream env (random policy here)
 
 ![dream](https://github.com/Hauf3n/World-Models-CarRacing-PyTorch/blob/master/media/dream_r.gif)
 
 - working poorly, mdn-rnn only gives a good prediction when you insert a sequence of 3+ env steps.
 One step prediction doesnt seem to work properly ...
