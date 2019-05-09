# AlphaUT3
AlphaZero for ultimate tic-tac-toe.

## What is ultimate tic-tac-toe?
It's like tic-tac-toe, but each square of the game contains another game of tic-tac-toe in it! Win small games to claim the squares in the big game. Simple, right? But there is a catch: Whichever small square you pick is the next big square your opponent must play in. [Read more...](https://docs.riddles.io/ultimate-tic-tac-toe/rules)

![ultimate tic-tac-toe gif](https://static-content.riddles.io/ultimate-tic-tac-toe-objectives-small-squares.gif)

## What is AlphaZero?
AlphaZero is a reinforcement learning algorithm trained only using self-play. It combines a neural network and Monte Carlo Tree Search in an elegant policy iteration framework to achieve stable learning. [Read more...](https://web.stanford.edu/~surag/posts/alphazero.html)

## Experiments
Coming soon.

## To-do
 - ~~Reshape nnet policy output to match board shape, i.e. retain the spatial relationship between game state and policy vector.~~
 - [Scale terminal results by the game length to prefer shorter games](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68).
 - [Introduce Dirichlet noise into the MCTS](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5).
 - [Use an average of `v` and `q` as a training target](https://medium.com/oracledevs/lessons-from-alphazero-part-4-improving-the-training-target-6efba2e71628).
 - [Cyclical learning rate](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a).
 - [~~Slow window size increase~~](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a).
 - Implement UT3 neural network in other frameworks, eg: TensorFlow.

## Requirements
 - [PyTorch](https://pytorch.org/)

## Thanks
 - [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
 - [pytorch_connect4](https://github.com/tfolkman/pytorch_connect4)
 - [pytorch-classification](https://github.com/bearpaw/pytorch-classification)
 - [progress](https://github.com/verigak/progress)
