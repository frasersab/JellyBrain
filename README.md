# JellyBrain

JellyBrain is a simple neural network written in Javascript. This was written as an exercise to learn how neural networks work. You can also test out the neural network with hand drawn numbers here: https://frasersab.github.io/JellyBrainInteractive/

## Installation
```
npm install jellybrain
```

## Simple Usage

```javascript
const JellyBrain = require('../JellyBrain.js');

let brain = new JellyBrain(2, 2, 1);    // 2 inputs, 2 hidden nodes, 1 output

brain.train([0.2, 0.5], 1);
brain.guess([0.1, 0.6]);
```

## Advanced Usage
It is possible to use a variety of cost functions and activation functions. It is also possible to train in batches, although the implementation avoids using tensors to simplify things.

```javascript
const {JellyBrain, costFuncs, activationFuncs} = require('../JellyBrain.js');

// example parameters for training on the mnist dataset
let brain = new JellyBrain(784, 784, 10, costFuncs.crossEntropy, 0.001, activationFuncs.sigmoid, activationFuncs.softmax);
let simpleBrain = new JellyBrain(2, 2, 1);

simpleBrain.addToBatch([0.2, 0.5], 1);
simpleBrain.addToBatch([0.6, 0.4], 0.7);
simpleBrain.addToBatch([0.1, 0.2], 0.2);
simpleBrain.computeBatch();
simpleBrain.guess([0.1, 0.6]);
```

## License
[ISC](https://choosealicense.com/licenses/isc/)
