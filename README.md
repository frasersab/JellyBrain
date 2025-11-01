# JellyBrain

JellyBrain is a simple neural network written in Javascript. This was written as an exercise to learn how neural networks work. You can also test out the neural network with hand drawn numbers here: https://frasersab.github.io/JellyBrainInteractive/

## Installation
```
npm install jellybrain
```

## Simple Usage

```javascript
const {JellyBrain} = require('../JellyBrain.js');

let brain = new JellyBrain(2, 2, 1);    // 2 inputs, 2 hidden nodes, 1 output

brain.train([0.2, 0.5], [1]);
brain.guess([0.1, 0.6]);
```

## Available Functions

### Activation Functions
- `sigmoid` - Sigmoid activation (output range 0-1)
- `tanh` - Hyperbolic tangent (output range -1 to 1)
- `relu` - Rectified Linear Unit
- `lrelu` - Leaky ReLU
- `linear` - Linear activation (no transformation)
- `softmax` - Softmax activation (for multi-class classification)

### Cost Functions
- `errorSquared` - Mean squared error (default)
- `crossEntropy` - Cross entropy (for multi-class with softmax)
- `binaryCrossEntropy` - Binary cross entropy

## Advanced Usage

### Custom Configuration
```javascript
const {JellyBrain, costFuncs, activationFuncs} = require('../JellyBrain.js');

// Constructor: (inputNodes, hiddenNodes, outputNodes, costFunction, learningRate, hiddenActivation, outputActivation)
let brain = new JellyBrain(
    784,                           // input nodes
    784,                           // hidden nodes
    10,                            // output nodes
    costFuncs.crossEntropy,        // cost function
    0.001,                         // learning rate
    activationFuncs.sigmoid,       // hidden layer activation
    activationFuncs.softmax        // output layer activation
);
```

### Batch Training
```javascript
let simpleBrain = new JellyBrain(2, 2, 1);

simpleBrain.addToBatch([0.2, 0.5], [1]);
simpleBrain.addToBatch([0.6, 0.4], [0.7]);
simpleBrain.addToBatch([0.1, 0.2], [0.2]);
simpleBrain.computeBatch();
simpleBrain.clearBatch();
```

### Saving and Loading Brains
```javascript
// Export brain state
let brainData = brain.exportBrain();
let jsonString = JSON.stringify(brainData);

// Import brain state
let loadedData = JSON.parse(jsonString);
brain.importBrain(loadedData);
```

## Example Configurations

### Simple Linear Regression
```javascript
const {JellyBrain, sigmoid} = require('../JellyBrain.js');
let brain = new JellyBrain(1, 8, 1, undefined, 0.5, sigmoid, sigmoid);
```

### Binary Classification
```javascript
const {JellyBrain} = require('../JellyBrain.js');
let brain = new JellyBrain(2, 5, 1);
brain.setLearningRate(0.1);
```

### Multi-class Classification (MNIST)
```javascript
const {JellyBrain, costFuncs, activationFuncs} = require('../JellyBrain.js');
let brain = new JellyBrain(784, 784, 10, costFuncs.crossEntropy, 0.0008, activationFuncs.sigmoid, activationFuncs.softmax);
```

## Examples

The `src/examples/` directory contains several working examples:

- **simpleLinearRegression.js** - Learning y = 2x using sigmoid activation
- **multipleLinearRegression.js** - Learning y = 2a + 3b using sigmoid activation
- **binaryClassification.js** - Classifying points above/below a line
- **numberIdentifier.js** - MNIST digit recognition with pre-trained models


## Utility Scripts

Generate PNG images from dataset files:

```bash
# Generate MNIST images
npm run generate-mnist -- 0 10 test   # First 10 test images
npm run generate-mnist -- 0 10 train  # First 10 training images

# Generate custom dataset images
npm run generate-custom -- 0 10       # First 10 custom images
```

## License
[ISC](https://choosealicense.com/licenses/isc/)