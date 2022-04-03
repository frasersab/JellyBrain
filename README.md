# JellyBrain

JellyBrain is a simple neural network written in Javascript. This was written as an exercise to learn how neural networks work.

## Usage

```javascript
const JellyBrain = require('./JellyBrain');

let brain = new JellyBrain(2, 2, 1);    // 2 inputs, 2 hidden nodes, 1 output

brain.train([0.2, 0.5], [1, 1]);
brain.guess([0.1, 0.6]);
```

## License
[ISC](https://choosealicense.com/licenses/isc/)