# JellyBrain

JellyBrain is a simple neural network written in Javascript.

## Installation

Currently you will need to download it from gtihub yourself

## Usage

```javascript
const JellyBrain = require('./JellyBrain');

brain = new JellyBrain(2, 2, 1);    // 2 inputs, 2 hidden, 1 output

brain.train([0.2, 0.5], [1, 1]);
brain.guess([0.1, 0.6]);
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[ISC](https://choosealicense.com/licenses/isc/)