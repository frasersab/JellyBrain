const math = require('mathjs')

// activation function class
class ActivationFunction {
    constructor(func, dfunc) {
        this.func = func;
        this.dfunc = dfunc;
    }
}

let lrelu = new ActivationFunction(
    x => { if (x < 0) { return 0.1 * x } else { return x } },
    y => { if (y < 0) { return 0.1 } else { return 1 } }
);

let relu = new ActivationFunction(
    x => { if (x < 0) { return 0 } else { return x } },
    y => { if (y <= 0) { return 0 } else { return 1 } }
);

let sigmoid = new ActivationFunction(
    x => 1 / (1 + math.exp(-x)),
    y => y * (1 - y)
);

let tanh = new ActivationFunction(
    x => math.tanh(x),
    y => 1 - (y * y)
);

class JellyBrain {
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate = 0.1, activationFunction = relu) {
        // set the parameteres for the neural network
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;

        // find the optimal range to initialise the weights to prevent saturation
        let initIHRange = 1 / math.sqrt(this.hiddenNodes);
        let initHORange = 1 / math.sqrt(this.outputNodes);

        // create the weights matrix with random weights within the specified range
        this.weightsIH = math.random([this.hiddenNodes, this.inputNodes], -initIHRange, initIHRange);
        this.weightsHO = math.random([this.outputNodes, this.hiddenNodes], -initHORange, initHORange);

        // create bias arrays with initialisation to 0
        this.biasHidden = math.zeros([1, this.hiddenNodes]);
        this.biasOutput = math.zeros([1, outputNodes]);

    }

    guess(inputs) {
        // // Generate hidden layer
        // let intputArray = Matrix.fromArray(inputs);
        // let hiddenArray = Matrix.multiply(this.weightsIH, intputArray);
        // hiddenArray.add(this.biasHidden);
        // hiddenArray.map(this.activationFunction.func);

        // // Generate outputs
        // let outputs = Matrix.multiply(this.weightsHO, hiddenArray);
        // outputs.add(this.biasOutput);
        // outputs.map(this.activationFunction.func);

        // // send back array of outputs
        // return outputs.toArray();
    }
    train() {

    }


    clone(brain) {
        // clone an existing trained brain
    }
}

test = new JellyBrain(2, 4, 2);

console.log(relu.dfunc(3));
// let ran = 1 / math.sqrt(4)
// console.log(ran);
// let matrix = math.random([4, 4], -ran, ran);
// console.table(matrix);