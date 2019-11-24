const Matrix = require('./Matrix.js')

class ActivationFunction {
    constructor(func, dfunc) {
        this.func = func;
        this.dfunc = dfunc;
    }
}

let sigmoid = new ActivationFunction(
    x => 1 / (1 + Math.exp(-x)),
    y => y * (1 - y)
);

let tanh = new ActivationFunction(
    x => Math.tanh(x),
    y => 1 - (y * y)
);

class JellyBrain {
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate = 0.1, activationFunction = sigmoid) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;
        this.activationFunction = activationFunction;

        this.weightsIH = new Matrix(hiddenNodes, inputNodes);
        this.weightsHO = new Matrix(outputNodes, hiddenNodes);
        this.weightsIH.randomise();
        this.weightsHO.randomise();

        this.biasHidden = new Matrix(this.hiddenNodes, 1);
        this.biasOutput = new Matrix(this.outputNodes, 1);
        this.biasHidden.randomise();
        this.biasOutput.randomise();
    }

    guess(inputs) {
        // Generate hidden layer
        let intputArray = Matrix.fromArray(inputs);
        let hiddenArray = Matrix.multiply(this.weightsIH, intputArray);
        hiddenArray.add(this.biasHidden);
        hiddenArray.map(this.activationFunction.func);

        // Generate outputs
        let outputs = Matrix.multiply(this.weightsHO, hiddenArray);
        outputs.add(this.biasOutput);
        outputs.map(this.activationFunction.func);

        // send back array of outputs
        return outputs.toArray();
    }
    train() {

    }


    clone(brain) {
        // clone an existing trained brain
    }
}

const matrix = math.matrix([[7, 1], [-2, 3]])

console.log(matrix);
test = new JellyBrain(2, 4, 2);
console.log(test.guess([0, 1]));