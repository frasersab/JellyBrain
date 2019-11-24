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
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate = 0.1, activationFunction = sigmoid) {
        // set the parameteres for the neural network
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;
        this.activation = activationFunction;

        // find the optimal range to initialise the weights to prevent saturation
        let initIHRange = 1 / math.sqrt(this.hiddenNodes);
        let initHORange = 1 / math.sqrt(this.outputNodes);

        // create the weights matrix with random weights within the specified range
        this.weightsIH = math.random([this.inputNodes, this.hiddenNodes], -initIHRange, initIHRange);
        this.weightsHO = math.random([this.hiddenNodes, this.outputNodes], -initHORange, initHORange);

        // create bias arrays with initialisation to 0
        this.biasH = math.zeros(this.hiddenNodes);
        this.biasO = math.zeros(this.outputNodes);

    }

    guess(inputs) {
        // --Feedforward algorithm--
        // generate hidden layer Z and A
        let hiddenZ = math.add(math.multiply(inputs, this.weightsIH), this.biasH);
        let hiddenA = math.map(hiddenZ, this.activation.func);

        // generate outputs Z and A
        let outputZ = math.add(math.multiply(hiddenA, this.weightsHO), this.biasO);
        let outputA = math.map(outputZ, this.activation.func);

        // send back array of outputs
        return outputA.toArray();
    }

    train(inputs, targets) {
        // --Feedforward algorithm--
        // generate hidden layer Z and A
        let hiddenZ = math.add(math.multiply(inputs, this.weightsIH), this.biasH);
        let hiddenA = math.map(hiddenZ, this.activation.func);

        // generate outputs Z and A
        let outputZ = math.add(math.multiply(hiddenA, this.weightsHO), this.biasO);
        let outputA = math.map(outputZ, this.activation.func);

        // --Backpropogation algorithm--
        // dc/da(outputs)
        let dcdao = math.subtract(targets, inputs);

        // da/dz(outputs)
        let dadzo = math.map(outputZ, this.activation.dfunc);

        // Set biases for output layer
        // dc/db(outputs) = dc/dao .* da/dzo .* dz/dbo where dz/dbo = 1
        let dcdbo = math.dotMultiply(dcdao, dadzo);
        this.biasO = math.add(this.biasO, math.multiply(dcdbo, this.learningRate));

        // Set weights for output layer
        // dc/dw(outputs) = dc/dao .* da/dzo .* dz/dwo = dc/dbo .* dz/dwo = dc/dbo .* hiddenA
        let dcdwo = math.zeros(math.size(this.weightsHO));
        for (let i = 0; i < math.size(this.weightsHO)[0]; i++) {
            for (let j = 0; j < math.size(this.weightsHO)[1]; j++) {
                dcdwo[i][j] = hiddenA[i] * dcdbo[j];
            }
        }

        // dc/da(hidden)
        let dcdah = math.multiply(dcdbo, math.transpose(this.weightsHO));

        // da/dz(hidden)
        let dadzh = math.map(hiddenZ, this.activation.dfunc);

        // Set biases for hidden layer
        // dc/db(hidden) = dc/dah .* da/dzh .* dz/dbh where dz/dbh = 1
        let dcdbh = math.dotMultiply(dcdah, dadzh);
        this.biasH = math.add(this.biasH, math.multiply(dcdbh, this.learningRate));

        // Set weights for output layer
        // dc/dw(outputs) = dc/dao .* da/dzo .* dz/dwo = dc/dbo .* dz/dwo = dc/dbo .* hiddenA
        let dcdwh = math.zeros(math.size(this.weightsIH));
        for (let i = 0; i < math.size(this.weightsIH)[0]; i++) {
            for (let j = 0; j < math.size(this.weightsIH)[1]; j++) {
                dcdwh[i][j] = inputs[i] * dcdbh[j];
            }
        }

        // Update weights
        this.weightsHO = math.add(this.weightsHO, math.multiply(dcdwo, this.learningRate));
        this.weightsIH = math.add(this.weightsIH, math.multiply(dcdwh, this.learningRate));

        console.table(this.weightsHO);
        console.table(this.weightsIH);
        return outputA.toArray();
    }

    clone(brain) {
        // clone an existing trained brain
    }


}

test = new JellyBrain(2, 2, 2);

//console.table(test.guess([5, 2]));
console.table(test.train([0.4, 0.8], [1, 1]));
