// This is a simple neural network
// By Fraser Sabine

import * as math from 'mathjs';

// activation function class
class ActivationFunction
{
    constructor(func, dfunc)
    {
        this.func = func;
        this.dfunc = dfunc;
    }
}

export let lrelu = new ActivationFunction
(
    x => { if (x < 0) { return 0.1 * x } else { return x } },
    y => { if (y < 0) { return 0.1 } else { return 1 } }
);

export let relu = new ActivationFunction
(
    x => { if (x < 0) { return 0 } else { return x } },
    y => { if (y <= 0) { return 0 } else { return 1 } }
);

export let sigmoid = new ActivationFunction
(
    x => 1 / (1 + math.exp(-x)),
    y => (1 / (1 + math.exp(-y))) * (1 - (1 / (1 + math.exp(-y))))
);

export let tanh = new ActivationFunction
(
    x => math.tanh(x),
    y => 1 - (math.tanh(y) * math.tanh(y))
);

export let linear = new ActivationFunction
(
    x => x,
    y => y
);

export class JellyBrain
{
    constructor(inputNodes, hiddenNodes, outputNodes, learningRate = 0.3, activationFunction = tanh, activationFunctionOutput = tanh)
    {
        // set the parameteres for the neural network
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.learningRate = learningRate;
        this.activation = activationFunction;
        this.activationOutput = activationFunctionOutput;

        // find the optimal range to initialise the weights to prevent saturation
        let initIHRange = 1 / math.sqrt(this.hiddenNodes);
        let initHORange = 1 / math.sqrt(this.outputNodes);

        // create the weights matrix with random weights within the specified range
        this.weightsIH = math.random([this.inputNodes, this.hiddenNodes], -initIHRange, initIHRange);
        this.weightsHO = math.random([this.hiddenNodes, this.outputNodes], -initHORange, initHORange);

        // create bias arrays with initialisation to 0
        this.biasH = math.zeros(this.hiddenNodes).toArray();
        this.biasO = math.zeros(this.outputNodes).toArray();

    }

    guess(inputs)
    {
        // --Feedforward algorithm--
        // generate hidden layer Z and A
        let hiddenZ = math.add(math.multiply(inputs, this.weightsIH), this.biasH);
        let hiddenA = math.map(hiddenZ, this.activation.func);

        // generate outputs Z and A
        let outputZ = math.add(math.multiply(hiddenA, this.weightsHO), this.biasO);
        let outputA = math.map(outputZ, this.activationOutput.func);

        // send back array of outputs
        return outputA;
    }

    train(inputs, targets)
    {
        // --Feedforward algorithm--
        // generate hidden layer Z and A
        let hiddenZ = math.add(math.multiply(inputs, this.weightsIH), this.biasH);
        let hiddenA = math.map(hiddenZ, this.activation.func);

        // generate outputs Z and A
        let outputZ = math.add(math.multiply(hiddenA, this.weightsHO), this.biasO);
        let outputA = math.map(outputZ, this.activationOutput.func);

        // --Backpropogation algorithm--
        // -Layer 1-
        // dc/da(outputs)
        let dcdao = math.subtract(targets, outputA);

        // da/dz(outputs)
        let dadzo = math.map(outputZ, this.activation.dfunc);

        // dc/dz(outputs) = dc/dao ⊙ da/dzo
        let dcdzo = math.dotMultiply(dcdao, dadzo);

        // dc/dw(outputs) = dzo/dwo(T) * dc/dzo
        let dcdwo = math.multiply(math.transpose([hiddenA]), [dcdzo]);

        // -Layer 2-
        // dc/da(hidden)
        let dcdah = math.multiply(dcdzo, math.transpose(this.weightsHO));

        // da/dz(hidden)
        let dadzh = math.map(hiddenZ, this.activation.dfunc);

        // dc/dz(hidden) = dc/dah ⊙ da/dzh
        let dcdzh = math.dotMultiply(dcdah, dadzh);

        // dc/dw(hidden) = dzh/dwh(T) * dc/dzh
        let dcdwh = math.multiply(math.transpose([inputs]), [dcdzh]);

        // Update Biases
        this.biasO = math.add(this.biasO, math.multiply(dcdzo, this.learningRate));
        this.biasH = math.add(this.biasH, math.multiply(dcdzh, this.learningRate));

        // Update weights
        this.weightsHO = math.add(this.weightsHO, math.multiply(dcdwo, this.learningRate));
        this.weightsIH = math.add(this.weightsIH, math.multiply(dcdwh, this.learningRate));

        return dcdao;
    }

    clone(brain)
    {
        // clone an existing trained brain
    }
}