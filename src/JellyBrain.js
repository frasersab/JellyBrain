// This is a simple neural network
// By Fraser Sabine

const math = require('mathjs')

// cost function class
class costFunction{
    constructor(dfunc)
    {
        this.dfunc = dfunc;
    }
}

let errorMeanSquared = new costFunction
(
    (x,y) => {return math.subtract(x,y)}
)

let crossEntropy = new costFunction
(
    (x,y) => {return math.dotDivide(math.subtract(x,y), math.dotMultiply(math.map(x, z => 1 - z), x))}
)

// activation function class
class ActivationFunction
{
    constructor(func, dfunc)
    {
        this.func = func;
        this.dfunc = dfunc;
    }
}

let softmax = new ActivationFunction
(
    x =>
    {
        const maxx = math.max(x);
        let expx = math.map(x, z => math.exp(z - maxx));
        const sumexpx = math.sum(expx);
        return math.map(expx, z => math.divide(z, sumexpx))
    },
    y =>
    {
        let softy = softmax.func(y)
        let jacobianDiag = math.diag(softy);
        let jacobianRow = softy;
        for (let i = 0; i < (softy.length - 1) ; i++)
        {
            jacobianRow = jacobianRow.concat(softy);
        }
        jacobianRow = math.reshape(jacobianRow, math.size(jacobianDiag));
        let jacobianCol = math.transpose(jacobianRow);
        let jacobiaSub = math.dotMultiply(jacobianRow, jacobianCol);
        let jacobian = math.subtract(jacobianDiag, jacobiaSub);
        return jacobian;
    }
);


let lrelu = new ActivationFunction
(
    //x => { if (x < 0) { return 0.1 * x } else { return x } },
    //y => { if (y < 0) { return 0.1 } else { return 1 } }
    x => {return math.map(x, z => {if (z < 0) { return 0.1 * z } else { return z }})},
    y => {return math.map(y, z => {if (z < 0) { return 0.1 } else { return 1 }})}  
);

let relu = new ActivationFunction
(
    //x => { if (x < 0) { return 0 } else { return x } },
    //y => { if (y <= 0) { return 0 } else { return 1 } }
    x => {return math.map(x, z => {if (z < 0) { return 0 } else { return z }})},
    y => {return math.map(y, z => {if (z <= 0) { return 0 } else { return 1 }})}
);

let sigmoid = new ActivationFunction
(
    //x => 1 / (1 + math.exp(-x)),
    //y => (1 / (1 + math.exp(-y))) * (1 - (1 / (1 + math.exp(-y))))
    x => {return math.map(x, z => {1 / (1 + math.exp(-z))})},
    y => {return math.map(y, (1 / (1 + math.exp(-z))) * (1 - (1 / (1 + math.exp(-z)))))}
);

let tanh = new ActivationFunction
(
    //x => math.tanh(x),
    //y => 1 - (math.tanh(y) * math.tanh(y))
    x => {return math.map(x, z => math.tanh(z))},
    y => {return math.map(y, z => 1 - (math.tanh(z) * math.tanh(z)))}
);

let linear = new ActivationFunction
(
    x => x,
    y => y
);

class JellyBrain
{
    constructor(inputNodes, hiddenNodes, outputNodes, costFunction = errorMeanSquared, learningRate = 0.3, activationFunction = sigmoid, activationFunctionOutput = sigmoid)
    {
        // set the parameteres for the neural network
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;
        this.costFunction = costFunction;
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
        let hiddenA = this.activation.func(hiddenZ);

        // generate outputs Z and A
        let outputZ = math.add(math.multiply(hiddenA, this.weightsHO), this.biasO);
        let outputA = this.activationOutput.func(outputZ);

        // send back array of outputs
        return outputA;
    }

    train(inputs, targets)
    {
        // --Feedforward algorithm--
        // generate hidden layer Z and A
        let hiddenZ = math.add(math.multiply(inputs, this.weightsIH), this.biasH);
        let hiddenA = this.activation.func(hiddenZ);

        // generate outputs Z and A
        let outputZ = math.add(math.multiply(hiddenA, this.weightsHO), this.biasO);
        let outputA = this.activationOutput.func(outputZ);

        // --Backpropogation algorithm--
        // -Layer 1-
        // dc/da(outputs)
        let dcdao = this.costFunction.dfunc(targets, outputA);

        // da/dz(outputs)
        let dadzo = this.activationOutput.dfunc(outputZ);

        // dc/dz(outputs) = dc/dao ○ da/dzo (elemnt wise)
        let dcdzo = math.dotMultiply(dcdao, dadzo);

        // dc/dw(outputs) = dzo/dwo(T) ⋅ dc/dzo (dot product)
        let dcdwo = math.multiply(math.transpose([hiddenA]), [dcdzo]);

        // -Layer 2-
        // dc/da(hidden)
        let dcdah = math.multiply(dcdzo, math.transpose(this.weightsHO));

        // da/dz(hidden)
        let dadzh = this.activation.dfunc(hiddenZ);

        // dc/dz(hidden) = dc/dah ○ da/dzh (element wise)
        let dcdzh = math.dotMultiply(dcdah, dadzh);

        // dc/dw(hidden) = dzh/dwh(T) ⋅ dc/dzh (dot product)
        let dcdwh = math.multiply(math.transpose([inputs]), [dcdzh]);

        // Update Biases
        this.biasO = math.add(this.biasO, math.multiply(dcdzo, this.learningRate));
        this.biasH = math.add(this.biasH, math.multiply(dcdzh, this.learningRate));

        // Update weights
        this.weightsHO = math.add(this.weightsHO, math.multiply(dcdwo, this.learningRate));
        this.weightsIH = math.add(this.weightsIH, math.multiply(dcdwh, this.learningRate));

        return dcdao;
    }

    exportBrain()
    {
        var brainExport = {};
        brainExport["weightsIH"] = this.weightsIH;
        brainExport["weightsHO"] = this.weightsHO;
        brainExport["biasH"] = this.biasH;
        brainExport["biasO"] = this.biasO;
        return brainExport;
    }

    importBrain(brainImport)
    {
        this.weightsIH = brainImport.weightsIH;
        this.weightsHO = brainImport.weightsHO;
        this.biasH = brainImport.biasH;
        this.biasO = brainImport.biasO;
    }
}

exports.JellyBrain = JellyBrain
exports.errorMeanSquared = errorMeanSquared
exports.crossEntropy = crossEntropy
exports.softmax = softmax
exports.lrelu = lrelu
exports.relu = relu
exports.sigmoid = sigmoid
exports.tanh = tanh
exports.linear = linear