// This is a simple neural network
// By Fraser Sabine

const math = require('mathjs')

// enumerations
const costFuncNames = Object.freeze(
{
    errorSquared: Symbol(0),
    crossEntropy: Symbol(1),
    binaryCrossEntropy: Symbol(2)
})

const activationFuncNames = Object.freeze(
{
    softmax: Symbol(0),
    lrelu: Symbol(1),
    relu: Symbol(2),
    sigmoid: Symbol(3),
    tanh: Symbol(4),
    linear: Symbol(5)
})

const functionTypes = Object.freeze(
{
    vector: Symbol(0),
    scalar: Symbol(1)
})

// cost function class
class costFunction{
    constructor(name, dfunc)
    {
        this.name = name;
        this.dfunc = dfunc;
    }
}

let errorSquared = new costFunction
(
    costFuncNames.errorSquared,
    (x,y) => {return math.subtract(x,y)}
)

let crossEntropy = new costFunction
(
    costFuncNames.crossEntropy,
    (x,y) => {return math.dotDivide(x,y)}
)

let binaryCrossEntropy = new costFunction
(
    costFuncNames.binaryCrossEntropy,
    (x,y) => {return math.dotDivide(x, math.dotMultiply(y, math.log(2)))}
)

// activation function class
class ActivationFunction
{
    constructor(name, functionType, func, dfunc)
    {
        this.name = name;
        this.functionType = functionType;
        this.func = func;
        this.dfunc = dfunc;
    }
}

let softmax = new ActivationFunction
(
    activationFuncNames.softmax,
    functionTypes.vector,
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
    activationFuncNames.lrelu,
    functionTypes.scalar,
    x => {return math.map(x, z => {if (z < 0) { return 0.1 * z } else { return z }})},
    y => {return math.map(y, z => {if (z < 0) { return 0.1 } else { return 1 }})}  
);

let relu = new ActivationFunction
(
    activationFuncNames.relu,
    functionTypes.scalar,
    x => {return math.map(x, z => {if (z < 0) { return 0 } else { return z }})},
    y => {return math.map(y, z => {if (z <= 0) { return 0 } else { return 1 }})}
);

let sigmoid = new ActivationFunction
(
    activationFuncNames.sigmoid,
    functionTypes.scalar,
    x => {return math.map(x, z => 1 / (1 + math.exp(-z)))},
    y => {return math.map(y, z => (1 / (1 + math.exp(-z))) * (1 - (1 / (1 + math.exp(-z)))) )}
);

let tanh = new ActivationFunction
(
    activationFuncNames.tanh,
    functionTypes.scalar,
    x => {return math.map(x, z => math.tanh(z))},
    y => {return math.map(y, z => 1 - (math.tanh(z) * math.tanh(z)))}
);

let linear = new ActivationFunction
(
    activationFuncNames.linear,
    functionTypes.scalar,
    x => x,
    y => y
);

// Dictionary of cost functions
const costFuncs = Object.freeze(
{
    errorSquared: errorSquared,
    crossEntropy: crossEntropy,
    binaryCrossEntropy: binaryCrossEntropy
})

// Dictionary of activation functions
const activationFuncs = Object.freeze(
{
    softmax: softmax,
    lrelu: lrelu,
    relu: relu,
    sigmoid: sigmoid,
    tanh: tanh,
    linear: linear
})

class JellyBrain
{
    constructor(inputNodes, hiddenNodes, outputNodes, costFunction = errorSquared, learningRate = 0.3, activationFunction = sigmoid, activationFunctionOutput = sigmoid)
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
        let dcdao;
        let dadzo;
        let dcdzo;

        if (this.activationOutput.name == activationFuncNames.softmax && this.costFunction.name == costFuncNames.crossEntropy)
        {
            // when softmax and cross entropy is used together the dc/dz(outputs) calculation can be greatly simplified
            // dc/dz(outputs)
            dcdzo = math.subtract(outputA, targets);
        }
        else
        {
            // dc/da(outputs)
            dcdao = this.costFunction.dfunc(targets, outputA);

            // TODO: give activationOutput.dfunc abiity to cheat and use outputA as the dfunc uses the func for sigmoid and softmax
            // da/dz(outputs)
            dadzo = this.activationOutput.dfunc(outputZ);

            // dc/dz(outputs) is calculated differently depending on if the activationOutput is a scalar or vector function
            if(this.activationOutput.functionType == functionTypes.scalar)
            {
                // dc/dz(outputs) = dc/dao ○ da/dzo (elemnt wise)
                dcdzo = math.dotMultiply(dcdao, dadzo);
            }
            else
            {
                // dc/dz(outputs) = dc/dao(T) ⋅ da/dzo (dot product)
                dcdzo = math.multiply([math.transpose(dcdao)], dadzo);
            }
        }

        // dc/dw(outputs) = dzo/dwo(T) ⋅ dc/dzo (dot product)
        let dcdwo = math.multiply(math.transpose([hiddenA]), [dcdzo]);

        // -Layer 2- 
        // dc/da(hidden) = dc/dz(outputs) ⋅ dz/da(hidden)(T)
        let dcdah = math.multiply(dcdzo, math.transpose(this.weightsHO));

        // da/dz(hidden)
        let dadzh = this.activation.dfunc(hiddenZ);

        // dc/dz(hidden) = dc/dah ○ da/dzh (element wise)
        let dcdzh = math.dotMultiply(dcdah, dadzh);

        // dc/dw(hidden) = dzh/dwh(T) ⋅ dc/dzh (dot product)
        let dcdwh = math.multiply(math.transpose([inputs]), [dcdzh]);

        // Update biases
        this.biasO = math.subtract(this.biasO, math.multiply(dcdzo, this.learningRate));
        this.biasH = math.subtract(this.biasH, math.multiply(dcdzh, this.learningRate));

        // Update weights
        this.weightsHO = math.subtract(this.weightsHO, math.multiply(dcdwo, this.learningRate));
        this.weightsIH = math.subtract(this.weightsIH, math.multiply(dcdwh, this.learningRate));

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
exports.costFuncs = costFuncs
exports.activationFuncs = activationFuncs