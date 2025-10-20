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
    #hiddenZ;
    #hiddenA;
    #outputZ;
    #outputA;
    #batchSize;
    #weightsIHChange;
    #weightsHOChange;
    #biasIHChange;
    #biasHOChange;

    constructor(inputNodes, hiddenNodes, outputNodes, costFunction = errorSquared, learningRate = 0.005, activationFunction = sigmoid, activationFunctionOutput = sigmoid)
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
        this.biasIH = math.zeros(this.hiddenNodes).toArray();
        this.biasHO = math.zeros(this.outputNodes).toArray();

        // feedforward variables
        this.#hiddenZ;
        this.#hiddenA;
        this.#outputZ;
        this.#outputA;

        // backpropigation variables
        this.#batchSize = 0;
        this.#weightsIHChange = math.zeros(math.size(this.weightsIH));
        this.#weightsHOChange = math.zeros(math.size(this.weightsHO));
        this.#biasIHChange = math.zeros(this.biasIH.length).toArray();
        this.#biasHOChange = math.zeros(this.biasHO.length).toArray();
    }

    #readOnlyProxy(obj)
    {
        if (obj === null || obj === undefined) return obj;
        
        return new Proxy(obj, {
            set(target, property, value) {
                console.error(`Attempt to modify read-only property '${property}'`);
                return false;
            },
            deleteProperty(target, property) {
                console.error(`Attempt to delete read-only property '${property}'`);
                return false;
            },
            get(target, property) {
                const value = target[property];
                if (value !== null && typeof value === 'object') {
                    return new Proxy(value, this);
                }
                return value;
            }
        });
    }

    #inputValidation(input)
    {
        if (!Array.isArray(input) || input.length !== this.inputNodes) {
            console.error(`Input must be an array of length ${this.inputNodes}`);
            return true;
        }
        return false;
    }

    #targetValidation(target)
    {
        if (!Array.isArray(target) || target.length !== this.outputNodes) {
            console.error(`Target must be an array of length ${this.outputNodes}`);
            return true;
        }
        return false;
    }

    guess(input)
    {
        // validate input
        if (this.#inputValidation(input)) {
            return null;
        }

        // generate hidden layer Z and A
        this.#hiddenZ = math.add(math.multiply(input, this.weightsIH), this.biasIH);
        this.#hiddenA = this.activation.func(this.#hiddenZ);

        // generate outputs Z and A
        this.#outputZ = math.add(math.multiply(this.#hiddenA, this.weightsHO), this.biasHO);
        this.#outputA = this.activationOutput.func(this.#outputZ);

        // send back array of outputs
        return this.#outputA;
    }

    addToBatch(input, target)
    {
        // validate input and target
        if (this.#inputValidation(input) || this.#targetValidation(target)) {
            return;
        }

        this.guess(input);
        this.#backprop(input, target);
        this.#batchSize++;
    }

    computeBatch()
    {
        if (this.#batchSize >= 1)
        {       
            // Update biases
            this.biasHO = math.add(this.biasHO, this.#biasHOChange);
            this.biasIH = math.add(this.biasIH, this.#biasIHChange);

            // Update weights
            this.weightsHO = math.add(this.weightsHO, this.#weightsHOChange);
            this.weightsIH = math.add(this.weightsIH, this.#weightsIHChange);
            this.clearBatch();
        }
        else
        {
            console.warn("No batches to compute, nothing will happen");
        }
    }

    clearBatch()
    {
        this.#batchSize = 0;
        this.#weightsIHChange = math.zeros(math.size(this.weightsIH));
        this.#weightsHOChange = math.zeros(math.size(this.weightsHO));
        this.#biasIHChange = math.zeros(this.biasIH.length).toArray();
        this.#biasHOChange = math.zeros(this.biasHO.length).toArray();
    }

    train(input, target)
    {
        if (this.#batchSize >= 1)
        {
            console.warn("Please compute or discard current batch to train single example");
        }
        else
        {
            this.clearBatch();
            this.addToBatch(input, target);
            this.computeBatch();
        }
    }

    #backprop(input, target)
    {
        let dcdzo;
        let dcdao;
        let dadzo;

        // -Output layer-
        // when softmax and cross entropy are used together the dc/dz(outputs) calculation can be greatly simplified
        if (this.activationOutput.name == activationFuncNames.softmax && this.costFunction.name == costFuncNames.crossEntropy)
        {
            // dc/dz(outputs) = #outputA - target
            dcdzo = math.subtract(this.#outputA, target);
        }
        else
        {
            // dc/da(outputs)
            dcdao = this.costFunction.dfunc(target, this.#outputA);

            // TODO: give activationOutput.dfunc ability to cheat and use #outputA as the dfunc uses the func for sigmoid and softmax
            // da/dz(outputs)
            dadzo = this.activationOutput.dfunc(this.#outputZ);

            // dc/dz(outputs) is calculated differently depending on if the activationOutput is a scalar or vector function
            if(this.activationOutput.functionType == functionTypes.scalar)
            {
                // dc/dz(outputs) = dc/dao ○ da/dzo (element wise)
                dcdzo = math.dotMultiply(dcdao, dadzo);
            }
            else
            {
                // dc/dz(outputs) = dc/dao(T) ⋅ da/dzo (dot product)
                let test = [dcdao];
                let test2 = math.transpose([dcdao])
                let test3  = [math.transpose(dcdao)]
                dcdzo = math.squeeze(math.multiply([dcdao], dadzo));
            }
        }

        // dc/dw(outputs) = dzo/dwo(T) ⋅ dc/dzo (dot product)
        let dcdwo = math.multiply(math.transpose([this.#hiddenA]), [dcdzo]);

        // -Hidden layer- 
        // dc/da(hidden) = dc/dz(outputs) ⋅ dz/da(hidden)(T)
        let dcdah = math.multiply([dcdzo], math.transpose(this.weightsHO));

        // da/dz(hidden)
        let dadzh = [this.activation.dfunc(this.#hiddenZ)];

        // dc/dz(hidden) = dc/dah ○ da/dzh (element wise)
        let dcdzh = math.dotMultiply(dcdah, dadzh);

        // dc/dw(hidden) = dzh/dwh(T) ⋅ dc/dzh (dot product)
        let dcdwh = math.multiply(math.transpose([input]), dcdzh);

        // Update biases
        this.#biasHOChange = math.subtract(this.#biasHOChange, math.multiply(math.squeeze(dcdzo), this.learningRate));
        this.#biasIHChange = math.subtract(this.#biasIHChange, math.multiply(math.squeeze(dcdzh), this.learningRate));

        // Update weights
        this.#weightsHOChange = math.subtract(this.#weightsHOChange, math.multiply(math.squeeze(dcdwo), this.learningRate));
        this.#weightsIHChange = math.subtract(this.#weightsIHChange, math.multiply(math.squeeze(dcdwh), this.learningRate));
    }

    changeLearningRate(newLearningRate)
    {
        if (isNaN(newLearningRate))
        {
            console.warn("Cannot update learning rate because input was not a number")
        }
        else
        {
           this.learningRate = newLearningRate;
        }
    }

    exportBrain()
    {
        var brainExport = {};
        brainExport["weightsIH"] = this.weightsIH;
        brainExport["weightsHO"] = this.weightsHO;
        brainExport["biasH"] = this.biasIH;
        brainExport["biasO"] = this.biasHO;
        return this.#readOnlyProxy(brainExport);
    }

    importBrain(brainImport)
    {
        this.weightsIH = brainImport.weightsIH;
        this.weightsHO = brainImport.weightsHO;
        this.biasIH = brainImport.biasH;
        this.biasHO = brainImport.biasO;
    }

    getHiddenZ()
    {
        return this.#readOnlyProxy(this.#hiddenZ);
    }

    getHiddenA()
    {
        return this.#readOnlyProxy(this.#hiddenA);
    }

    getOutputZ()
    {
        return this.#readOnlyProxy(this.#outputZ);
    }

    getOutputA()
    {
        return this.#readOnlyProxy(this.#outputA);
    }

    getWeightsIH()
    {
        return this.#readOnlyProxy(this.weightsIH);
    }

    getWeightsHO()
    {
        return this.#readOnlyProxy(this.weightsHO);
    }

    getBiasIH()
    {
        return this.#readOnlyProxy(this.biasIH);
    }

    getBiasHO()
    {
        return this.#readOnlyProxy(this.biasHO);
    }

    getBatchSize()
    {
        return this.#batchSize;
    }

    getWeightsIHChange()
    {
        return this.#readOnlyProxy(this.#weightsIHChange);
    }

    getWeightsHOChange()
    {
        return this.#readOnlyProxy(this.#weightsHOChange);
    }

    getBiasIHChange()
    {
        return this.#readOnlyProxy(this.#biasIHChange);
    }

    getBiasHOChange()
    {
        return this.#readOnlyProxy(this.#biasHOChange);
    }
}

exports.JellyBrain = JellyBrain
exports.costFuncs = costFuncs
exports.activationFuncs = activationFuncs