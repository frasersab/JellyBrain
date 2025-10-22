// This is a simple neural network
// By Fraser Sabine

const math = require('mathjs')

// numerations
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
    (x,y) => {return math.subtract(y,x)}
)

let crossEntropy = new costFunction
(
    costFuncNames.crossEntropy,
    (x,y) => {return math.dotDivide(math.multiply(x, -1), y)}
)

let binaryCrossEntropy = new costFunction
(
    costFuncNames.binaryCrossEntropy,
    (x,y) => {
        // dC/dpred = -(y/(p*ln(2)) - (1-y)/((1-p)*ln(2)))
        // = -[y/p - (1-y)/(1-p)] / ln(2)
        const term1 = math.dotDivide(x, y); // target / pred
        const term2 = math.dotDivide(math.subtract(1, x), math.subtract(1, y)); // (1-target) / (1-pred)
        return math.dotDivide(math.multiply(math.subtract(term1, term2), -1), Math.log(2));
    }
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
    y => math.ones(math.size(y))
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
    #inputNodes;
    #hiddenNodes;
    #outputNodes;
    #costFunction;
    #learningRate;
    #activation;
    #activationOutput;
    #weightsIH;
    #weightsHO;
    #biasIH;
    #biasHO;
    #hiddenZ;
    #hiddenA;
    #outputZ;
    #outputA;
    #batchSize;
    #weightsIHChange;
    #weightsHOChange;
    #biasIHChange;
    #biasHOChange;

    constructor(inputNodes, hiddenNodes, outputNodes, costFunction = errorSquared, learningRate = 0.05, activationFunction = sigmoid, activationFunctionOutput = sigmoid)
    {
        // set the parameteres for the neural network
        this.#inputNodes = inputNodes;
        this.#hiddenNodes = hiddenNodes;
        this.#outputNodes = outputNodes;
        this.#costFunction = costFunction;
        this.#learningRate = learningRate;
        this.#activation = activationFunction;
        this.#activationOutput = activationFunctionOutput;

        // find the optimal range to initialise the weights to prevent saturation
        let initIHRange = 1 / math.sqrt(this.#hiddenNodes);
        let initHORange = 1 / math.sqrt(this.#outputNodes);

        // create the weights matrix with random weights within the specified range
        this.#weightsIH = math.random([this.#inputNodes, this.#hiddenNodes], -initIHRange, initIHRange);
        this.#weightsHO = math.random([this.#hiddenNodes, this.#outputNodes], -initHORange, initHORange);

        // create bias arrays with initialisation to 0
        this.#biasIH = math.zeros(this.#hiddenNodes).toArray();
        this.#biasHO = math.zeros(this.#outputNodes).toArray();

        // backpropigation variables
        this.#batchSize = 0;
        this.#weightsIHChange = math.zeros(math.size(this.#weightsIH));
        this.#weightsHOChange = math.zeros(math.size(this.#weightsHO));
        this.#biasIHChange = math.zeros(this.#biasIH.length).toArray();
        this.#biasHOChange = math.zeros(this.#biasHO.length).toArray();
    }

    #inputValidation(input)
    {
        if (!Array.isArray(input) || input.length !== this.#inputNodes) {
            console.error(`Input must be an array of length ${this.#inputNodes}`);
            return true;
        }
        return false;
    }

    #targetValidation(target)
    {
        if (!Array.isArray(target) || target.length !== this.#outputNodes) {
            console.error(`Target must be an array of length ${this.#outputNodes}`);
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

        // Dimension notation:
        // I - array length of inputs
        // H - array lrngth of hidden nodes
        // T - array length of targets/outputs

        // Forward propagation - Hidden Layer
        // hiddenZ = input × weightsIH + biasIH
        // [H] = [1×I] × [I×H] + [H]
        this.#hiddenZ = math.add(math.multiply(input, this.#weightsIH), this.#biasIH);
        
        // hiddenA = activation(hiddenZ)
        // [H] = activation([H])
        this.#hiddenA = this.#activation.func(this.#hiddenZ);


        // Forward propagation - Output Layer
        // outputZ = hiddenA × weightsHO + biasHO
        // [T] = [1×H] × [H×O] + [T]
        this.#outputZ = math.add(math.multiply(this.#hiddenA, this.#weightsHO), this.#biasHO);
        
        // outputA = activation(outputZ)
        // [T] = activation([T])
        this.#outputA = this.#activationOutput.func(this.#outputZ);

        // return output activations
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
            this.#biasHO = math.add(this.#biasHO, this.#biasHOChange);
            this.#biasIH = math.add(this.#biasIH, this.#biasIHChange);

            // Update weights
            this.#weightsHO = math.add(this.#weightsHO, this.#weightsHOChange);
            this.#weightsIH = math.add(this.#weightsIH, this.#weightsIHChange);
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
        this.#weightsIHChange = math.zeros(math.size(this.#weightsIH));
        this.#weightsHOChange = math.zeros(math.size(this.#weightsHO));
        this.#biasIHChange = math.zeros(this.#biasIH.length).toArray();
        this.#biasHOChange = math.zeros(this.#biasHO.length).toArray();
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
        // Dimension notation:
        // I - array length of inputs
        // H - array lrngth of hidden nodes
        // T - array length of targets/outputs

        let dcdzo;
        let dcdao;
        let dadzo;

        // -Output layer-
        // when softmax and cross entropy are used together the dc/dz(outputs) calculation can be greatly simplified
        if (this.#activationOutput.name == activationFuncNames.softmax && this.#costFunction.name == costFuncNames.crossEntropy)
        {
            // dc/dz(outputs) = outputA - target
            // [T] = [T] - [T]
            dcdzo = math.subtract(this.#outputA, target);
        }
        else
        {
            // dc/da(outputs)
            // [T]
            dcdao = this.#costFunction.dfunc(target, this.#outputA);

            // TODO: give activationOutput.dfunc ability to cheat and use outputA as the dfunc uses the func for sigmoid and softmax
            // da/dz(outputs) depends on activation type:
            // - scalar activation: [T]
            // - vector activation (softmax): [T×T] - Jacobian matrix
            dadzo = this.#activationOutput.dfunc(this.#outputZ);

            // dc/dz(outputs) is calculated differently depending on if the activationOutput is a scalar or vector function
            if(this.#activationOutput.functionType == functionTypes.scalar)
            {
                // dc/dz(outputs) = dc/dao ○ da/dzo (element wise)
                // [T] = [T] ○ [T]
                dcdzo = math.dotMultiply(dcdao, dadzo);
            }
            else
            {
                // dc/dz(outputs) = dc/dao(T) ⋅ da/dzo (dot product)
                // [T] = squeeze([1xT]) = squeeze([1xT] ⋅ [TxT])
                dcdzo = math.squeeze(math.multiply([dcdao], dadzo));
            }
        }

        // dc/dw(outputs) = dzo/dwo(T) ⋅ dc/dzo (dot product)
        // [HxT] = [Hx1] ⋅ [1xT]
        let dcdwo = math.multiply(math.transpose([this.#hiddenA]), [dcdzo]);

        // -Hidden layer- 
        // dc/da(hidden) = dc/dz(outputs) ⋅ dz/da(hidden)(T)
        // [1xH] = [1xT] ⋅ [TxH]
        let dcdah = math.multiply([dcdzo], math.transpose(this.#weightsHO));

        // da/dz(hidden)
        // [1×H] - wrapped in array for consistency
        let dadzh = [this.#activation.dfunc(this.#hiddenZ)];

        // dc/dz(hidden) = dc/dah ○ da/dzh (element wise)
        // [1×H] = [1×H] ○ [1×H]
        let dcdzh = math.dotMultiply(dcdah, dadzh);

        // dc/dw(hidden) = dzh/dwh(T) ⋅ dc/dzh (dot product)
        // [I×H] = [I×1] ⋅ [1×H]
        let dcdwh = math.multiply(math.transpose([input]), dcdzh);

        // Update biases
        // [T] = [T] - ([T] * learningRate)
        this.#biasHOChange = math.subtract(this.#biasHOChange, math.multiply(dcdzo, this.#learningRate));
        // [H] = [H] - ([H] * learningRate)
        this.#biasIHChange = math.subtract(this.#biasIHChange, math.multiply(dcdzh[0], this.#learningRate));

        // Update weights
        // [H×T] = [H×T] - ([H×T] * learningRate)
        this.#weightsHOChange = math.subtract(this.#weightsHOChange, math.multiply(dcdwo, this.#learningRate));
        // [I×H] = [I×H] - ([I×H] * learningRate)
        this.#weightsIHChange = math.subtract(this.#weightsIHChange, math.multiply(dcdwh, this.#learningRate));
    }

    setLearningRate(newLearningRate)
    {
        if(isNaN(newLearningRate))
        {
            console.warn("Cannot update learning rate because input was not a number")
            return
        }
        else if(newLearningRate == 0)
        {
            console.warn("Learning rate is 0. Nothing will be learned");
        }
        else if(newLearningRate < 0)
        {
            console.warn("Learning rate is negative. Model will get worse");
        }
        this.#learningRate = newLearningRate;
    }

    exportBrain()
    {
        var brainExport = {};
        brainExport["weightsIH"] = this.#weightsIH;
        brainExport["weightsHO"] = this.#weightsHO;
        brainExport["biasH"] = this.#biasIH;
        brainExport["biasO"] = this.#biasHO;
        return structuredClone(brainExport);
    }

    importBrain(brainImport)
    {
        // validate the import structure
        if (!brainImport || typeof brainImport !== 'object') {
            console.error('Invalid brain import data. Import cancelled.');
            return
        }
        // validate dimensions match
        if (!Array.isArray(brainImport.weightsIH) || brainImport.weightsIH.length !== this.#inputNodes || !Array.isArray(brainImport.weightsIH[0]) ||brainImport.weightsIH[0].length !== this.#hiddenNodes) 
        {
            console.error('WeightsIH dimensions do not match network architecture. Import cancelled.');
            return;
        }    
        if (!Array.isArray(brainImport.weightsHO) || brainImport.weightsHO.length !== this.#hiddenNodes || !Array.isArray(brainImport.weightsHO[0]) ||brainImport.weightsHO[0].length !== this.#outputNodes) 
        {
            console.error('WeightsHO dimensions do not match network architecture. Import cancelled.');
            return;
        }
        if (!Array.isArray(brainImport.biasH) || brainImport.biasH.length !== this.#hiddenNodes) 
        {
            console.error('BiasH dimensions do not match network architecture. Import cancelled.');
            return;
        }
        if (!Array.isArray(brainImport.biasO) || brainImport.biasO.length !== this.#outputNodes) 
        {
            console.error('BiasO dimensions do not match network architecture. Import cancelled.');
            return;
        }
        this.#weightsIH = structuredClone(brainImport.weightsIH);
        this.#weightsHO = structuredClone(brainImport.weightsHO);
        this.#biasIH = structuredClone(brainImport.biasH);
        this.#biasHO = structuredClone(brainImport.biasO);
    }

    getInputNodes()
    {
        return this.#inputNodes;
    }

    getHiddenNodes()
    {
        return this.#hiddenNodes;
    }

    getOutputNodes()
    {
        return this.#outputNodes;
    }

    getLearningRate()
    {
        return this.#learningRate;
    }

    getHiddenZ()
    {
        return structuredClone(this.#hiddenZ);
    }

    getHiddenA()
    {
        return structuredClone(this.#hiddenA);
    }

    getOutputZ()
    {
        return structuredClone(this.#outputZ);
    }

    getOutputA()
    {
        return structuredClone(this.#outputA);
    }

    getWeightsIH()
    {
        return structuredClone(this.#weightsIH);
    }

    getWeightsHO()
    {
        return structuredClone(this.#weightsHO);
    }

    getBiasIH()
    {
        return structuredClone(this.#biasIH);
    }

    getBiasHO()
    {
        return structuredClone(this.#biasHO);
    }

    getBatchSize()
    {
        return this.#batchSize;
    }

    getWeightsIHChange()
    {
        return structuredClone(this.#weightsIHChange);
    }

    getWeightsHOChange()
    {
        return structuredClone(this.#weightsHOChange);
    }

    getBiasIHChange()
    {
        return structuredClone(this.#biasIHChange);
    }

    getBiasHOChange()
    {
        return structuredClone(this.#biasHOChange);
    }
}

exports.JellyBrain = JellyBrain
exports.costFuncs = costFuncs
exports.activationFuncs = activationFuncs