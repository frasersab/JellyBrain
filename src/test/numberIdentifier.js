const math = require('mathjs');
const path = require('path');
const {JellyBrain, costFuncs, activationFuncs} = require('../JellyBrain.js');
const {readMNIST, saveMNIST} = require('../mnist_dataset/mnist_reader.js');
const fs = require('fs');
const cliProgress = require('cli-progress');


let brain = new JellyBrain(784, 784, 10, costFuncs.crossEntropy, 0.001, activationFuncs.sigmoid, activationFuncs.softmax);

// save brain
function saveBrain(brain, name)
{
    let contents = JSON.stringify(brain.exportBrain());
    fs.writeFileSync(path.join(__dirname, '..'
    , 'brains', `${name}.json`), contents);
}

// load brain
function loadBrain(brain, name)
{
    let contents = fs.readFileSync(path.join(__dirname, '..', 'brains', `${name}.json`));
    contents = JSON.parse(contents);
    brain.importBrain(contents);
}


// training function
function trainer(brain, numberOfBatches, batchSize = 1, start = 0) {
    const progressBar = new cliProgress.SingleBar({format: 'Training Progress |' + '{bar}' + '| {percentage}% | {value}/{total} | ETA: {eta}s'}, cliProgress.Presets.shades_classic);
    progressBar.start(numberOfBatches * batchSize, 0);

    for (let batchNumber = 0; batchNumber < numberOfBatches; batchNumber++)
    {
        let batchStart = start + (batchNumber * batchSize);
        let pixelValues = readMNIST(batchStart, batchStart + batchSize, '\\train_images_60k.idx3-ubyte', '\\train_labels_60k.idx1-ubyte', true);
        pixelValues.forEach(function(image)
        {
            targetArray = new Array(10).fill(0);
            targetArray[image.label] = 1;
            brain.addToBatch(image.pixels, targetArray);
            progressBar.increment();
        })
        brain.computeBatch();
    }
    progressBar.stop();
}

// testing function
function tester(brain, amount, start = 0) {
    var pixelValues = readMNIST(start, start + amount, '\\test_images_10k.idx3-ubyte', '\\test_labels_10k.idx1-ubyte', true);
    const progressBar = new cliProgress.SingleBar({format: 'Testing Progress |' + '{bar}' + '| {percentage}% | {value}/{total} | ETA: {eta}s'}, cliProgress.Presets.shades_classic);
    progressBar.start(amount, 0);
    let accuracy = 0;
    pixelValues.forEach(function(image)
    {
        targetArray = new Array(10).fill(0);
        targetArray[image.label] = 1;
        var guessArray = brain.guess(image.pixels);
        var highestOutput = math.max(guessArray);
        var guessNumber = guessArray.indexOf(highestOutput);
        if (guessNumber == image.label){
            accuracy++;
        }
        progressBar.increment();
    })

    progressBar.stop();
    return accuracy = (accuracy / amount) * 100;
}

let numberOfBatches = 5;
let batchSize = 10;
let startFrom = 0;
let testAmount = 50;

//saveBrain(brain, "brain_before");
//loadBrain(brain, "brain21000")


let accuracyTable = Array();
accuracyTable.push(["Training Samples", "Accuracy"]);
accuracyTable.push([0, tester(brain, testAmount) + "%"]);
trainer(brain, numberOfBatches, batchSize, startFrom);
accuracyTable.push([numberOfBatches * batchSize, tester(brain, testAmount) + "%"]);

saveBrain(brain, "brainBatchTest");

console.table(accuracyTable);