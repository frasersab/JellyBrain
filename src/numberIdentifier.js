const math = require('mathjs');
const {JellyBrain, linear} = require('./JellyBrain.js');
const {readMNIST, saveMNIST} = require('./mnist_dataset/mnist_reader');
const cliProgress = require('cli-progress');


let brain = new JellyBrain(784, 784, 10);

// training function
function trainer(brain, amount) {
    var pixelValues = readMNIST(0, amount, '\\train_images_60k.idx3-ubyte', '\\train_labels_60k.idx1-ubyte');
    const progressBar = new cliProgress.SingleBar({format: 'Training Progress |' + '{bar}' + '| {percentage}% | {value}/{total} | ETA: {eta}s'}, cliProgress.Presets.shades_classic);
    progressBar.start(amount, 0);
    
    pixelValues.forEach(function(image)
    {
        targetArray = new Array(10).fill(0);
        targetArray[image.label] = 1;
        brain.train(image.pixels, targetArray);
        progressBar.increment();
    })
    progressBar.stop();
}

// testing function
function tester(brain, amount) {
    var pixelValues = readMNIST(0, amount, '\\test_images_10k.idx3-ubyte', '\\test_labels_10k.idx1-ubyte');
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

let trainAmount = 50;
let testAmount = 20;

let accuracyTable = Array();
accuracyTable.push(["Training Samples", "Accuracy"]);

accuracyTable.push([0, tester(brain, testAmount)]);
trainer(brain, trainAmount);
accuracyTable.push([trainAmount, tester(brain, testAmount)]);

console.table(accuracyTable);