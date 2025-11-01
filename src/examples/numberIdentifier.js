const math = require('mathjs');
const path = require('path');
const {JellyBrain, costFuncs, activationFuncs} = require('../JellyBrain.js');
const {readMNIST} = require('../mnist_dataset/mnist_reader.js');
const fs = require('fs');

// create brain for mnist dataset
let numberBrain = new JellyBrain(784, 784, 10, costFuncs.crossEntropy, 0.0008, activationFuncs.sigmoid, activationFuncs.softmax);

// load brain
function loadBrain(brain, name)
{
    let contents = fs.readFileSync(path.join(__dirname, '..', 'brains', `${name}.json`));
    contents = JSON.parse(contents);
    brain.importBrain(contents);
}

// testing function
function tester(brain, amount, start = 0) {
    var imagesData = readMNIST(start, start + amount, 'test_images_10k.idx3-ubyte', 'test_labels_10k.idx1-ubyte', true);
    
    let accuracy = 0;
    imagesData.forEach(function(image)
    {
        targetArray = new Array(10).fill(0);
        targetArray[image.label] = 1;
        var guessArray = brain.guess(image.pixels);
        var highestOutput = math.max(guessArray);
        var guessNumber = guessArray.indexOf(highestOutput);
        if (guessNumber == image.label){
            accuracy++;
        }
    })

    return accuracy = (accuracy / amount) * 100;
}

console.log("MNIST Digit Recognition Test");
console.log("=============================\n");

let testAmount = 100;
let accuracyTable = Array();
accuracyTable.push(["Model", "Training Samples", "Accuracy (%)"]);

// Test untrained brain
console.log("Testing untrained brain...");
let untrainedBrain = new JellyBrain(784, 784, 10, costFuncs.crossEntropy, 0.0008, activationFuncs.sigmoid, activationFuncs.softmax);
let accuracyUntrained = tester(untrainedBrain, testAmount);
accuracyTable.push(["Untrained", "0", accuracyUntrained.toFixed(2)]);

// Test brain trained with 20,000 samples
console.log("Testing brain20000...");
let trainedBrain = new JellyBrain(784, 784, 10, costFuncs.crossEntropy, 0.0008, activationFuncs.sigmoid, activationFuncs.softmax);
loadBrain(trainedBrain, "brain20000");
let accuracy20k = tester(trainedBrain, testAmount);
accuracyTable.push(["brain20000", "20,000", accuracy20k.toFixed(2)]);

// Test brain trained with 60,000 samples (full training set)
console.log("Testing brain60000...");
loadBrain(trainedBrain, "brain60000");
let accuracy60k = tester(trainedBrain, testAmount);
accuracyTable.push(["brain60000", "60,000", accuracy60k.toFixed(2)]);

console.table(accuracyTable);

// Summary
console.log("\n📊 Summary:");
let improvement = accuracy60k - accuracyUntrained;
let percentImprovement = improvement / accuracyUntrained * 100;
let success = accuracy60k > 90;

console.log(`Untrained Accuracy: ${accuracyUntrained.toFixed(2)}%`);
console.log(`Final Accuracy (60k samples): ${accuracy60k.toFixed(2)}%`);
console.log(`Improvement: ${improvement.toFixed(2)}% (${percentImprovement.toFixed(1)}% relative improvement)`);
console.log(`Success (Accuracy > 90%): ${success ? '✅' : '❌'}`);
