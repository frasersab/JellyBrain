const math = require('mathjs')
const {JellyBrain, linear} = require('./JellyBrain.js')
const {readMNIST, saveMNIST} = require('./mnist_dataset/mnist_reader')


let brain = new JellyBrain(784, 784, 10);

// training function
function trainer(brain, amount) {
    var pixelValues = readMNIST(0, amount, '\\train_images_60k.idx3-ubyte', '\\train_labels_60k.idx1-ubyte');

    pixelValues.forEach(function(image)
    {
        targetArray = new Array(10).fill(0);
        targetArray[image.label] = 1;
        brain.train(image.pixels, targetArray);
    })
}

// training function
function tester(brain, amount) {
    var pixelValues = readMNIST(0, amount, '\\test_images_10k.idx3-ubyte', '\\test_labels_10k.idx1-ubyte');
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
    })

    return accuracy = (accuracy / amount) * 100;
}

let accuracyTable = Array();
accuracyTable.push(["Training Samples", "Accuracy"]);

accuracyTable.push([0, tester(brain, 20)]);
trainer(brain, 20);
accuracyTable.push([20, tester(brain, 20)]);

console.table(accuracyTable);