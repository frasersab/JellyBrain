// This tests the neural networks ability to predict a simple linear regression

const math = require('mathjs')
const {JellyBrain, sigmoid} = require('../JellyBrain.js')

let brain = new JellyBrain(1, 8, 1, undefined, 0.5, sigmoid, sigmoid);

// line function
function y(x) {
    return 2 * x;
}

// normalize output for sigmoid (maps to ~[0,1] range)
function normalize(value) {
    return (value + 1) / 3;  // maps [-1, 2] to [0, 1]
}

// denormalize for comparison
function denormalize(value) {
    return value * 3 - 1;
}

// r squared function
function rsquared(actual, guess) {
    const avarage = actual.reduce((prev, curr) => prev + curr) / actual.length;

    let sstot = 0;
    actual.forEach((value) => {
        sstot += math.square(value - avarage);
    })

    let ssres = 0;
    actual.forEach((value, index) => {
        ssres += math.square(value - guess[index]);
    })

    return 1 - (ssres / sstot);
}

// training function
function trainer(brain, amount) {
    let inputs = Array(amount);
    let targets = Array(amount);

    for (let i = 0; i < amount; i++) {
        inputs[i] = [Math.random()];
        targets[i] = [normalize(y(inputs[i][0]))]
        brain.train(inputs[i], targets[i]);
    }
}

// testing function
function tester(brain, amount) {
    let inputs = Array(amount);
    let targets = Array(amount);
    let guess = Array(amount);

    for (let i = 0; i < amount; i++) {
        inputs[i] = [Math.random()];
        targets[i] = y(inputs[i][0]);
        guess[i] = denormalize(brain.guess(inputs[i])[0]);
    }

    return rsquared(targets, guess);
}

console.log("Simple Linear Regression Test (y = 2x)");
console.log("=======================================\n");

let accuracyTable = Array();
accuracyTable.push(["Training Samples", "R Squared"]);

let initialAccuracy = tester(brain, 10000);
accuracyTable.push([0, initialAccuracy.toFixed(6)]);

trainer(brain, 100);
accuracyTable.push([100, tester(brain, 10000).toFixed(6)]);

trainer(brain, 400);
accuracyTable.push([500, tester(brain, 10000).toFixed(6)]);

trainer(brain, 1500);
accuracyTable.push([2000, tester(brain, 10000).toFixed(6)]);

trainer(brain, 3000);
accuracyTable.push([5000, tester(brain, 10000).toFixed(6)]);

trainer(brain, 5000);
let finalAccuracy = tester(brain, 10000);
accuracyTable.push([10000, finalAccuracy.toFixed(6)]);

console.table(accuracyTable);

// Summary
console.log("\n📊 Summary:");
let improvement = finalAccuracy - initialAccuracy;
let percentImprovement = Math.abs(improvement / Math.abs(initialAccuracy) * 100);
let success = finalAccuracy > 0.90;

console.log(`Initial R²: ${initialAccuracy.toFixed(6)}`);
console.log(`Final R²: ${finalAccuracy.toFixed(6)}`);
console.log(`Improvement: ${improvement.toFixed(6)} (${percentImprovement.toFixed(1)}%)`);
console.log(`Success (R² > 0.90): ${success ? '✅' : '❌'}`);
