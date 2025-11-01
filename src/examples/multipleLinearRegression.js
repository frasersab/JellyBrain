// This tests the neural networks ability to predict a multiple linear regression

const math = require('mathjs')
const {JellyBrain, sigmoid} = require('../JellyBrain.js')

let brain = new JellyBrain(2, 10, 1, undefined, 0.5, sigmoid, sigmoid);

// line function
function y(a, b) {
    return (2 * a) + (3 * b);
}

// normalize output for sigmoid (maps to ~[0,1] range)
function normalize(value) {
    return (value + 1) / 7;  // maps [-1, 5] to [0, 0.86]
}

// denormalize for comparison
function denormalize(value) {
    return value * 7 - 1;
}

// TODO: get adjusted r squared
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
        inputs[i] = [Math.random(), Math.random()];
        targets[i] = [normalize(y(inputs[i][0], inputs[i][1]))]
        brain.train(inputs[i], targets[i]);
    }
}

// testing function
function tester(brain, amount) {
    let inputs = Array(amount);
    let targets = Array(amount);
    let guess = Array(amount);

    for (let i = 0; i < amount; i++) {
        inputs[i] = [Math.random(), Math.random()];
        targets[i] = y(inputs[i][0], inputs[i][1])
        guess[i] = denormalize(brain.guess(inputs[i])[0]);
    }

    return rsquared(targets, guess);
}

console.log("Multiple Linear Regression Test (y = 2a + 3b)");
console.log("==============================================\n");

let accuracyTable = Array();
accuracyTable.push(["Training Samples", "R Squared"]);

let initialAccuracy = tester(brain, 10000);
accuracyTable.push([0, initialAccuracy.toFixed(6)]);

trainer(brain, 200);
accuracyTable.push([200, tester(brain, 10000).toFixed(6)]);

trainer(brain, 800);
accuracyTable.push([1000, tester(brain, 10000).toFixed(6)]);

trainer(brain, 2000);
accuracyTable.push([3000, tester(brain, 10000).toFixed(6)]);

trainer(brain, 3000);
accuracyTable.push([6000, tester(brain, 10000).toFixed(6)]);

trainer(brain, 4000);
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
