// This tests the neural networks ability to classify if a point is above or below a line

const { JellyBrain } = require('../JellyBrain')

let brain = new JellyBrain(2, 5, 1);
brain.setLearningRate(0.1);

// line function
function y(x) {
    return x;
}

// training function
function trainer(brain, amount) {
    let inputs = Array(amount);
    let targets = Array(amount);

    for (let i = 0; i < amount; i++) {
        inputs[i] = [Math.random(), Math.random()];
        // if input is below line then 0, if above then 1
        if (y(inputs[i][0]) > inputs[i][1]) {
            targets[i] = [0];
        }
        else {
            targets[i] = [1];
        }
        brain.train(inputs[i], targets[i]);
    }
}

// testing function
function tester(brain, amount) {
    let inputs = Array(amount);
    let targets = Array(amount);
    let guess = Array(amount);
    let accuracy = 0;

    for (let i = 0; i < amount; i++) {
        inputs[i] = [Math.random(), Math.random()];
        // if input is below line then 0, if above then 1
        if (y(inputs[i][0]) > inputs[i][1]) {
            targets[i] = [0];
        }
        else {
            targets[i] = [1];
        }

        guess[i] = Math.round(brain.guess(inputs[i])[0]);

        if (guess[i] == targets[i][0]) {
            accuracy++;
        }
    }
    return accuracy = (accuracy / amount) * 100;
}

console.log("Binary Classification Test (Above/Below line: y = x)");
console.log("=====================================================\n");

let accuracyTable = Array();
accuracyTable.push(["Training Samples", "Accuracy (%)"]);

let initialAccuracy = tester(brain, 10000);
accuracyTable.push([0, initialAccuracy.toFixed(2)]);

trainer(brain, 1000);
accuracyTable.push([1000, tester(brain, 10000).toFixed(2)]);

trainer(brain, 1000);
accuracyTable.push([2000, tester(brain, 10000).toFixed(2)]);

trainer(brain, 2000);
accuracyTable.push([4000, tester(brain, 10000).toFixed(2)]);

trainer(brain, 2000);
accuracyTable.push([6000, tester(brain, 10000).toFixed(2)]);

trainer(brain, 2000);
let finalAccuracy = tester(brain, 10000);
accuracyTable.push([8000, finalAccuracy.toFixed(2)]);

console.table(accuracyTable);

// Summary
console.log("\n📊 Summary:");
let improvement = finalAccuracy - initialAccuracy;
let percentImprovement = improvement / initialAccuracy * 100;
let success = finalAccuracy > 95;

console.log(`Initial Accuracy: ${initialAccuracy.toFixed(2)}%`);
console.log(`Final Accuracy: ${finalAccuracy.toFixed(2)}%`);
console.log(`Improvement: ${improvement.toFixed(2)}% (${percentImprovement.toFixed(1)}% relative improvement)`);
console.log(`Success (Accuracy > 95%): ${success ? '✅' : '❌'}`);
