// This tests the neural networks ability to classify if a point is above or bellow a line

const { JellyBrain } = require('../JellyBrain')

let brain = new JellyBrain(2, 1, 1);

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

let accuracyTable = Array();
accuracyTable.push(["Training Samples", "Accuracy"]);

accuracyTable.push([0, tester(brain, 10000)]);
trainer(brain, 10);
accuracyTable.push([10, tester(brain, 10000)]);
trainer(brain, 90);
accuracyTable.push([100, tester(brain, 10000)]);
trainer(brain, 400);
accuracyTable.push([500, tester(brain, 10000)]);
trainer(brain, 500);
accuracyTable.push([1000, tester(brain, 10000)]);

console.table(accuracyTable);
