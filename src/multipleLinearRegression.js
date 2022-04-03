// This tests the neural networks ability to predict a simple linear regression

import * as math from 'mathjs';
import {JellyBrain, tanh, linear} from './JellyBrain.js'

let brain = new JellyBrain(2, 2, 1, undefined, undefined, linear);

// line function
function y(a , b)
{
    return (2 * a) + (3 * b);
}

// todo: get adjusted r squared
// r squared function
function rsquared(actual, guess)
{
    const avarage = actual.reduce((prev, curr) => prev + curr) / actual.length;

    let sstot = 0;
    actual.forEach((value) =>
    {
        sstot += math.square(value - avarage);
    })

    let ssres = 0;
    actual.forEach((value, index) =>
    {
        ssres += math.square(value - guess[index]);
    })

    return 1 - (ssres / sstot);
}

// training function
function trainer(brain, amount)
{
    let inputs = Array(amount);
    let outputs = Array(amount);

    for (let i = 0; i < amount; i++)
    {
        inputs[i] = [Math.random(), Math.random()];
        outputs[i] = y(inputs[i][0], inputs[i][1])
        brain.train(inputs[i], outputs[i]);
    }
}

// testing function
function tester(brain, amount)
{
    let inputs = Array(amount);
    let outputs = Array(amount);
    let guess = Array(amount);

    for (let i = 0; i < amount; i++)
    {
        inputs[i] = [Math.random(), Math.random()];
        outputs[i] = y(inputs[i][0], inputs[i][1])
        guess[i] = brain.guess(inputs[i])[0];
    }

    return rsquared(outputs, guess);
}

let accuracyTable = Array();
accuracyTable.push(["Training Samples", "R Squared"]);

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