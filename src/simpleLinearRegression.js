// This tests the neural networks ability to predict a simple linear regression

import {JellyBrain, tanh, linear} from './JellyBrain.js'

let brain = new JellyBrain(1, 1, 1, undefined, undefined, linear);

// line function
function y(x)
{
    return 2 * x;
}

// training function
function trainer(brain, amount)
{
    let inputs = Array(amount);
    let outputs = Array(amount);

    for (let i = 0; i < amount; i++)
    {
        inputs[i] = [Math.random()];
        outputs[i] = y(inputs[i])
        brain.train(inputs[i], outputs[i]);
    }
}

// testing function
function tester(brain, amount)
{
    let inputs = Array(amount);
    let outputs = Array(amount);
    let guess = Array(amount);
    let matrix = Array(amount);

    for (let i = 0; i < amount; i++)
    {
        inputs[i] = [Math.random()];
        outputs[i] = y(inputs[i]);
        guess[i] = brain.guess(inputs[i])[0];
        matrix[i] = [outputs[i], guess[i]];
    }

    console.table(matrix);
    // TODO: calculate R squared
    console.log(`R Squared:`);
}

trainer(brain, 10000);
tester(brain, 10);
