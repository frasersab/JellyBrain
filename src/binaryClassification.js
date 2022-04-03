// This tests the neural networks ability to classify if a point is above or bellow a line

const JellyBrain = require('./JellyBrain');
brain = new JellyBrain(2, 1, 1);

// line function
function y(x)
{
    return x;
}

// training function
function trainer(brain, amount)
{
    let inputs = Array(amount);
    let outputs = Array(amount);

    for (let i = 0; i < amount; i++)
    {
        inputs[i] = [Math.random(), Math.random()];
        // if input is below line then 0, if above then 1
        if (y(inputs[i][0]) > inputs[i][1])
        {
            outputs[i] = 0;
        } 
        else
        {
            outputs[i] = 1;
        }
        brain.train(inputs[i], outputs[i]);
    }
}

// testing function
function tester(brain, amount)
{
    let inputs = Array(amount);
    let outputs = Array(amount);
    let correct = Array(amount);
    let accuracy = 0;

    for (let i = 0; i < amount; i++)
    {
        inputs[i] = [Math.random(), Math.random()];
        // if input is below line then 0, if above then 1
        if (y(inputs[i][0]) > inputs[i][1])
        {
            outputs[i] = 0;
        } 
        else
        {
            outputs[i] = 1;
        }
        let rawguess = brain.guess(inputs[i]);
        let guess = Math.round(brain.guess(inputs[i]));
        let answer = outputs[i];
        if (guess == outputs[i])
        {
            correct[i] = 1;
            accuracy++;
        }
        else
        {
            correct[i] = 0;
        }
    }
    accuracy = accuracy / (amount / 100);
    console.log(accuracy + "% Accurate")
}

//trainer(brain, 10);
tester(brain, 1000);
