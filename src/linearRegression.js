const JellyBrain = require('./JellyBrain');

brain = new JellyBrain(2, 1, 1);

// Line function
function y(x) {
    return x;
}

// --Training--
// create inputs/targets
let inputs = Array(2000);
let targets = Array(inputs.length);
for (let i = 0; i < inputs.length; i++) {
    // create random inputs between 0 and 1
    inputs[i] = [Math.random(), Math.random()];
    // if input is below line then 0, if above then 1
    if (y(inputs[i][0]) < inputs[i][1]) {
        targets[i] = 0;
    } else {
        targets[i] = 1;
    }
    //train brain
    //console.log(brain.train(inputs[i], targets[i]));
    brain.train(inputs[i], targets[i]);
}

// --Testing--
// create inputs/targets
let inputs2 = Array(100);
let targets2 = Array(inputs2.length);
let accuracy = 0;
for (let i = 0; i < inputs2.length; i++) {
    // create random inputs between 0 and 1
    inputs2[i] = [Math.random(), Math.random()];
    // if input is below line then 0, if above then 1
    if (y(inputs2[i][0]) < inputs2[i][1]) {
        targets2[i] = 0;
    } else {
        targets2[i] = 1;
    }
    //train brain
    //console.log(Math.round(brain.guess(inputs[i])), targets2[i], Math.round(brain.guess(inputs[i])) == targets2[i]);
    if (Math.round(brain.guess(inputs[i])) == targets2[i]) {
        accuracy++;
    }

}

accuracy = accuracy / (inputs2.length / 100);

console.log(accuracy);