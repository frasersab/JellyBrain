const JellyBrain = require('./JellyBrain');

brain = new JellyBrain(2, 2, 1);

// --Training--
// create inputs/targets
let inputs = Array(1000);
let targets = Array(inputs.length);
for (let i = 0; i < inputs.length; i++) {
    // create random inputs between 0 and 1
    inputs[i] = [Math.random(), Math.random()];
    // if input is below line then 0, if above then 1
    if (inputs[i][0] > inputs[i][1]) {
        targets[i] = 1;
    } else {
        targets[i] = 0;
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
    if (inputs2[i][0] > inputs2[i][1]) {
        targets2[i] = 1;
    } else {
        targets2[i] = 0;
    }
    //train brain
    //console.log(Math.round(brain.guess(inputs2[i])), brain.guess(inputs2[i], targets2[i]), targets[i]);
    if (brain.guess(inputs2[i]) - targets2[i] < 0.5) {
        accuracy++;
    }

}

// calculate and log accuracy
accuracy = accuracy / (inputs2.length / 100);

console.log(accuracy + '% Accurate');