const { JellyBrain, costFuncs, activationFuncs } = require('../JellyBrain');

describe('Learning: Binary Classification', () => {
  // Helper function to generate data for line classification
  function generateLineData(amount, lineFunc) {
    const data = [];
    for (let i = 0; i < amount; i++) {
      const x = Math.random();
      const y = Math.random();
      const target = lineFunc(x) > y ? [0] : [1]; // Below line = 0, above = 1
      data.push({ input: [x, y], target });
    }
    return data;
  }

  // Helper function to calculate accuracy
  function calculateAccuracy(brain, testData) {
    let correct = 0;
    testData.forEach(({ input, target }) => {
      const output = brain.guess(input);
      const prediction = Math.round(output[0]);
      if (prediction === target[0]) {
        correct++;
      }
    });
    return (correct / testData.length) * 100;
  }

  describe('Linear Boundary (y = x)', () => {
    test('learns to classify points above/below y=x line', () => {
      const brain = new JellyBrain(
        2, 4, 1,
        costFuncs.errorSquared,
        0.1,
        activationFuncs.sigmoid,
        activationFuncs.sigmoid
      );

      const lineFunc = (x) => x;

      // Initial accuracy should be around 50% (random)
      const initialTestData = generateLineData(1000, lineFunc);
      const initialAccuracy = calculateAccuracy(brain, initialTestData);
      expect(initialAccuracy).toBeGreaterThanOrEqual(30);
      expect(initialAccuracy).toBeLessThanOrEqual(70);

      // Train
      for (let i = 0; i < 4000; i++) {
        const trainingData = generateLineData(1, lineFunc);
        brain.train(trainingData[0].input, trainingData[0].target);
      }

      // Final accuracy should be high
      const finalTestData = generateLineData(1000, lineFunc);
      const finalAccuracy = calculateAccuracy(brain, finalTestData);
      expect(finalAccuracy).toBeGreaterThan(85);
    }, 10000);

    test('accuracy improves with more training', () => {
      const brain = new JellyBrain(2, 4, 1, costFuncs.errorSquared, 0.1);
      const lineFunc = (x) => x;

      const accuracies = [];
      const testData = generateLineData(1000, lineFunc);

      // Record accuracy at different training stages
      accuracies.push(calculateAccuracy(brain, testData));

      for (let stage = 0; stage < 5; stage++) {
        // Train for 500 samples
        for (let i = 0; i < 1000; i++) {
          const trainingData = generateLineData(1, lineFunc);
          brain.train(trainingData[0].input, trainingData[0].target);
        }
        accuracies.push(calculateAccuracy(brain, testData));
      }

      // Accuracy should generally improve
      expect(accuracies[5]).toBeGreaterThan(accuracies[0]);
      expect(accuracies[5]).toBeGreaterThan(85);
    }, 10000);
  });

  describe('Linear Boundary (y = 0.5x + 0.2)', () => {
    test('learns to classify points with non-trivial line', () => {
      const brain = new JellyBrain(
        2, 5, 1,
        costFuncs.errorSquared,
        0.1,
        activationFuncs.sigmoid,
        activationFuncs.sigmoid
      );

      const lineFunc = (x) => 0.5 * x + 0.2;

      // Train
      for (let i = 0; i < 4000; i++) {
        const trainingData = generateLineData(1, lineFunc);
        brain.train(trainingData[0].input, trainingData[0].target);
      }

      // Test
      const testData = generateLineData(1000, lineFunc);
      const accuracy = calculateAccuracy(brain, testData);
      expect(accuracy).toBeGreaterThan(85);
    }, 10000);
  });

  describe('Vertical Boundary (x = 0.5)', () => {
    test('learns to classify points left/right of vertical line', () => {
      const brain = new JellyBrain(
        2, 4, 1,
        costFuncs.errorSquared,
        0.1,
        activationFuncs.sigmoid,
        activationFuncs.sigmoid
      );

      // Generate data for vertical line at x = 0.5
      const generateVerticalData = (amount) => {
        const data = [];
        for (let i = 0; i < amount; i++) {
          const x = Math.random();
          const y = Math.random();
          const target = x < 0.5 ? [0] : [1];
          data.push({ input: [x, y], target });
        }
        return data;
      };

      // Train
      for (let i = 0; i < 4000; i++) {
        const trainingData = generateVerticalData(1);
        brain.train(trainingData[0].input, trainingData[0].target);
      }

      // Test
      const testData = generateVerticalData(1000);
      const accuracy = calculateAccuracy({ guess: (input) => brain.guess(input) }, testData);
      expect(accuracy).toBeGreaterThan(85);
    }, 10000);
  });
});