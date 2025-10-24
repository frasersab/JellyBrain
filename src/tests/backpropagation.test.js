const { JellyBrain, costFuncs, activationFuncs } = require('../JellyBrain');
const math = require('mathjs');

describe('Backpropagation and Training', () => {
  describe('Input/Target Validation', () => {
    test('validates input length during training', () => {
      const brain = new JellyBrain(2, 3, 2);
      const initialWeights = brain.getWeightsIH();
      
      brain.train([1], [0, 1]); // Wrong input length
      
      // Weights should not change due to validation failure
      const finalWeights = brain.getWeightsIH();
      expect(finalWeights).toEqual(initialWeights);
    });

    test('validates target length during training', () => {
      const brain = new JellyBrain(2, 3, 2);
      const initialWeights = brain.getWeightsHO();
      
      brain.train([1, 0], [0]); // Wrong target length
      
      const finalWeights = brain.getWeightsHO();
      expect(finalWeights).toEqual(initialWeights);
    });

    test('validates both input and target in addToBatch', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      brain.addToBatch([1], [0, 1]); // Wrong input
      expect(brain.getBatchSize()).toBe(0);
      
      brain.addToBatch([1, 0], [0]); // Wrong target
      expect(brain.getBatchSize()).toBe(0);
      
      brain.addToBatch([1, 0], [0, 1]); // Correct
      expect(brain.getBatchSize()).toBe(1);
    });
  });

  describe('Weight Updates', () => {
    test('single training step updates all weights and biases', () => {
      const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
      
      const initialWeightsIH = math.clone(brain.getWeightsIH());
      const initialWeightsHO = math.clone(brain.getWeightsHO());
      const initialBiasIH = [...brain.getBiasIH()];
      const initialBiasHO = [...brain.getBiasHO()];
      
      brain.train([0.5, 0.7], [1, 0]);
      
      const finalWeightsIH = brain.getWeightsIH();
      const finalWeightsHO = brain.getWeightsHO();
      const finalBiasIH = brain.getBiasIH();
      const finalBiasHO = brain.getBiasHO();
      
      // Check that weights changed
      let weightsIHChanged = false;
      for (let i = 0; i < initialWeightsIH.length; i++) {
        for (let j = 0; j < initialWeightsIH[i].length; j++) {
          if (Math.abs(initialWeightsIH[i][j] - finalWeightsIH[i][j]) > 1e-10) {
            weightsIHChanged = true;
            break;
          }
        }
      }
      expect(weightsIHChanged).toBe(true);
      
      let weightsHOChanged = false;
      for (let i = 0; i < initialWeightsHO.length; i++) {
        for (let j = 0; j < initialWeightsHO[i].length; j++) {
          if (Math.abs(initialWeightsHO[i][j] - finalWeightsHO[i][j]) > 1e-10) {
            weightsHOChanged = true;
            break;
          }
        }
      }
      expect(weightsHOChanged).toBe(true);
      
      // Check that biases changed
      const biasIHChanged = initialBiasIH.some((val, idx) => 
        Math.abs(val - finalBiasIH[idx]) > 1e-10
      );
      expect(biasIHChanged).toBe(true);
      
      const biasHOChanged = initialBiasHO.some((val, idx) => 
        Math.abs(val - finalBiasHO[idx]) > 1e-10
      );
      expect(biasHOChanged).toBe(true);
    });

    test('learning rate affects weight update magnitude', () => {
      const brain1 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.01);
      const brain2 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
      
      // Make them identical
      const exported = brain1.exportBrain();
      brain2.importBrain(exported);
      
      const initialWeights = brain1.getWeightsIH();
      
      brain1.train([0.5, 0.7], [1, 0]);
      brain2.train([0.5, 0.7], [1, 0]);
      
      const weights1 = brain1.getWeightsIH();
      const weights2 = brain2.getWeightsIH();
      
      // Calculate total change
      let change1 = 0, change2 = 0;
      for (let i = 0; i < initialWeights.length; i++) {
        for (let j = 0; j < initialWeights[i].length; j++) {
          change1 += Math.abs(weights1[i][j] - initialWeights[i][j]);
          change2 += Math.abs(weights2[i][j] - initialWeights[i][j]);
        }
      }
      
      expect(change2).toBeGreaterThan(change1 * 5); // Roughly 10x difference
    });
  });

  describe('Error Reduction', () => {
    test('training reduces error for simple patterns', () => {
      const brain = new JellyBrain(2, 4, 1, costFuncs.errorSquared, 0.1);
      
      const input = [0.5, 0.5];
      const target = [1];
      
      const initialOutput = brain.guess(input);
      const initialError = Math.pow(initialOutput[0] - target[0], 2);
      
      // Train multiple times
      for (let i = 0; i < 100; i++) {
        brain.train(input, target);
      }
      
      const finalOutput = brain.guess(input);
      const finalError = Math.pow(finalOutput[0] - target[0], 2);
      
      expect(finalError).toBeLessThan(initialError);
      expect(finalError).toBeLessThan(0.1);
    });

    test('can learn identity function', () => {
      const brain = new JellyBrain(2, 4, 2, costFuncs.errorSquared, 0.1);
      
      const trainingData = [
        { input: [0.2, 0.8], target: [0.2, 0.8] },
        { input: [0.7, 0.3], target: [0.7, 0.3] },
        { input: [0.5, 0.5], target: [0.5, 0.5] },
        { input: [0.9, 0.1], target: [0.9, 0.1] }
      ];
      
      for (let epoch = 0; epoch < 4000; epoch++) {
        for (const data of trainingData) {
          brain.train(data.input, data.target);
        }
      }
      
      // Test on training data
      trainingData.forEach(data => {
        const output = brain.guess(data.input);
        expect(output[0]).toBeCloseTo(data.target[0], 0);
        expect(output[1]).toBeCloseTo(data.target[1], 0);
      });
    });

    test('can learn AND logic gate', () => {
      const brain = new JellyBrain(2, 3, 1, costFuncs.errorSquared, 0.5);
      
      const trainingData = [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [0] },
        { input: [1, 0], target: [0] },
        { input: [1, 1], target: [1] }
      ];
      
      for (let epoch = 0; epoch < 2000; epoch++) {
        for (const data of trainingData) {
          brain.train(data.input, data.target);
        }
      }
      
      expect(brain.guess([0, 0])[0]).toBeLessThan(0.2);
      expect(brain.guess([0, 1])[0]).toBeLessThan(0.2);
      expect(brain.guess([1, 0])[0]).toBeLessThan(0.2);
      expect(brain.guess([1, 1])[0]).toBeGreaterThan(0.8);
    });

    test('can learn OR logic gate', () => {
      const brain = new JellyBrain(2, 3, 1, costFuncs.errorSquared, 0.5);
      
      const trainingData = [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [1] }
      ];
      
      for (let epoch = 0; epoch < 500; epoch++) {
        for (const data of trainingData) {
          brain.train(data.input, data.target);
        }
      }
      
      expect(brain.guess([0, 0])[0]).toBeLessThan(0.2);
      expect(brain.guess([0, 1])[0]).toBeGreaterThan(0.8);
      expect(brain.guess([1, 0])[0]).toBeGreaterThan(0.8);
      expect(brain.guess([1, 1])[0]).toBeGreaterThan(0.8);
    });
  });

  describe('Batch Training', () => {
    test('batch size increments correctly', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      expect(brain.getBatchSize()).toBe(0);
      brain.addToBatch([1, 0], [0, 1]);
      expect(brain.getBatchSize()).toBe(1);
      brain.addToBatch([0, 1], [1, 0]);
      expect(brain.getBatchSize()).toBe(2);
    });

    test('computeBatch updates weights', () => {
      const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
      const initialWeights = math.clone(brain.getWeightsIH());
      
      brain.addToBatch([1, 0], [1, 0]);
      brain.addToBatch([0, 1], [0, 1]);
      brain.computeBatch();
      
      const finalWeights = brain.getWeightsIH();
      
      let changed = false;
      for (let i = 0; i < initialWeights.length; i++) {
        for (let j = 0; j < initialWeights[i].length; j++) {
          if (Math.abs(initialWeights[i][j] - finalWeights[i][j]) > 1e-10) {
            changed = true;
            break;
          }
        }
      }
      expect(changed).toBe(true);
    });

    test('computeBatch resets batch size', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      brain.addToBatch([1, 0], [1, 0]);
      brain.addToBatch([0, 1], [0, 1]);
      expect(brain.getBatchSize()).toBe(2);
      
      brain.computeBatch();
      expect(brain.getBatchSize()).toBe(0);
    });

    test('clearBatch resets without updating', () => {
      const brain = new JellyBrain(2, 3, 2);
      const initialWeights = math.clone(brain.getWeightsIH());
      
      brain.addToBatch([1, 0], [1, 0]);
      brain.addToBatch([0, 1], [0, 1]);
      brain.clearBatch();
      
      const finalWeights = brain.getWeightsIH();
      expect(finalWeights).toEqual(initialWeights);
      expect(brain.getBatchSize()).toBe(0);
    });

    test('gradients accumulate in batch', () => {
      const brain = new JellyBrain(2, 3, 1, costFuncs.errorSquared, 0.1);
      
      brain.addToBatch([1, 0], [1]);
      const change1 = math.clone(brain.getWeightsIHChange());
      
      brain.addToBatch([0, 1], [0]);
      const change2 = brain.getWeightsIHChange();
      
      // Gradients should have accumulated (changed)
      let accumulated = false;
      for (let i = 0; i < change1.length; i++) {
        for (let j = 0; j < change1[i].length; j++) {
          if (Math.abs(change1[i][j] - change2[i][j]) > 1e-10) {
            accumulated = true;
            break;
          }
        }
      }
      expect(accumulated).toBe(true);
    });

    test('train method clears batch before training', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      brain.addToBatch([1, 0], [1, 0]);
      expect(brain.getBatchSize()).toBe(1);
      
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      brain.train([0, 1], [0, 1]);
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });
  });

  describe('Different Activation and Cost Combinations', () => {
    test('softmax with cross entropy trains correctly', () => {
      const brain = new JellyBrain(
        2, 4, 3,
        costFuncs.crossEntropy,
        0.1,
        activationFuncs.sigmoid,
        activationFuncs.softmax
      );
      
      const target = [0, 1, 0]; // Class 2
      const input = [0.5, 0.8];
      
      const initialOutput = brain.guess(input);
      const initialProb = initialOutput[1];
      
      for (let i = 0; i < 100; i++) {
        brain.train(input, target);
      }
      
      const finalOutput = brain.guess(input);
      const finalProb = finalOutput[1];
      
      expect(finalProb).toBeGreaterThan(initialProb);
      expect(finalProb).toBeGreaterThan(finalOutput[0]);
      expect(finalProb).toBeGreaterThan(finalOutput[2]);
    });

    test('ReLU hidden layer trains effectively', () => {
      const brain = new JellyBrain(
        2, 5, 1,
        costFuncs.errorSquared,
        0.1,
        activationFuncs.relu,
        activationFuncs.sigmoid
      );
      
      const trainingData = [
        { input: [0, 0], target: [0] },
        { input: [1, 1], target: [1] }
      ];
      
      for (let epoch = 0; epoch < 200; epoch++) {
        for (const data of trainingData) {
          brain.train(data.input, data.target);
        }
      }
      
      expect(brain.guess([0, 0])[0]).toBeLessThan(0.3);
      expect(brain.guess([1, 1])[0]).toBeGreaterThan(0.7);
    });

    test('linear output for regression tasks', () => {
      const brain = new JellyBrain(
        1, 4, 1,
        costFuncs.errorSquared,
        0.01,
        activationFuncs.tanh,
        activationFuncs.linear
      );
      
      // Learn f(x) = 2x
      const trainingData = [];
      for (let i = 0; i <= 10; i++) {
        const x = i / 10;
        trainingData.push({ input: [x], target: [2 * x] });
      }
      
      for (let epoch = 0; epoch < 500; epoch++) {
        for (const data of trainingData) {
          brain.train(data.input, data.target);
        }
      }
      
      // Test
      expect(brain.guess([0.5])[0]).toBeCloseTo(1.0, 1);
      expect(brain.guess([0.25])[0]).toBeCloseTo(0.5, 1);
    });
  });

  describe('Convergence Behavior', () => {
    test('repeated training on same example converges', () => {
      const brain = new JellyBrain(2, 3, 1, costFuncs.errorSquared, 0.1);
      
      const input = [0.5, 0.5];
      const target = [0.8];
      
      const errors = [];
      for (let i = 0; i < 200; i++) {
        brain.train(input, target);
        const output = brain.guess(input);
        errors.push(Math.abs(output[0] - target[0]));
      }
      
      // Error should decrease over time
      expect(errors[199]).toBeLessThan(errors[0]);
      expect(errors[199]).toBeLessThan(0.1);
      
      // Check that error generally decreases (allowing for some fluctuation)
      let decreasing = 0;
      for (let i = 10; i < errors.length; i += 10) {
        if (errors[i] < errors[i - 10]) decreasing++;
      }
      expect(decreasing).toBeGreaterThan(15); // Most intervals should show decrease
    });

    test('batch training produces smooth convergence', () => {
      const brain = new JellyBrain(2, 4, 1, costFuncs.errorSquared, 0.1);
      
      const trainingData = [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [0] }
      ];
      
      const errors = [];
      
      for (let epoch = 0; epoch < 50; epoch++) {
        // Add all examples to batch
        for (const data of trainingData) {
          brain.addToBatch(data.input, data.target);
        }
        brain.computeBatch();
        
        // Calculate average error
        let totalError = 0;
        for (const data of trainingData) {
          const output = brain.guess(data.input);
          totalError += Math.pow(output[0] - data.target[0], 2);
        }
        errors.push(totalError / trainingData.length);
      }
      
      // Error should generally decrease
      expect(errors[49]).toBeLessThan(errors[0]);
    });
  });
});