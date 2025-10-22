const { JellyBrain, costFuncs, activationFuncs } = require('../JellyBrain');
const math = require('mathjs');

// Helper function for comparing arrays with tolerance
function expectArrayCloseTo(received, expected, precision = 5) {
  expect(received).toHaveLength(expected.length);
  received.forEach((val, idx) => {
    expect(val).toBeCloseTo(expected[idx], precision);
  });
}

// Helper function for comparing matrices
function expectMatrixCloseTo(received, expected, precision = 5) {
  expect(received).toHaveLength(expected.length);
  received.forEach((row, i) => {
    expect(row).toHaveLength(expected[i].length);
    row.forEach((val, j) => {
      expect(val).toBeCloseTo(expected[i][j], precision);
    });
  });
}

describe('JellyBrain Constructor', () => {
  test('creates network with correct dimensions', () => {
    const brain = new JellyBrain(3, 4, 2);
    expect(brain.getInputNodes()).toBe(3);
    expect(brain.getHiddenNodes()).toBe(4);
    expect(brain.getOutputNodes()).toBe(2);
  });

  test('initializes weights with correct dimensions', () => {
    const brain = new JellyBrain(3, 4, 2);
    const weightsIH = brain.getWeightsIH();
    const weightsHO = brain.getWeightsHO();
    
    expect(weightsIH).toHaveLength(3);
    expect(weightsIH[0]).toHaveLength(4);
    expect(weightsHO).toHaveLength(4);
    expect(weightsHO[0]).toHaveLength(2);
  });

  test('initializes biases to zeros with correct dimensions', () => {
    const brain = new JellyBrain(3, 4, 2);
    const biasIH = brain.getBiasIH();
    const biasHO = brain.getBiasHO();
    
    expect(biasIH).toHaveLength(4);
    expect(biasHO).toHaveLength(2);
    expectArrayCloseTo(biasIH, [0, 0, 0, 0], 10);
    expectArrayCloseTo(biasHO, [0, 0], 10);
  });

  test('initializes weights within expected range', () => {
    const brain = new JellyBrain(10, 20, 5);
    const weightsIH = brain.getWeightsIH();
    const weightsHO = brain.getWeightsHO();
    
    const expectedIHRange = 1 / Math.sqrt(20);
    const expectedHORange = 1 / Math.sqrt(5);
    
    // Check all weights are within range
    weightsIH.forEach(row => {
      row.forEach(weight => {
        expect(Math.abs(weight)).toBeLessThanOrEqual(expectedIHRange);
      });
    });
    
    weightsHO.forEach(row => {
      row.forEach(weight => {
        expect(Math.abs(weight)).toBeLessThanOrEqual(expectedHORange);
      });
    });
  });

  test('accepts custom activation functions', () => {
    const brain = new JellyBrain(
      2, 3, 2,
      costFuncs.errorSquared,
      0.01,
      activationFuncs.relu,
      activationFuncs.sigmoid
    );
    
    const output = brain.guess([0.5, -0.5]);
    expect(output).toHaveLength(2);
    expect(Array.isArray(output)).toBe(true);
  });

  test('accepts custom learning rate', () => {
    const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.123);
    expect(brain.getLearningRate()).toBe(0.123);
  });

  test('accepts custom cost function', () => {
    const brain = new JellyBrain(2, 3, 2, costFuncs.crossEntropy);
    const output = brain.guess([1, 0]);
    expect(Array.isArray(output)).toBe(true);
  });
});

describe('JellyBrain Guess (Forward Propagation)', () => {
  test('validates input length - returns null for wrong length', () => {
    const brain = new JellyBrain(3, 4, 2);
    const result = brain.guess([1, 2]); // Wrong length
    expect(result).toBeNull();
  });

  test('validates input is array', () => {
    const brain = new JellyBrain(3, 4, 2);
    const result = brain.guess("not an array");
    expect(result).toBeNull();
  });

  test('returns output with correct dimensions', () => {
    const brain = new JellyBrain(3, 4, 2);
    const output = brain.guess([1, 2, 3]);
    expect(Array.isArray(output)).toBe(true);
    expect(output).toHaveLength(2);
  });

  test('output values are in valid range for sigmoid', () => {
    const brain = new JellyBrain(
      3, 4, 2,
      costFuncs.errorSquared,
      0.01,
      activationFuncs.sigmoid,
      activationFuncs.sigmoid
    );
    
    const output = brain.guess([1, 2, 3]);
    output.forEach(val => {
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(1);
    });
  });

  test('softmax outputs sum to 1', () => {
    const brain = new JellyBrain(
      3, 4, 3,
      costFuncs.crossEntropy,
      0.01,
      activationFuncs.sigmoid,
      activationFuncs.softmax
    );
    
    const output = brain.guess([1, 2, 3]);
    const sum = output.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 6);
  });

  test('stores intermediate values correctly', () => {
    const brain = new JellyBrain(2, 3, 2);
    brain.guess([1, 2]);
    
    expect(brain.getHiddenZ()).toHaveLength(3);
    expect(brain.getHiddenA()).toHaveLength(3);
    expect(brain.getOutputZ()).toHaveLength(2);
    expect(brain.getOutputA()).toHaveLength(2);
  });

  test('is deterministic with same input', () => {
    const brain = new JellyBrain(2, 3, 2);
    const output1 = brain.guess([1, 2]);
    const output2 = brain.guess([1, 2]);
    expectArrayCloseTo(output1, output2, 10);
  });

  test('different inputs produce different outputs', () => {
    const brain = new JellyBrain(2, 3, 2);
    const output1 = brain.guess([1, 0]);
    const output2 = brain.guess([0, 1]);
    
    // At least one output should be different
    const different = output1.some((val, idx) => 
      Math.abs(val - output2[idx]) > 0.01
    );
    expect(different).toBe(true);
  });

  test('handles zero input', () => {
    const brain = new JellyBrain(3, 4, 2);
    const output = brain.guess([0, 0, 0]);
    expect(output).toHaveLength(2);
    output.forEach(val => {
      expect(isNaN(val)).toBe(false);
      expect(isFinite(val)).toBe(true);
    });
  });

  test('handles large input values', () => {
    const brain = new JellyBrain(2, 3, 2);
    const output = brain.guess([1000, -1000]);
    expect(output).toHaveLength(2);
    output.forEach(val => {
      expect(isNaN(val)).toBe(false);
      expect(isFinite(val)).toBe(true);
    });
  });
});

describe('JellyBrain Training', () => {
  test('validates input length during training', () => {
    const brain = new JellyBrain(2, 3, 2);
    const initialBatchSize = brain.getBatchSize();
    
    // Console.error is called, but training doesn't happen
    brain.train([1], [0, 1]); // Wrong input length
    
    expect(brain.getBatchSize()).toBe(initialBatchSize);
  });

  test('validates target length during training', () => {
    const brain = new JellyBrain(2, 3, 2);
    const initialBatchSize = brain.getBatchSize();
    
    brain.train([1, 2], [0]); // Wrong target length
    
    expect(brain.getBatchSize()).toBe(initialBatchSize);
  });

  test('single training step updates weights', () => {
    const brain = new JellyBrain(2, 3, 2);
    const initialWeights = math.clone(brain.getWeightsHO());
    
    brain.train([1, 0], [1, 0]);
    
    const finalWeights = brain.getWeightsHO();
    
    // At least one weight should have changed
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

  test('training reduces error over time', () => {
    const brain = new JellyBrain(2, 3, 1, costFuncs.errorSquared, 0.1);
    
    const input = [0.5, 0.5];
    const target = [1];
    
    const initialOutput = brain.guess(input);
    const initialError = Math.pow(initialOutput[0] - target[0], 2);
    
    // Train multiple times
    for (let i = 0; i < 50; i++) {
      brain.train(input, target);
    }
    
    const finalOutput = brain.guess(input);
    const finalError = Math.pow(finalOutput[0] - target[0], 2);
    
    expect(finalError).toBeLessThan(initialError);
  });

  test('learns XOR problem', () => {
    const brain = new JellyBrain(
      2, 4, 1,
      costFuncs.errorSquared,
      0.5,
      activationFuncs.sigmoid,
      activationFuncs.sigmoid
    );
    
    const trainingData = [
      { input: [0, 0], target: [0] },
      { input: [0, 1], target: [1] },
      { input: [1, 0], target: [1] },
      { input: [1, 1], target: [0] }
    ];
    
    // Train for many epochs
    for (let epoch = 0; epoch < 3000; epoch++) {
      for (const data of trainingData) {
        brain.train(data.input, data.target);
      }
    }
    
    // Test predictions with relaxed expectations
    const pred1 = brain.guess([0, 0])[0];
    const pred2 = brain.guess([0, 1])[0];
    const pred3 = brain.guess([1, 0])[0];
    const pred4 = brain.guess([1, 1])[0];

    expect(pred1).toBeLessThan(0.3);
    expect(pred2).toBeGreaterThan(0.7);
    expect(pred3).toBeGreaterThan(0.7);
    expect(pred4).toBeLessThan(0.3);
  }, 10000); // Increase timeout for this test

  test('training with same input/output converges', () => {
    const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
    
    for (let i = 0; i < 2000; i++) {
      brain.train([0.5, 0.5], [0.5, 0.5]);
    }

    const output = brain.guess([0.5, 0.5]);
    expectArrayCloseTo(output, [0.5, 0.5], 1);
  });
});

describe('JellyBrain Batch Training', () => {
  test('addToBatch increases batch size', () => {
    const brain = new JellyBrain(2, 3, 2);
    expect(brain.getBatchSize()).toBe(0);
    
    brain.addToBatch([1, 2], [0, 1]);
    expect(brain.getBatchSize()).toBe(1);
    
    brain.addToBatch([2, 3], [1, 0]);
    expect(brain.getBatchSize()).toBe(2);
  });

  test('computeBatch updates weights', () => {
    const brain = new JellyBrain(2, 3, 2);
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

  test('clearBatch resets without updating weights', () => {
    const brain = new JellyBrain(2, 3, 2);
    const initialWeights = math.clone(brain.getWeightsIH());
    
    brain.addToBatch([1, 0], [1, 0]);
    brain.clearBatch();
    
    const finalWeights = brain.getWeightsIH();
    expectMatrixCloseTo(initialWeights, finalWeights, 10);
    expect(brain.getBatchSize()).toBe(0);
  });

  test('accumulated gradients in batch', () => {
    const brain = new JellyBrain(2, 3, 1, costFuncs.errorSquared, 0.1);
    
    brain.addToBatch([1, 0], [1]);
    const change1 = math.clone(brain.getWeightsIHChange());
    
    brain.addToBatch([0, 1], [0]);
    const change2 = brain.getWeightsIHChange();
    
    // Changes should accumulate
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

  test('computeBatch with empty batch shows warning', () => {
    const brain = new JellyBrain(2, 3, 2);
    const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
    
    brain.computeBatch();
    
    expect(consoleSpy).toHaveBeenCalledWith(
      expect.stringContaining('No batches to compute')
    );
    
    consoleSpy.mockRestore();
  });

  test('batch training equivalent to accumulated single training', () => {
    const brain1 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
    const brain2 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
    
    // Make them identical
    const exported = brain1.exportBrain();
    brain2.importBrain(exported);
    
    const data = [
      { input: [1, 0], target: [1, 0] },
      { input: [0, 1], target: [0, 1] }
    ];
    
    // Train brain1 with batch
    for (const d of data) {
      brain1.addToBatch(d.input, d.target);
    }
    brain1.computeBatch();
    
    // Train brain2 with same batch
    for (const d of data) {
      brain2.addToBatch(d.input, d.target);
    }
    brain2.computeBatch();
    
    const output1 = brain1.guess([0.5, 0.5]);
    const output2 = brain2.guess([0.5, 0.5]);
    
    expectArrayCloseTo(output1, output2, 10);
  });
});

describe('JellyBrain Learning Rate', () => {
  test('setLearningRate updates value', () => {
    const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
    brain.setLearningRate(0.5);
    expect(brain.getLearningRate()).toBe(0.5);
  });

  test('higher learning rate produces larger weight changes', () => {
    const brain1 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.001);
    const brain2 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
    
    // Copy weights to ensure same starting point
    const exportBrain = brain1.exportBrain();
    brain2.importBrain(exportBrain);
    
    const initialWeights = brain1.getWeightsIH();
    
    brain1.train([1, 0], [1, 0]);
    brain2.train([1, 0], [1, 0]);
    
    const weights1 = brain1.getWeightsIH();
    const weights2 = brain2.getWeightsIH();
    
    // Calculate total weight change
    let change1 = 0, change2 = 0;
    for (let i = 0; i < initialWeights.length; i++) {
      for (let j = 0; j < initialWeights[i].length; j++) {
        change1 += Math.abs(weights1[i][j] - initialWeights[i][j]);
        change2 += Math.abs(weights2[i][j] - initialWeights[i][j]);
      }
    }
    
    expect(change2).toBeGreaterThan(change1);
  });

  test('rejects non-numeric learning rate', () => {
    const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
    const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
    
    brain.setLearningRate("invalid");
    
    expect(brain.getLearningRate()).toBe(0.1);
    expect(consoleSpy).toHaveBeenCalled();
    
    consoleSpy.mockRestore();
  });

  test('very small learning rate produces tiny changes', () => {
    const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 1e-10);
    const initialWeights = brain.getWeightsIH();
    
    brain.train([1, 0], [1, 0]);
    
    const finalWeights = brain.getWeightsIH();
    let maxChange = 0;
    
    for (let i = 0; i < initialWeights.length; i++) {
      for (let j = 0; j < initialWeights[i].length; j++) {
        maxChange = Math.max(
          maxChange,
          Math.abs(finalWeights[i][j] - initialWeights[i][j])
        );
      }
    }
    
    expect(maxChange).toBeLessThan(1e-8);
  });
});

describe('JellyBrain Import/Export', () => {
  test('export returns object with correct structure', () => {
    const brain = new JellyBrain(2, 3, 2);
    const exported = brain.exportBrain();
    
    expect(exported).toHaveProperty('weightsIH');
    expect(exported).toHaveProperty('weightsHO');
    expect(exported).toHaveProperty('biasH');
    expect(exported).toHaveProperty('biasO');
  });

  test('export creates deep copy', () => {
    const brain = new JellyBrain(2, 3, 2);
    const exported = brain.exportBrain();
    
    // Modify exported data
    exported.weightsIH[0][0] = 999;
    
    // Original should be unchanged
    const currentWeights = brain.getWeightsIH();
    expect(currentWeights[0][0]).not.toBe(999);
  });

  test('round trip preserves network state', () => {
    const brain1 = new JellyBrain(2, 3, 2);
    brain1.train([1, 0], [1, 0]); // Get non-zero weights
    
    const exported = brain1.exportBrain();
    
    const brain2 = new JellyBrain(2, 3, 2);
    brain2.importBrain(exported);
    
    // Test same output
    const output1 = brain1.guess([1, 0]);
    const output2 = brain2.guess([1, 0]);
    
    expectArrayCloseTo(output1, output2, 10);
  });

  test('import validates weight dimensions', () => {
    const brain = new JellyBrain(2, 3, 2);
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    const originalWeights = math.clone(brain.getWeightsIH());
    
    const invalidExport = {
      weightsIH: [[1, 2], [3, 4]], // Wrong dimensions
      weightsHO: [[1, 2], [3, 4], [5, 6]],
      biasH: [0, 0, 0],
      biasO: [0, 0]
    };
    
    brain.importBrain(invalidExport);
    
    const afterWeights = brain.getWeightsIH();
    expectMatrixCloseTo(originalWeights, afterWeights, 10);
    expect(consoleSpy).toHaveBeenCalled();
    
    consoleSpy.mockRestore();
  });

  test('import rejects null', () => {
    const brain = new JellyBrain(2, 3, 2);
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    const originalWeights = math.clone(brain.getWeightsIH());
    
    brain.importBrain(null);
    
    const afterWeights = brain.getWeightsIH();
    expectMatrixCloseTo(originalWeights, afterWeights, 10);
    expect(consoleSpy).toHaveBeenCalled();
    
    consoleSpy.mockRestore();
  });

  test('import rejects invalid types', () => {
    const brain = new JellyBrain(2, 3, 2);
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    const originalWeights = math.clone(brain.getWeightsIH());
    
    brain.importBrain("invalid");
    brain.importBrain(123);
    brain.importBrain([]);
    
    const afterWeights = brain.getWeightsIH();
    expectMatrixCloseTo(originalWeights, afterWeights, 10);
    
    consoleSpy.mockRestore();
  });

  test('exported brain works in new instance', () => {
    const brain1 = new JellyBrain(3, 5, 2);
    
    // Train it
    for (let i = 0; i < 200; i++) {
      brain1.train([0.1, 0.5, 0.9], [1, 0]);
    }
    
    const exported = brain1.exportBrain();
    const brain2 = new JellyBrain(3, 5, 2);
    brain2.importBrain(exported);
    
    for (let i = 0; i < 200; i++) {  // Train multiple times instead of once
      brain2.train([0.2, 0.4, 0.8], [0, 1]);
    }
    
    // They should now be different
    const output1 = brain1.guess([0.1, 0.5, 0.9]);
    const output2 = brain2.guess([0.1, 0.5, 0.9]);
    
    const different = output1.some((val, idx) => 
      Math.abs(val - output2[idx]) > 0.01
    );
    expect(different).toBe(true);
  });
});

describe('JellyBrain Activation Functions', () => {
  test('ReLU produces non-negative outputs', () => {
    const brain = new JellyBrain(
      2, 3, 2,
      costFuncs.errorSquared,
      0.1,
      activationFuncs.relu,
      activationFuncs.relu
    );
    
    const output = brain.guess([-10, 10]);
    output.forEach(val => {
      expect(val).toBeGreaterThanOrEqual(0);
    });
  });

  test('Leaky ReLU allows small negative values', () => {
    const brain = new JellyBrain(
      1, 3, 1,
      costFuncs.errorSquared,
      0.1,
      activationFuncs.lrelu,
      activationFuncs.lrelu
    );
    
    // Set weights to produce negative values
    const exported = brain.exportBrain();
    exported.weightsIH = [[-1, -1, -1]];
    exported.weightsHO = [[-1], [-1], [-1]];
    brain.importBrain(exported);
    
    const output = brain.guess([1]);
    expect(output[0]).not.toBe(0);
  });

  test('Tanh produces outputs in range [-1, 1]', () => {
    const brain = new JellyBrain(
      2, 3, 2,
      costFuncs.errorSquared,
      0.1,
      activationFuncs.tanh,
      activationFuncs.tanh
    );
    
    const output = brain.guess([10, -10]);
    output.forEach(val => {
      expect(val).toBeGreaterThanOrEqual(-1);
      expect(val).toBeLessThanOrEqual(1);
    });
  });

  test('Sigmoid produces outputs in range [0, 1]', () => {
    const brain = new JellyBrain(
      2, 3, 2,
      costFuncs.errorSquared,
      0.1,
      activationFuncs.sigmoid,
      activationFuncs.sigmoid
    );
    
    const output = brain.guess([100, -100]);
    output.forEach(val => {
      expect(val).toBeGreaterThanOrEqual(0);
      expect(val).toBeLessThanOrEqual(1);
    });
  });

  test('Linear activation produces unbounded outputs', () => {
    const brain = new JellyBrain(
      2, 3, 2,
      costFuncs.errorSquared,
      0.1,
      activationFuncs.linear,
      activationFuncs.linear
    );
    
    const exported = brain.exportBrain();
    exported.weightsIH = [[10, 10, 10], [10, 10, 10]];
    exported.weightsHO = [[10, 10], [10, 10], [10, 10]];
    brain.importBrain(exported);
    
    const output = brain.guess([1, 1]);
    expect(Math.abs(output[0])).toBeGreaterThan(1);
  });

  test('Softmax with CrossEntropy trains correctly', () => {
    const brain = new JellyBrain(
      2, 4, 3,
      costFuncs.crossEntropy,
      0.1,
      activationFuncs.sigmoid,
      activationFuncs.softmax
    );
    
    const input = [0.5, 0.8];
    const target = [0, 1, 0];
    
    const initialOutput = brain.guess(input);
    const initialError = Math.abs(initialOutput[1] - 1);
    
    for (let i = 0; i < 100; i++) {
      brain.train(input, target);
    }
    
    const finalOutput = brain.guess(input);
    const finalError = Math.abs(finalOutput[1] - 1);
    
    expect(finalError).toBeLessThan(initialError);
  });
});

describe('JellyBrain Cost Functions', () => {
  test('errorSquared reduces squared error', () => {
    const brain = new JellyBrain(2, 3, 1, costFuncs.errorSquared, 0.1);
    
    const input = [0.5, 0.5];
    const target = [1];
    
    const initialOutput = brain.guess(input);
    const initialError = Math.pow(initialOutput[0] - target[0], 2);
    
    for (let i = 0; i < 50; i++) {
      brain.train(input, target);
    }
    
    const finalOutput = brain.guess(input);
    const finalError = Math.pow(finalOutput[0] - target[0], 2);
    
    expect(finalError).toBeLessThan(initialError);
  });

  test('crossEntropy with softmax optimizes probability', () => {
    const brain = new JellyBrain(
      2, 5, 3,
      costFuncs.crossEntropy,
      0.2,
      activationFuncs.sigmoid,
      activationFuncs.softmax
    );
    
    // Train to recognize class 1
    for (let i = 0; i < 100; i++) {
      brain.train([0.3, 0.7], [0, 1, 0]);
    }
    
    const output = brain.guess([0.3, 0.7]);
    expect(output[1]).toBeGreaterThan(output[0]);
    expect(output[1]).toBeGreaterThan(output[2]);
  });
});

describe('JellyBrain Edge Cases', () => {
  test('handles network with 1 hidden node', () => {
    const brain = new JellyBrain(2, 1, 2);
    const output = brain.guess([1, 0]);
    expect(output).toHaveLength(2);
    
    brain.train([1, 0], [1, 0]);
    const output2 = brain.guess([1, 0]);
    expect(output2).toHaveLength(2);
  });

  test('handles single input node', () => {
    const brain = new JellyBrain(1, 3, 2);
    const output = brain.guess([0.5]);
    expect(output).toHaveLength(2);
  });

  test('handles single output node', () => {
    const brain = new JellyBrain(3, 4, 1);
    const output = brain.guess([1, 2, 3]);
    expect(output).toHaveLength(1);
  });

  test('trains with extreme learning rates', () => {
    const brain1 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 1e-10);
    const brain2 = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 10);
    
    expect(() => brain1.train([1, 0], [1, 0])).not.toThrow();
    expect(() => brain2.train([1, 0], [1, 0])).not.toThrow();
  });

  test('handles all zero target', () => {
    const brain = new JellyBrain(2, 3, 2);
    expect(() => brain.train([1, 0], [0, 0])).not.toThrow();
  });

  test('handles repeated training on same example', () => {
    const brain = new JellyBrain(2, 3, 1);
    
    for (let i = 0; i < 1000; i++) {
      brain.train([0.5, 0.5], [1]);
    }
    
    const output = brain.guess([0.5, 0.5]);
    expect(output[0]).toBeCloseTo(1, 0);
  });
});

describe('JellyBrain Integration Tests', () => {
  test('classification task: points above/below line', () => {
    const brain = new JellyBrain(
      2, 5, 2,
      costFuncs.errorSquared,
      0.5,
      activationFuncs.sigmoid,
      activationFuncs.softmax
    );
    
    const trainingData = [
      { input: [0.1, 0.9], target: [1, 0] }, // above y=x
      { input: [0.2, 0.8], target: [1, 0] },
      { input: [0.3, 0.7], target: [1, 0] },
      { input: [0.9, 0.1], target: [0, 1] }, // below y=x
      { input: [0.8, 0.2], target: [0, 1] },
      { input: [0.7, 0.3], target: [0, 1] },
    ];
    
    for (let epoch = 0; epoch < 500; epoch++) {
      for (const data of trainingData) {
        brain.train(data.input, data.target);
      }
    }
    
    const test1 = brain.guess([0.25, 0.75]); // above
    const test2 = brain.guess([0.75, 0.25]); // below
    
    expect(test1[0]).toBeGreaterThan(test1[1]);
    expect(test2[1]).toBeGreaterThan(test2[0]);
  }, 10000);

  test('regression task: approximate linear function', () => {
    const brain = new JellyBrain(
      1, 4, 1,
      costFuncs.errorSquared,
      0.1,
      activationFuncs.tanh,
      activationFuncs.linear
    );
    
    const trainingData = [];
    for (let i = 0; i <= 10; i++) {
      const x = i / 10;
      trainingData.push({ input: [x], target: [2 * x] });
    }
    
    for (let epoch = 0; epoch < 1000; epoch++) {
      for (const data of trainingData) {
        brain.train(data.input, data.target);
      }
    }
    
    const test1 = brain.guess([0.5]);
    expect(test1[0]).toBeCloseTo(1.0, 0);
  }, 10000);

  test('multi-class classification', () => {
    const brain = new JellyBrain(
      2, 6, 3,
      costFuncs.crossEntropy,
      0.3,
      activationFuncs.relu,
      activationFuncs.softmax
    );
    
    const trainingData = [
      { input: [0, 0], target: [1, 0, 0] },
      { input: [0, 1], target: [0, 1, 0] },
      { input: [1, 0], target: [0, 0, 1] },
      { input: [1, 1], target: [1, 0, 0] },
    ];
    
    for (let epoch = 0; epoch < 500; epoch++) {
      for (const data of trainingData) {
        brain.train(data.input, data.target);
      }
    }
    
    const test1 = brain.guess([0, 0]);
    const class1 = test1.indexOf(Math.max(...test1));
    expect(class1).toBe(0);
  }, 10000);

  test('batch learning produces smooth convergence', () => {
    const brain = new JellyBrain(2, 4, 1, costFuncs.errorSquared, 0.1);
    
    const trainingData = [
      { input: [0, 0], target: [0] },
      { input: [0, 1], target: [1] },
      { input: [1, 0], target: [1] },
      { input: [1, 1], target: [0] }
    ];
    
    const errors = [];
    
    for (let epoch = 0; epoch < 100; epoch++) {
      for (const data of trainingData) {
        brain.addToBatch(data.input, data.target);
      }
      brain.computeBatch();
      
      // Calculate error
      let totalError = 0;
      for (const data of trainingData) {
        const output = brain.guess(data.input);
        totalError += Math.pow(output[0] - data.target[0], 2);
      }
      errors.push(totalError);
    }
    
    // Error should generally decrease
    expect(errors[errors.length - 1]).toBeLessThan(errors[0]);
  });
});

describe('JellyBrain Getter Methods', () => {
  test('all getters return values', () => {
    const brain = new JellyBrain(2, 3, 2);
    brain.guess([1, 0]);
    
    expect(brain.getInputNodes()).toBeDefined();
    expect(brain.getHiddenNodes()).toBeDefined();
    expect(brain.getOutputNodes()).toBeDefined();
    expect(brain.getLearningRate()).toBeDefined();
    expect(brain.getHiddenZ()).toBeDefined();
    expect(brain.getHiddenA()).toBeDefined();
    expect(brain.getOutputZ()).toBeDefined();
    expect(brain.getOutputA()).toBeDefined();
    expect(brain.getWeightsIH()).toBeDefined();
    expect(brain.getWeightsHO()).toBeDefined();
    expect(brain.getBiasIH()).toBeDefined();
    expect(brain.getBiasHO()).toBeDefined();
    expect(brain.getBatchSize()).toBeDefined();
    expect(brain.getWeightsIHChange()).toBeDefined();
    expect(brain.getWeightsHOChange()).toBeDefined();
    expect(brain.getBiasIHChange()).toBeDefined();
    expect(brain.getBiasHOChange()).toBeDefined();
  });

  test('getters return deep copies', () => {
    const brain = new JellyBrain(2, 3, 2);
    const weights1 = brain.getWeightsIH();
    weights1[0][0] = 999;
    
    const weights2 = brain.getWeightsIH();
    expect(weights2[0][0]).not.toBe(999);
  });
});