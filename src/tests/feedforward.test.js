const { JellyBrain, costFuncs, activationFuncs } = require('../JellyBrain');
const math = require('mathjs');

describe('Feedforward (Forward Propagation)', () => {
  describe('Input Validation', () => {
    test('rejects wrong input length', () => {
      const brain = new JellyBrain(3, 4, 2);
      const result = brain.guess([1, 2]); // Wrong length
      expect(result).toBeNull();
    });

    test('rejects non-array input', () => {
      const brain = new JellyBrain(3, 4, 2);
      expect(brain.guess("not an array")).toBeNull();
      expect(brain.guess(123)).toBeNull();
      expect(brain.guess(null)).toBeNull();
      expect(brain.guess(undefined)).toBeNull();
    });

    test('accepts correct input', () => {
      const brain = new JellyBrain(3, 4, 2);
      const result = brain.guess([1, 2, 3]);
      expect(result).not.toBeNull();
      expect(Array.isArray(result)).toBe(true);
      expect(result).toHaveLength(2);
    });
  });

  describe('Output Characteristics', () => {
    test('sigmoid activation produces outputs in [0, 1]', () => {
      const brain = new JellyBrain(
        2, 3, 2,
        costFuncs.errorSquared,
        0.01,
        activationFuncs.sigmoid,
        activationFuncs.sigmoid
      );
      
      const testInputs = [
        [0, 0],
        [1, 1],
        [-10, 10],
        [100, -100]
      ];
      
      testInputs.forEach(input => {
        const output = brain.guess(input);
        output.forEach(val => {
          expect(val).toBeGreaterThanOrEqual(0);
          expect(val).toBeLessThanOrEqual(1);
        });
      });
    });

    test('tanh activation produces outputs in [-1, 1]', () => {
      const brain = new JellyBrain(
        2, 3, 2,
        costFuncs.errorSquared,
        0.01,
        activationFuncs.tanh,
        activationFuncs.tanh
      );
      
      const output = brain.guess([10, -10]);
      output.forEach(val => {
        expect(val).toBeGreaterThanOrEqual(-1);
        expect(val).toBeLessThanOrEqual(1);
      });
    });

    test('relu activation produces non-negative outputs', () => {
      const brain = new JellyBrain(
        2, 3, 2,
        costFuncs.errorSquared,
        0.01,
        activationFuncs.relu,
        activationFuncs.relu
      );
      
      const output = brain.guess([-10, 10]);
      output.forEach(val => {
        expect(val).toBeGreaterThanOrEqual(0);
      });
    });

    test('softmax outputs sum to 1', () => {
      const brain = new JellyBrain(
        3, 4, 5,
        costFuncs.crossEntropy,
        0.01,
        activationFuncs.sigmoid,
        activationFuncs.softmax
      );
      
      const testInputs = [
        [1, 2, 3],
        [0, 0, 0],
        [-1, -2, -3],
        [100, 200, 300]
      ];
      
      testInputs.forEach(input => {
        const output = brain.guess(input);
        const sum = output.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1.0, 10);
      });
    });

    test('linear activation can produce unbounded outputs', () => {
      const brain = new JellyBrain(
        2, 3, 2,
        costFuncs.errorSquared,
        0.01,
        activationFuncs.linear,
        activationFuncs.linear
      );
      
      // Set large weights to ensure large outputs
      const exported = brain.exportBrain();
      exported.weightsIH = [[100, 100, 100], [100, 100, 100]];
      exported.weightsHO = [[100, 100], [100, 100], [100, 100]];
      brain.importBrain(exported);
      
      const output = brain.guess([1, 1]);
      expect(Math.abs(output[0])).toBeGreaterThan(1);
      expect(Math.abs(output[1])).toBeGreaterThan(1);
    });
  });

  describe('Determinism and Consistency', () => {
    test('same input produces same output', () => {
      const brain = new JellyBrain(3, 4, 2);
      const input = [0.5, 0.7, 0.3];
      
      const output1 = brain.guess(input);
      const output2 = brain.guess(input);
      const output3 = brain.guess(input);
      
      expect(output1).toEqual(output2);
      expect(output2).toEqual(output3);
    });

    test('different inputs produce different outputs with fixed weights', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      // Set specific weights to ensure different outputs
      const exported = brain.exportBrain();
      exported.weightsIH = [[0.5, -0.3, 0.8], [0.2, 0.7, -0.4]];
      exported.weightsHO = [[0.6, -0.2], [0.1, 0.9], [-0.5, 0.3]];
      exported.biasH = [0.1, -0.1, 0.2];
      exported.biasO = [0.0, 0.1];
      brain.importBrain(exported);
      
      const output1 = brain.guess([1, 0]);
      const output2 = brain.guess([0, 1]);
      const output3 = brain.guess([0.5, 0.5]);
      
      // Verify outputs are different
      const diff12 = Math.abs(output1[0] - output2[0]) + Math.abs(output1[1] - output2[1]);
      const diff13 = Math.abs(output1[0] - output3[0]) + Math.abs(output1[1] - output3[1]);
      const diff23 = Math.abs(output2[0] - output3[0]) + Math.abs(output2[1] - output3[1]);
      
      expect(diff12).toBeGreaterThan(0.01);
      expect(diff13).toBeGreaterThan(0.01);
      expect(diff23).toBeGreaterThan(0.01);
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

    test('handles extreme input values', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      const extremeInputs = [
        [1000, -1000],
        [-999999, 999999],
        [1e10, -1e10]
      ];
      
      extremeInputs.forEach(input => {
        const output = brain.guess(input);
        expect(output).toHaveLength(2);
        output.forEach(val => {
          expect(isNaN(val)).toBe(false);
          expect(isFinite(val)).toBe(true);
        });
      });
    });
  });

  describe('Intermediate Values Storage', () => {
    test('stores hidden layer values correctly', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.guess([1, 2]);
      
      const hiddenZ = brain.getHiddenZ();
      const hiddenA = brain.getHiddenA();
      
      expect(hiddenZ).toHaveLength(3);
      expect(hiddenA).toHaveLength(3);
      
      // Hidden activations should be sigmoid of hiddenZ
      hiddenA.forEach((val, idx) => {
        const expected = 1 / (1 + Math.exp(-hiddenZ[idx]));
        expect(val).toBeCloseTo(expected, 10);
      });
    });

    test('stores output layer values correctly', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.guess([1, 2]);
      
      const outputZ = brain.getOutputZ();
      const outputA = brain.getOutputA();
      
      expect(outputZ).toHaveLength(2);
      expect(outputA).toHaveLength(2);
      
      // Output activations should be sigmoid of outputZ
      outputA.forEach((val, idx) => {
        const expected = 1 / (1 + Math.exp(-outputZ[idx]));
        expect(val).toBeCloseTo(expected, 10);
      });
    });

    test('guess returns same values as getOutputA', () => {
      const brain = new JellyBrain(2, 3, 2);
      const output = brain.guess([0.5, 0.7]);
      const outputA = brain.getOutputA();
      
      expect(output).toEqual(outputA);
    });
  });

  describe('Mixed Activation Functions', () => {
    test('different hidden and output activations work correctly', () => {
      const configs = [
        { hidden: activationFuncs.relu, output: activationFuncs.sigmoid },
        { hidden: activationFuncs.tanh, output: activationFuncs.softmax },
        { hidden: activationFuncs.sigmoid, output: activationFuncs.linear },
        { hidden: activationFuncs.lrelu, output: activationFuncs.tanh }
      ];
      
      configs.forEach(config => {
        const brain = new JellyBrain(
          2, 3, 3,
          costFuncs.errorSquared,
          0.01,
          config.hidden,
          config.output
        );
        
        const output = brain.guess([0.5, 0.7]);
        expect(output).toHaveLength(3);
        output.forEach(val => {
          expect(isNaN(val)).toBe(false);
          expect(isFinite(val)).toBe(true);
        });
      });
    });
  });

  describe('Network Architecture Variations', () => {
    test('single hidden node network', () => {
      const brain = new JellyBrain(2, 1, 2);
      const output = brain.guess([1, 0]);
      expect(output).toHaveLength(2);
    });

    test('single input node network', () => {
      const brain = new JellyBrain(1, 3, 2);
      const output = brain.guess([0.5]);
      expect(output).toHaveLength(2);
    });

    test('single output node network', () => {
      const brain = new JellyBrain(3, 4, 1);
      const output = brain.guess([1, 2, 3]);
      expect(output).toHaveLength(1);
    });

    test('large network', () => {
      const brain = new JellyBrain(100, 50, 25);
      const input = new Array(100).fill(0.5);
      const output = brain.guess(input);
      expect(output).toHaveLength(25);
    });
  });
});