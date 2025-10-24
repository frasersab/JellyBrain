const { activationFuncs } = require('../JellyBrain');
const math = require('mathjs');

describe('Activation Functions', () => {
  describe('Sigmoid', () => {
    test('outputs are in range [0, 1]', () => {
      const inputs = [-100, -10, -1, 0, 1, 10, 100];
      inputs.forEach(x => {
        const output = activationFuncs.sigmoid.func([x]);
        expect(output[0]).toBeGreaterThanOrEqual(0);
        expect(output[0]).toBeLessThanOrEqual(1);
      });
    });

    test('sigmoid(0) = 0.5', () => {
      const output = activationFuncs.sigmoid.func([0]);
      expect(output[0]).toBeCloseTo(0.5, 10);
    });

    test('derivative is correct', () => {
      // For sigmoid: f'(x) = f(x) * (1 - f(x))
      const x = [0.5];
      const fx = activationFuncs.sigmoid.func(x);
      const dfx = activationFuncs.sigmoid.dfunc(x);
      const expected = fx[0] * (1 - fx[0]);
      expect(dfx[0]).toBeCloseTo(expected, 10);
    });

    test('handles arrays', () => {
      const input = [-1, 0, 1];
      const output = activationFuncs.sigmoid.func(input);
      expect(output).toHaveLength(3);
      output.forEach(val => {
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThanOrEqual(1);
      });
    });
  });

  describe('Tanh', () => {
    test('outputs are in range [-1, 1]', () => {
      const inputs = [-100, -10, -1, 0, 1, 10, 100];
      inputs.forEach(x => {
        const output = activationFuncs.tanh.func([x]);
        expect(output[0]).toBeGreaterThanOrEqual(-1);
        expect(output[0]).toBeLessThanOrEqual(1);
      });
    });

    test('tanh(0) = 0', () => {
      const output = activationFuncs.tanh.func([0]);
      expect(output[0]).toBeCloseTo(0, 10);
    });

    test('derivative is correct', () => {
      // For tanh: f'(x) = 1 - f(x)^2
      const x = [0.5];
      const fx = activationFuncs.tanh.func(x);
      const dfx = activationFuncs.tanh.dfunc(x);
      const expected = 1 - Math.pow(fx[0], 2);
      expect(dfx[0]).toBeCloseTo(expected, 5);
    });

    test('is antisymmetric', () => {
      const x = 2.5;
      const positive = activationFuncs.tanh.func([x]);
      const negative = activationFuncs.tanh.func([-x]);
      expect(positive[0]).toBeCloseTo(-negative[0], 10);
    });
  });

  describe('ReLU', () => {
    test('outputs are non-negative', () => {
      const inputs = [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100];
      inputs.forEach(x => {
        const output = activationFuncs.relu.func([x]);
        expect(output[0]).toBeGreaterThanOrEqual(0);
      });
    });

    test('negative inputs become 0', () => {
      const negatives = [-100, -10, -1, -0.01];
      negatives.forEach(x => {
        const output = activationFuncs.relu.func([x]);
        expect(output[0]).toBe(0);
      });
    });

    test('positive inputs remain unchanged', () => {
      const positives = [0.1, 1, 10, 100];
      positives.forEach(x => {
        const output = activationFuncs.relu.func([x]);
        expect(output[0]).toBeCloseTo(x, 10);
      });
    });

    test('derivative is 0 for negative, 1 for positive', () => {
      const negative = activationFuncs.relu.dfunc([-5]);
      expect(negative[0]).toBe(0);
      
      const positive = activationFuncs.relu.dfunc([5]);
      expect(positive[0]).toBe(1);
      
      const zero = activationFuncs.relu.dfunc([0]);
      expect(zero[0]).toBe(0);
    });
  });

  describe('Leaky ReLU', () => {
    test('positive inputs remain unchanged', () => {
      const positives = [0.1, 1, 10, 100];
      positives.forEach(x => {
        const output = activationFuncs.lrelu.func([x]);
        expect(output[0]).toBeCloseTo(x, 10);
      });
    });

    test('negative inputs are scaled by 0.1', () => {
      const negatives = [-100, -10, -1, -0.1];
      negatives.forEach(x => {
        const output = activationFuncs.lrelu.func([x]);
        expect(output[0]).toBeCloseTo(0.1 * x, 10);
      });
    });

    test('derivative is 0.1 for negative, 1 for positive', () => {
      const negative = activationFuncs.lrelu.dfunc([-5]);
      expect(negative[0]).toBe(0.1);
      
      const positive = activationFuncs.lrelu.dfunc([5]);
      expect(positive[0]).toBe(1);
    });
  });

  describe('Linear', () => {
    test('output equals input', () => {
      const inputs = [-100, -1, 0, 1, 100];
      inputs.forEach(x => {
        const output = activationFuncs.linear.func([x]);
        expect(output[0]).toBe(x);
      });
    });

    test('derivative is always 1', () => {
      const inputs = [-100, -1, 0, 1, 100];
      inputs.forEach(x => {
        const output = activationFuncs.linear.dfunc([x]);
        expect(output[0]).toBe(1);
      });
    });

    test('handles arrays', () => {
      const input = [-2, -1, 0, 1, 2];
      const output = activationFuncs.linear.func(input);
      expect(output).toEqual(input);
    });
  });

  describe('Softmax', () => {
    test('outputs sum to 1', () => {
      const testCases = [
        [1, 2, 3],
        [-1, 0, 1],
        [100, 200, 300],
        [0.1, 0.2, 0.3],
        [-5, -10, -15]
      ];
      
      testCases.forEach(input => {
        const output = activationFuncs.softmax.func(input);
        const sum = output.reduce((a, b) => a + b, 0);
        expect(sum).toBeCloseTo(1.0, 10);
      });
    });

    test('all outputs are positive', () => {
      const input = [-100, 0, 100];
      const output = activationFuncs.softmax.func(input);
      output.forEach(val => {
        expect(val).toBeGreaterThan(0);
        expect(val).toBeLessThanOrEqual(1);
      });
    });

    test('larger inputs have larger probabilities', () => {
      const input = [1, 2, 3];
      const output = activationFuncs.softmax.func(input);
      expect(output[2]).toBeGreaterThan(output[1]);
      expect(output[1]).toBeGreaterThan(output[0]);
    });

    test('handles overflow with large values', () => {
      const input = [1000, 2000, 3000];
      const output = activationFuncs.softmax.func(input);
      output.forEach(val => {
        expect(isFinite(val)).toBe(true);
        expect(isNaN(val)).toBe(false);
      });
      expect(Math.abs(output[0]) + Math.abs(output[1])).toBeLessThan(0.001);
      expect(output[2]).toBeCloseTo(1.0, 5);
    });

    test('derivative returns correct Jacobian matrix', () => {
      const input = [1, 2, 3];
      const jacobian = activationFuncs.softmax.dfunc(input);
      
      // Jacobian should be n×n for n inputs
      expect(jacobian).toHaveLength(3);
      expect(jacobian[0]).toHaveLength(3);
      
      // Check if it's a valid Jacobian (all values finite)
      jacobian.forEach(row => {
        row.forEach(val => {
          expect(isFinite(val)).toBe(true);
        });
      });
    });
  });
});