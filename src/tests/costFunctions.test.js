const { costFuncs } = require('../JellyBrain');
const math = require('mathjs');

describe('Cost Functions', () => {
  describe('Error Squared', () => {
    test('derivative is (prediction - target)', () => {
      const target = [1, 0, 0.5];
      const prediction = [0.8, 0.2, 0.7];
      const derivative = costFuncs.errorSquared.dfunc(target, prediction);
      
      const expected = math.subtract(prediction, target);
      expect(derivative).toEqual(expected);
    });

    test('derivative is zero when prediction equals target', () => {
      const target = [0.5, 0.5];
      const prediction = [0.5, 0.5];
      const derivative = costFuncs.errorSquared.dfunc(target, prediction);
      
      derivative.forEach(val => {
        expect(Math.abs(val)).toBeLessThan(1e-10);
      });
    });

    test('handles arrays of different sizes correctly', () => {
      const target1 = [1];
      const prediction1 = [0.5];
      const derivative1 = costFuncs.errorSquared.dfunc(target1, prediction1);
      expect(derivative1).toHaveLength(1);
      
      const target5 = [1, 0, 1, 0, 1];
      const prediction5 = [0.9, 0.1, 0.8, 0.2, 0.7];
      const derivative5 = costFuncs.errorSquared.dfunc(target5, prediction5);
      expect(derivative5).toHaveLength(5);
    });
  });

  describe('Cross Entropy', () => {
    test('derivative for classification', () => {
      const target = [0, 1, 0];  // One-hot encoded
      const prediction = [0.2, 0.7, 0.1];
      const derivative = costFuncs.crossEntropy.dfunc(target, prediction);
      
      // For cross entropy: -target/prediction
      const expected = [
        -target[0] / prediction[0],
        -target[1] / prediction[1],
        -target[2] / prediction[2]
      ];
      
      expect(derivative[0]).toBeCloseTo(expected[0], 5);
      expect(derivative[1]).toBeCloseTo(expected[1], 5);
      expect(derivative[2]).toBeCloseTo(expected[2], 5);
    });

    test('handles near-zero predictions safely', () => {
      const target = [0, 1, 0];
      const prediction = [0.0001, 0.9998, 0.0001];
      const derivative = costFuncs.crossEntropy.dfunc(target, prediction);
      
      derivative.forEach(val => {
        expect(isFinite(val)).toBe(true);
        expect(isNaN(val)).toBe(false);
      });
    });

    test('derivative is large when prediction is far from target', () => {
      const target = [0, 1, 0];
      const goodPrediction = [0.1, 0.8, 0.1];
      const badPrediction = [0.1, 0.2, 0.7];
      
      const goodDerivative = costFuncs.crossEntropy.dfunc(target, goodPrediction);
      const badDerivative = costFuncs.crossEntropy.dfunc(target, badPrediction);
      
      // The derivative magnitude should be larger for bad prediction
      expect(Math.abs(badDerivative[1])).toBeGreaterThan(Math.abs(goodDerivative[1]));
    });
  });

  describe('Binary Cross Entropy', () => {
    test('derivative calculation', () => {
      const target = [1, 0, 0.5];
      const prediction = [0.8, 0.2, 0.6];
      const derivative = costFuncs.binaryCrossEntropy.dfunc(target, prediction);
      
      derivative.forEach(val => {
        expect(isFinite(val)).toBe(true);
        expect(isNaN(val)).toBe(false);
      });
    });

    test('derivative is zero when prediction equals target', () => {
      const target = [0.7, 0.3];
      const prediction = [0.7, 0.3];
      const derivative = costFuncs.binaryCrossEntropy.dfunc(target, prediction);
      
      derivative.forEach(val => {
        expect(Math.abs(val)).toBeLessThan(0.001);
      });
    });

    test('handles edge cases near 0 and 1', () => {
      const target = [1, 0];
      const prediction = [0.9999, 0.0001];
      const derivative = costFuncs.binaryCrossEntropy.dfunc(target, prediction);
      
      derivative.forEach(val => {
        expect(isFinite(val)).toBe(true);
        expect(isNaN(val)).toBe(false);
      });
    });
  });
});