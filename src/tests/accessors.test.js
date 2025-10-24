const { JellyBrain, costFuncs, activationFuncs } = require('../JellyBrain');
const math = require('mathjs');

describe('JellyBrain Accessors and Properties', () => {
  describe('Constructor Properties', () => {
    test('getInputNodes returns correct value', () => {
      const brain = new JellyBrain(5, 4, 3);
      expect(brain.getInputNodes()).toBe(5);
    });

    test('getHiddenNodes returns correct value', () => {
      const brain = new JellyBrain(5, 4, 3);
      expect(brain.getHiddenNodes()).toBe(4);
    });

    test('getOutputNodes returns correct value', () => {
      const brain = new JellyBrain(5, 4, 3);
      expect(brain.getOutputNodes()).toBe(3);
    });

    test('getLearningRate returns initial value', () => {
      const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.123);
      expect(brain.getLearningRate()).toBe(0.123);
    });

    test('default learning rate is 0.05', () => {
      const brain = new JellyBrain(2, 3, 2);
      expect(brain.getLearningRate()).toBe(0.05);
    });
  });

  describe('Weight and Bias Getters', () => {
    test('getWeightsIH returns correct dimensions', () => {
      const brain = new JellyBrain(3, 4, 2);
      const weights = brain.getWeightsIH();
      expect(weights).toHaveLength(3);
      expect(weights[0]).toHaveLength(4);
    });

    test('getWeightsHO returns correct dimensions', () => {
      const brain = new JellyBrain(3, 4, 2);
      const weights = brain.getWeightsHO();
      expect(weights).toHaveLength(4);
      expect(weights[0]).toHaveLength(2);
    });

    test('getBiasIH returns correct dimensions', () => {
      const brain = new JellyBrain(3, 4, 2);
      const bias = brain.getBiasIH();
      expect(bias).toHaveLength(4);
    });

    test('getBiasHO returns correct dimensions', () => {
      const brain = new JellyBrain(3, 4, 2);
      const bias = brain.getBiasHO();
      expect(bias).toHaveLength(2);
    });

    test('getters return deep copies not references', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      const weights1 = brain.getWeightsIH();
      weights1[0][0] = 999;
      const weights2 = brain.getWeightsIH();
      expect(weights2[0][0]).not.toBe(999);
      
      const bias1 = brain.getBiasIH();
      bias1[0] = 999;
      const bias2 = brain.getBiasIH();
      expect(bias2[0]).not.toBe(999);
    });
  });

  describe('State Getters', () => {
    test('getHiddenZ and getHiddenA are populated after guess', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      // Before guess, they might be undefined
      brain.guess([0.5, 0.7]);
      
      const hiddenZ = brain.getHiddenZ();
      const hiddenA = brain.getHiddenA();
      
      expect(hiddenZ).toBeDefined();
      expect(hiddenA).toBeDefined();
      expect(hiddenZ).toHaveLength(3);
      expect(hiddenA).toHaveLength(3);
    });

    test('getOutputZ and getOutputA are populated after guess', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      brain.guess([0.5, 0.7]);
      
      const outputZ = brain.getOutputZ();
      const outputA = brain.getOutputA();
      
      expect(outputZ).toBeDefined();
      expect(outputA).toBeDefined();
      expect(outputZ).toHaveLength(2);
      expect(outputA).toHaveLength(2);
    });

    test('state getters return deep copies', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.guess([0.5, 0.7]);
      
      const hiddenA1 = brain.getHiddenA();
      hiddenA1[0] = 999;
      const hiddenA2 = brain.getHiddenA();
      expect(hiddenA2[0]).not.toBe(999);
    });
  });

  describe('Batch Training Getters', () => {
    test('getBatchSize starts at 0', () => {
      const brain = new JellyBrain(2, 3, 2);
      expect(brain.getBatchSize()).toBe(0);
    });

    test('getBatchSize increments with addToBatch', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.addToBatch([1, 0], [0, 1]);
      expect(brain.getBatchSize()).toBe(1);
      brain.addToBatch([0, 1], [1, 0]);
      expect(brain.getBatchSize()).toBe(2);
    });

    test('getBatchSize resets after computeBatch', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.addToBatch([1, 0], [0, 1]);
      brain.addToBatch([0, 1], [1, 0]);
      brain.computeBatch();
      expect(brain.getBatchSize()).toBe(0);
    });

    test('getWeightsIHChange returns accumulated gradients', () => {
      const brain = new JellyBrain(2, 3, 2);
      
      const initialChange = brain.getWeightsIHChange();
      // Fix: Just check it's a zero matrix without using ._data
      expect(initialChange).toEqual([[0, 0, 0], [0, 0, 0]]);
      
      brain.addToBatch([1, 0], [1, 0]);
      const afterChange = brain.getWeightsIHChange();
      
      // Some gradients should be non-zero
      let hasNonZero = false;
      for (let i = 0; i < afterChange.length; i++) {
        for (let j = 0; j < afterChange[i].length; j++) {
          if (Math.abs(afterChange[i][j]) > 1e-10) {
            hasNonZero = true;
            break;
          }
        }
      }
      expect(hasNonZero).toBe(true);
    });

    test('gradient getters return deep copies', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.addToBatch([1, 0], [1, 0]);
      
      const change1 = brain.getWeightsIHChange();
      change1[0][0] = 999;
      const change2 = brain.getWeightsIHChange();
      expect(change2[0][0]).not.toBe(999);
    });
  });

  describe('Learning Rate Setter', () => {
    test('setLearningRate updates the value', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.setLearningRate(0.5);
      expect(brain.getLearningRate()).toBe(0.5);
    });

    test('setLearningRate accepts zero with warning', () => {
      const brain = new JellyBrain(2, 3, 2);
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      brain.setLearningRate(0);
      expect(brain.getLearningRate()).toBe(0);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Learning rate is 0')
      );
      
      consoleSpy.mockRestore();
    });

    test('setLearningRate accepts negative with warning', () => {
      const brain = new JellyBrain(2, 3, 2);
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      brain.setLearningRate(-0.1);
      expect(brain.getLearningRate()).toBe(-0.1);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('Learning rate is negative')
      );
      
      consoleSpy.mockRestore();
    });

    test('setLearningRate rejects non-numeric values', () => {
      const brain = new JellyBrain(2, 3, 2, costFuncs.errorSquared, 0.1);
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();
      
      brain.setLearningRate("not a number");
      expect(brain.getLearningRate()).toBe(0.1); // Unchanged
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('not a number')
      );
      
      brain.setLearningRate(NaN);
      expect(brain.getLearningRate()).toBe(0.1); // Still unchanged
      
      consoleSpy.mockRestore();
    });

    test('setLearningRate works with very small values', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.setLearningRate(1e-10);
      expect(brain.getLearningRate()).toBe(1e-10);
    });

    test('setLearningRate works with large values', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.setLearningRate(100);
      expect(brain.getLearningRate()).toBe(100);
    });
  });

  describe('Import/Export', () => {
    test('exportBrain returns correct structure', () => {
      const brain = new JellyBrain(2, 3, 2);
      const exported = brain.exportBrain();
      
      expect(exported).toHaveProperty('weightsIH');
      expect(exported).toHaveProperty('weightsHO');
      expect(exported).toHaveProperty('biasH');
      expect(exported).toHaveProperty('biasO');
      
      expect(exported.weightsIH).toHaveLength(2);
      expect(exported.weightsIH[0]).toHaveLength(3);
      expect(exported.weightsHO).toHaveLength(3);
      expect(exported.weightsHO[0]).toHaveLength(2);
      expect(exported.biasH).toHaveLength(3);
      expect(exported.biasO).toHaveLength(2);
    });

    test('exportBrain creates deep copy', () => {
      const brain = new JellyBrain(2, 3, 2);
      const exported1 = brain.exportBrain();
      exported1.weightsIH[0][0] = 999;
      
      const exported2 = brain.exportBrain();
      expect(exported2.weightsIH[0][0]).not.toBe(999);
    });

    test('importBrain updates network state', () => {
      const brain1 = new JellyBrain(2, 3, 2);
      const brain2 = new JellyBrain(2, 3, 2);
      
      // Train brain1 to change its weights
      for (let i = 0; i < 10; i++) {
        brain1.train([0.5, 0.7], [1, 0]);
      }
      
      const exported = brain1.exportBrain();
      brain2.importBrain(exported);
      
      // Both should produce same output
      const output1 = brain1.guess([0.3, 0.6]);
      const output2 = brain2.guess([0.3, 0.6]);
      
      expect(output1).toEqual(output2);
    });

    test('importBrain validates dimensions', () => {
      const brain = new JellyBrain(2, 3, 2);
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      const invalidExport = {
        weightsIH: [[1, 2]], // Wrong dimensions (should be 2x3)
        weightsHO: [[1, 2], [3, 4], [5, 6]],
        biasH: [0, 0, 0],
        biasO: [0, 0]
      };
      
      const originalWeights = brain.getWeightsIH();
      brain.importBrain(invalidExport);
      const afterWeights = brain.getWeightsIH();
      
      // Weights should not have changed
      expect(afterWeights).toEqual(originalWeights);
      expect(consoleSpy).toHaveBeenCalled();
      
      consoleSpy.mockRestore();
    });

    test('importBrain rejects null and invalid types', () => {
      const brain = new JellyBrain(2, 3, 2);
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      const originalWeights = brain.getWeightsIH();
      
      brain.importBrain(null);
      expect(brain.getWeightsIH()).toEqual(originalWeights);
      
      brain.importBrain("invalid");
      expect(brain.getWeightsIH()).toEqual(originalWeights);
      
      brain.importBrain(123);
      expect(brain.getWeightsIH()).toEqual(originalWeights);
      
      brain.importBrain([]);
      expect(brain.getWeightsIH()).toEqual(originalWeights);
      
      expect(consoleSpy).toHaveBeenCalledTimes(4);
      consoleSpy.mockRestore();
    });

    test('importBrain creates deep copy of imported data', () => {
      const brain = new JellyBrain(2, 3, 2);
      const exportData = {
        weightsIH: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        weightsHO: [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        biasH: [0.1, 0.2, 0.3],
        biasO: [0.1, 0.2]
      };
      
      brain.importBrain(exportData);
      
      // Modify original export data
      exportData.weightsIH[0][0] = 999;
      
      // Brain's weights should not be affected
      const weights = brain.getWeightsIH();
      expect(weights[0][0]).toBeCloseTo(0.1, 10);
    });

    test('networks can share learned features via import/export', () => {
      const teacher = new JellyBrain(2, 4, 1, costFuncs.errorSquared, 0.1);
      
      // Teacher learns
      for (let i = 0; i < 100; i++) {
        teacher.train([0, 0], [0]);
        teacher.train([1, 1], [1]);
      }
      
      // Student copies teacher's knowledge
      const student = new JellyBrain(2, 4, 1);
      student.importBrain(teacher.exportBrain());
      
      // Both should perform similarly
      const teacherOutput = teacher.guess([0.5, 0.5]);
      const studentOutput = student.guess([0.5, 0.5]);
      
      expect(studentOutput).toEqual(teacherOutput);
    });
  });

  describe('All Getters Return Values', () => {
    test('all getters are defined and return expected types', () => {
      const brain = new JellyBrain(2, 3, 2);
      brain.guess([0.5, 0.7]); // Populate state
      brain.addToBatch([0.5, 0.7], [1, 0]); // Populate batch data
      
      // Node counts
      expect(typeof brain.getInputNodes()).toBe('number');
      expect(typeof brain.getHiddenNodes()).toBe('number');
      expect(typeof brain.getOutputNodes()).toBe('number');
      
      // Learning rate
      expect(typeof brain.getLearningRate()).toBe('number');
      
      // State arrays
      expect(Array.isArray(brain.getHiddenZ())).toBe(true);
      expect(Array.isArray(brain.getHiddenA())).toBe(true);
      expect(Array.isArray(brain.getOutputZ())).toBe(true);
      expect(Array.isArray(brain.getOutputA())).toBe(true);
      
      // Weight matrices
      expect(Array.isArray(brain.getWeightsIH())).toBe(true);
      expect(Array.isArray(brain.getWeightsHO())).toBe(true);
      
      // Bias arrays
      expect(Array.isArray(brain.getBiasIH())).toBe(true);
      expect(Array.isArray(brain.getBiasHO())).toBe(true);
      
      // Batch data
      expect(typeof brain.getBatchSize()).toBe('number');
      expect(Array.isArray(brain.getWeightsIHChange())).toBe(true);
      expect(Array.isArray(brain.getWeightsHOChange())).toBe(true);
      expect(Array.isArray(brain.getBiasIHChange())).toBe(true);
      expect(Array.isArray(brain.getBiasHOChange())).toBe(true);
    });
  });
});