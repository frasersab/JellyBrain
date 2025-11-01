const { JellyBrain, costFuncs, activationFuncs } = require('../JellyBrain');

describe('Learning: Multi-Class Classification', () => {
  // Helper to get predicted class
  function getPredictedClass(output) {
    return output.indexOf(Math.max(...output));
  }

  // Helper to calculate accuracy
  function calculateAccuracy(brain, testData) {
    let correct = 0;
    testData.forEach(({ input, target }) => {
      const output = brain.guess(input);
      const predictedClass = getPredictedClass(output);
      const actualClass = getPredictedClass(target);
      if (predictedClass === actualClass) {
        correct++;
      }
    });
    return (correct / testData.length) * 100;
  }

  describe('Three-Class Radial Classification', () => {
    // Generate data where class depends on distance from origin
    function generateRadialData(amount) {
      const data = [];
      for (let i = 0; i < amount; i++) {
        const x = Math.random() * 2 - 1; // [-1, 1]
        const y = Math.random() * 2 - 1; // [-1, 1]
        const distance = Math.sqrt(x * x + y * y);
        
        let target;
        if (distance < 0.4) {
          target = [1, 0, 0]; // Inner circle
        } else if (distance < 0.8) {
          target = [0, 1, 0]; // Middle ring
        } else {
          target = [0, 0, 1]; // Outer ring
        }
        
        data.push({ input: [x, y], target });
      }
      return data;
    }

    test('learns to classify points by distance from origin', () => {
      const brain = new JellyBrain(
        2, 8, 3,
        costFuncs.crossEntropy,
        0.2,
        activationFuncs.relu,
        activationFuncs.softmax
      );

      // Train
      for (let i = 0; i < 4000; i++) {
        const data = generateRadialData(1);
        brain.train(data[0].input, data[0].target);
      }

      // Test
      const testData = generateRadialData(1000);
      const accuracy = calculateAccuracy(brain, testData);
      expect(accuracy).toBeGreaterThan(75);
    }, 15000);
  });

  describe('Quadrant Classification', () => {
    // Classify which quadrant a point is in
    function generateQuadrantData(amount) {
      const data = [];
      for (let i = 0; i < amount; i++) {
        const x = Math.random() * 2 - 1; // [-1, 1]
        const y = Math.random() * 2 - 1; // [-1, 1]
        
        let target;
        if (x >= 0 && y >= 0) {
          target = [1, 0, 0, 0]; // Q1
        } else if (x < 0 && y >= 0) {
          target = [0, 1, 0, 0]; // Q2
        } else if (x < 0 && y < 0) {
          target = [0, 0, 1, 0]; // Q3
        } else {
          target = [0, 0, 0, 1]; // Q4
        }
        
        data.push({ input: [x, y], target });
      }
      return data;
    }

    test('learns to classify quadrants', () => {
      const brain = new JellyBrain(
        2, 8, 4,
        costFuncs.crossEntropy,
        0.1,
        activationFuncs.sigmoid,
        activationFuncs.softmax
      );

      // Train
      for (let i = 0; i < 4000; i++) {
        const data = generateQuadrantData(1);
        brain.train(data[0].input, data[0].target);
      }

      // Test specific points
      expect(getPredictedClass(brain.guess([0.5, 0.5]))).toBe(0); // Q1
      expect(getPredictedClass(brain.guess([-0.5, 0.5]))).toBe(1); // Q2
      expect(getPredictedClass(brain.guess([-0.5, -0.5]))).toBe(2); // Q3
      expect(getPredictedClass(brain.guess([0.5, -0.5]))).toBe(3); // Q4
    }, 10000);

    test('achieves high accuracy on quadrant classification', () => {
      const brain = new JellyBrain(
        2, 6, 4,
        costFuncs.crossEntropy,
        0.1,
        activationFuncs.relu,
        activationFuncs.softmax
      );

      // Train
      for (let i = 0; i < 4000; i++) {
        const data = generateQuadrantData(1);
        brain.train(data[0].input, data[0].target);
      }

      // Test
      const testData = generateQuadrantData(1000);
      const accuracy = calculateAccuracy(brain, testData);
      expect(accuracy).toBeGreaterThan(85);
    }, 10000);
  });

  describe('RGB Color Classification', () => {
    // Classify colors as Red, Green, or Blue dominant
    function generateColorData(amount) {
      const data = [];
      for (let i = 0; i < amount; i++) {
        const r = Math.random();
        const g = Math.random();
        const b = Math.random();
        
        const max = Math.max(r, g, b);
        let target;
        if (max === r) {
          target = [1, 0, 0]; // Red dominant
        } else if (max === g) {
          target = [0, 1, 0]; // Green dominant
        } else {
          target = [0, 0, 1]; // Blue dominant
        }
        
        data.push({ input: [r, g, b], target });
      }
      return data;
    }

    test('learns to classify dominant color', () => {
      const brain = new JellyBrain(
        3, 8, 3,
        costFuncs.crossEntropy,
        0.05,
        activationFuncs.relu,
        activationFuncs.softmax
      );

      // Train
      for (let i = 0; i < 5000; i++) {
        const data = generateColorData(1);
        brain.train(data[0].input, data[0].target);
      }

      // Test with clear cases
      expect(getPredictedClass(brain.guess([1, 0, 0]))).toBe(0); // Pure red
      expect(getPredictedClass(brain.guess([0, 1, 0]))).toBe(1); // Pure green
      expect(getPredictedClass(brain.guess([0, 0, 1]))).toBe(2); // Pure blue
      
      expect(getPredictedClass(brain.guess([0.9, 0.1, 0.1]))).toBe(0); // Mostly red
      expect(getPredictedClass(brain.guess([0.1, 0.9, 0.1]))).toBe(1); // Mostly green
      expect(getPredictedClass(brain.guess([0.1, 0.1, 0.9]))).toBe(2); // Mostly blue
    }, 10000);
  });

  describe('Confidence and Probability', () => {
    test('high confidence for clear cases', () => {
      const brain = new JellyBrain(
        2, 6, 3,
        costFuncs.crossEntropy,
        0.1,
        activationFuncs.sigmoid,
        activationFuncs.softmax
      );

      // Generate simple data: x-coordinate determines class
      const generateSimpleData = () => {
        const x = Math.random();
        const y = Math.random();
        let target;
        if (x < 0.33) target = [1, 0, 0];
        else if (x < 0.66) target = [0, 1, 0];
        else target = [0, 0, 1];
        return { input: [x, y], target };
      };

      // Train
      for (let i = 0; i < 4000; i++) {
        const data = generateSimpleData();
        brain.train(data.input, data.target);
      }

      // Test confidence
      const output1 = brain.guess([0.1, 0.5]); // Clearly class 0
      expect(output1[0]).toBeGreaterThan(0.7);
      
      const output2 = brain.guess([0.5, 0.5]); // Clearly class 1
      expect(output2[1]).toBeGreaterThan(0.7);
      
      const output3 = brain.guess([0.9, 0.5]); // Clearly class 2
      expect(output3[2]).toBeGreaterThan(0.7);
    }, 10000);
  });
});