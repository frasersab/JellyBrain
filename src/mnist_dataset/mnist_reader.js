var fs = require('fs');
var path = require('path');

function readMNIST(start, end, imageFile, labelFile, squished = false)
{
    var dataFileBuffer = fs.readFileSync(path.join(__dirname, imageFile));
    var labelFileBuffer = fs.readFileSync(path.join(__dirname, labelFile));
    var imagesData = [];
    
    for (var image = start; image < end; image++)
    { 
        var pixels = [];
        for (var y = 0; y <= 27; y++)
        {
            for (var x = 0; x <= 27; x++)
            {
                value = dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16];
                if (squished)
                {
                    value = value / 255;
                }
                pixels.push(value);
            }
        }

        var imageData  = {};
        imageData["index"] = image;
        imageData["label"] = labelFileBuffer[image + 8];
        imageData["pixels"] = pixels;
        imagesData.push(imageData);
    }
    return imagesData;
}

function saveMNIST(start, end, imageFile, labelFile)
{
    const {createCanvas} = require('canvas');
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');

    var imagesData = readMNIST(start, end, imageFile, labelFile);

    const imagesDir = path.join(__dirname, 'images');
    if (!fs.existsSync(imagesDir)) {
        fs.mkdirSync(imagesDir, { recursive: true });
    }

    imagesData.forEach(function(image)
    { 
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (var y = 0; y <= 27; y++)
        {
            for (var x = 0; x <= 27; x++)
            {
                var pixel = image.pixels[x + (y * 28)];
                var colour = 255 - pixel;
                ctx.fillStyle = `rgb(${colour}, ${colour}, ${colour})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        const buffer = canvas.toBuffer('image/png')
        const imagePath = path.join(imagesDir, `image${image.index}-${image.label}.png`);
        fs.writeFileSync(imagePath, buffer)
    })
    
    console.log(`Saved ${end - start} MNIST images to ${imagesDir}`);
}

if (require.main === module) {
    const args = process.argv.slice(2);
    
    if (args.length < 2) {
        console.log('Usage: node mnist_reader.js <start> <end> [dataset]');
        console.log('  start: Starting index (0-based)');
        console.log('  end: Ending index (exclusive)');
        console.log('  dataset: "train" or "test" (default: "test")');
        console.log('\nExample: node mnist_reader.js 0 10 test');
        process.exit(1);
    }

    const start = parseInt(args[0]);
    const end = parseInt(args[1]);
    const dataset = args[2] || 'test';
    
    const imageFile = dataset === 'train' ? 
        'train_images_60k.idx3-ubyte' : 
        'test_images_10k.idx3-ubyte';
    const labelFile = dataset === 'train' ? 
        'train_labels_60k.idx1-ubyte' : 
        'test_labels_10k.idx1-ubyte';

    try {
        saveMNIST(start, end, imageFile, labelFile);
    } catch (error) {
        if (error.message.includes('Cannot find module \'canvas\'')) {
            console.error('Error: canvas module is required for saving images');
            console.error('Install it with: npm install canvas');
        } else {
            console.error('Error:', error.message);
        }
        process.exit(1);
    }
}

exports.readMNIST = readMNIST
exports.saveMNIST = saveMNIST
