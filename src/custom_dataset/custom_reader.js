var fs = require('fs');
var path = require('path');
const imagesData = require('./custom_images.json');

function readCustom(start, end)
{
    return imagesData.slice(start, end);
}

function saveCustom(start, end)
{
    const {createCanvas} = require('canvas');
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    imagesDataSlice = readCustom(start, end);

    const imagesDir = path.join(__dirname, 'images');
    if (!fs.existsSync(imagesDir)) {
        fs.mkdirSync(imagesDir, { recursive: true });
    }

    imagesDataSlice.forEach(function(image)
    {
        pixels = image.pixels
        label = image.label

        for (var y = 0; y <= 27; y++)
        {
            for (var x = 0; x <= 27; x++)
            {
                var pixel = pixels[x + (y * 28)];
                var colour = 255 - pixel;
                ctx.fillStyle = `rgb(${colour}, ${colour}, ${colour})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        const buffer = canvas.toBuffer('image/png')
        const imagePath = path.join(imagesDir, `image${image.index}-${label}.png`);
        fs.writeFileSync(imagePath, buffer)
    })
    
    console.log(`Saved ${end - start} custom images to ${imagesDir}`);
}

if (require.main === module) {
    const args = process.argv.slice(2);
    
    if (args.length < 2) {
        console.log('Usage: node custom_reader.js <start> <end>');
        console.log('  start: Starting index (0-based)');
        console.log('  end: Ending index (exclusive)');
        console.log('\nExample: node custom_reader.js 0 10');
        console.log(`\nAvailable images: 0-${imagesData.length - 1}`);
        process.exit(1);
    }

    const start = parseInt(args[0]);
    const end = parseInt(args[1]);

    if (start < 0 || end > imagesData.length) {
        console.error(`Error: Index out of range. Available images: 0-${imagesData.length - 1}`);
        process.exit(1);
    }

    try {
        saveCustom(start, end);
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

exports.readCustom = readCustom
exports.saveCustom = saveCustom
