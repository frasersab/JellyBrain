var fs = require('fs');
const { createCanvas, loadImage } = require('canvas');
const { ctransposeDependencies } = require('mathjs');

var dataFileBuffer = fs.readFileSync('src\\mnist_dataset\\test_images_10k.idx3-ubyte');
var labelFileBuffer = fs.readFileSync('src\\mnist_dataset\\test_labels_10k.idx1-ubyte');
const canvas = createCanvas(28, 28)
const ctx = canvas.getContext('2d')
var pixelValues = [];


for (var image = 0; image < 3; image++) { 
    var pixels = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (var y = 0; y <= 27; y++) {
        for (var x = 0; x <= 27; x++) {
            var pixel = dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16];
            pixels.push(pixel);
            var colour = 255 - pixel;
            ctx.fillStyle = `rgb(${colour}, ${colour}, ${colour})`;
            ctx.fillRect(x, y, 1, 1);
        }
    }

    var label = labelFileBuffer[image + 8];
    var imageData  = {};
    imageData[JSON.stringify(label)] = pixels;
    pixelValues.push(imageData);

    const buffer = canvas.toBuffer('image/png')
    fs.writeFileSync(`./image${image}-${label}.png`, buffer)
}