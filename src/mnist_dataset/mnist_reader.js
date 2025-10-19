var fs = require('fs');
var path = require('path');
const {createCanvas} = require('canvas');

function readMNIST(start, end, imageFile, labelFile, squished = false)
{
    var dataFileBuffer = fs.readFileSync(__dirname + imageFile);
    var labelFileBuffer = fs.readFileSync(__dirname + labelFile);
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
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');

    var imagesData = readMNIST(start, end, imageFile, labelFile);

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
        const imagesDir = path.join(__dirname, 'images');
        if (!fs.existsSync(imagesDir)) {
            fs.mkdirSync(imagesDir, { recursive: true });
        }
        const imagePath = path.join(imagesDir, `image${image.index}-${image.label}.png`);
        fs.writeFileSync(imagePath, buffer)
    })
}

saveMNIST(0, 5, '\\test_images_10k.idx3-ubyte', '\\test_labels_10k.idx1-ubyte');

exports.readMNIST = readMNIST
exports.saveMNIST = saveMNIST