var fs = require('fs');
const {createCanvas} = require('canvas');

function readMNIST(start, end)
{
    var dataFileBuffer = fs.readFileSync(__dirname + '\\test_images_10k.idx3-ubyte');
    var labelFileBuffer = fs.readFileSync(__dirname + '\\test_labels_10k.idx1-ubyte');
    var pixelValues = [];
    
    for (var image = start; image < end; image++)
    { 
        var pixels = [];
        for (var y = 0; y <= 27; y++)
        {
            for (var x = 0; x <= 27; x++)
            {
                pixels.push(dataFileBuffer[(image * 28 * 28) + (x + (y * 28)) + 16]);
            }
        }

        var imageData  = {};
        imageData["index"] = image;
        imageData["label"] = labelFileBuffer[image + 8];
        imageData["pixels"] = pixels;
        pixelValues.push(imageData);
    }
    return pixelValues;
}

function saveMNIST(start, end)
{
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');

    var pixelValues = readMNIST(start, end);

    pixelValues.forEach(function(image)
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
        fs.writeFileSync(__dirname + `\\images\\image${image.index}-${image.label}.png`, buffer)
    })
}

saveMNIST(0, 5);