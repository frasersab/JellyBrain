var fs = require('fs');
var path = require('path');
const imagesData = require('./custom_images.json');
const {createCanvas} = require('canvas');

function readCustom(start, end)
{
    return imagesData.slice(start, end);
}

function saveCustom(start, end)
{
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    imagesDataSlice = readCustom(start, end);

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
        const imagesDir = path.join(__dirname, 'images');
        if (!fs.existsSync(imagesDir)) {
            fs.mkdirSync(imagesDir, { recursive: true });
        }
        const imagePath = path.join(imagesDir, `image${image.index}-${label}.png`);
        fs.writeFileSync(imagePath, buffer)
    })
}

saveCustom(0, 49);

exports.readCustom = readCustom
exports.saveCustom = saveCustom