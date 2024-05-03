var fs = require('fs');
var imageData = require('./custom_images.json');
const {createCanvas} = require('canvas');


function saveCustom(index)
{
    const canvas = createCanvas(28, 28);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    pixels = imageData[index].pixels
    label = imageData[index].label

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
    fs.writeFileSync(__dirname + `\\images\\image${index}-${label}.png`, buffer)

}

saveCustom(6);

//exports.readCustom = readCustom
exports.saveCustom = saveCustom