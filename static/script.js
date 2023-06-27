const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const clearButton = document.getElementById('clear-btn');
const predictButton = document.getElementById('predict-btn');
const resultDiv = document.getElementById('result');

let isDrawing = false;

function startPosition(e) {
    isDrawing = true;
    draw(e);
}

function endPosition() {
    isDrawing = false;
}

function draw(e) {
    if (!isDrawing) return;
    context.lineWidth = 10;
    context.lineCap = 'round';
    context.strokeStyle = 'black';
    context.lineTo(e.offsetX, e.offsetY);
    context.stroke();
    context.beginPath();
    context.moveTo(e.offsetX, e.offsetY);
}

function clearCanvas() {
    context.clearRect(0, 0, canvas.width, canvas.height);
    resultDiv.innerHTML = '';
}

function script() {
    const image = canvas.toDataURL();
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({image})
    })
        .then(response => response.json())
        .then(data => {
            resultDiv.innerHTML = `Prediction: ${data.prediction}`;
        });
}

canvas.addEventListener('mousedown', startPosition);
canvas.addEventListener('mouseup', endPosition);
canvas.addEventListener('mousemove', draw);
clearButton.addEventListener('click', clearCanvas);
predictButton.addEventListener('click', script);
