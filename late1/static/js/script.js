document.getElementById('predict-form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    // Prepare form data
    const formData = new FormData(this);

    // Send data to the server
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = data.prediction;
    })
    .catch(error => console.error('Error:', error));
});
