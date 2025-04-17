async function checkSMS() {
    const sms = document.getElementById('smsInput').value;
    const res = await fetch('http://127.0.0.1:5000/predict_sms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sms })
    });

    const data = await res.json();
    document.getElementById('result').textContent = data.prediction;

    const contribDiv = document.getElementById('contributions');
    contribDiv.innerHTML = '<h4>Feature Contributions:</h4><ul>' +
        Object.entries(data.contributions)
              .map(([feature, value]) => `<li>${feature}: ${value.toFixed(4)}</li>`)
              .join('') + '</ul>';
}
