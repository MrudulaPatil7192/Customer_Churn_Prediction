import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the trained AdaBoost model
MODEL_PATH = "Adaboost_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    print(f"Warning: '{MODEL_PATH}' not found. Please place it in the same directory.")

# Feature names identified from your model file:
# 1. Age (Numeric)
# 2. Gender (0 = Female, 1 = Male)
# 3. Tenure (Numeric - Months)
# 4. Usage Frequency (Numeric - Times/Month)
# 5. Support Calls (Numeric)
# 6. Payment Delay (Numeric - Days)
# 7. Subscription Type (0 = Basic, 1 = Standard, 2 = Premium)
# 8. Contract Length (0 = Monthly, 1 = Quarterly, 2 = Annual)
# 9. Total Spend (Numeric - USD)
# 10. Last Interaction (Numeric - Days)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(255, 255, 255, 0.05);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-purple: #8b5cf6;
            --accent-blue: #3b82f6;
            --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --input-bg: rgba(15, 23, 42, 0.6);
            --danger-glow: rgba(239, 68, 68, 0.2);
            --success-glow: rgba(34, 197, 94, 0.2);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(to right, #a7f3d0, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 0.02em;
        }

        .input-group input, .input-group select {
            width: 100%;
            padding: 0.75rem 1rem;
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s ease;
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--accent-purple);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
        }

        .input-group select option {
            background-color: #0f172a;
            color: #f8fafc;
        }

        .submit-btn {
            grid-column: 1 / -1;
            margin-top: 1rem;
            padding: 1rem;
            background: var(--accent-gradient);
            border: none;
            border-radius: 12px;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        }

        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
        }

        .submit-btn:active {
            transform: translateY(0);
        }

        .result-container {
            margin-top: 2rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            display: none;
            animation: fadeIn 0.4s ease forwards;
        }

        .result-container.show {
            display: block;
        }

        .result-container.churn {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            box-shadow: 0 0 25px var(--danger-glow);
        }

        .result-container.no-churn {
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.3);
            box-shadow: 0 0 25px var(--success-glow);
        }

        .result-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .result-container.churn .result-title { color: #f87171; }
        .result-container.no-churn .result-title { color: #4ade80; }

        .result-desc {
            font-size: 0.95rem;
            color: var(--text-muted);
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Customer Churn Predictor</h1>
        <p>AdaBoost Classification Inference Engine</p>
    </div>

    <form id="predictionForm">
        <div class="form-grid">
            <div class="input-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="Age" placeholder="e.g. 30" required min="18" max="100">
            </div>

            <div class="input-group">
                <label for="gender">Gender</label>
                <select id="gender" name="Gender" required>
                    <option value="0">Female</option>
                    <option value="1">Male</option>
                </select>
            </div>

            <div class="input-group">
                <label for="tenure">Tenure (Months)</label>
                <input type="number" id="tenure" name="Tenure" placeholder="e.g. 12" required min="0">
            </div>

            <div class="input-group">
                <label for="usage">Usage Frequency</label>
                <input type="number" id="usage" name="Usage Frequency" placeholder="e.g. 15" required min="0">
            </div>

            <div class="input-group">
                <label for="calls">Support Calls</label>
                <input type="number" id="calls" name="Support Calls" placeholder="e.g. 2" required min="0">
            </div>

            <div class="input-group">
                <label for="delay">Payment Delay (Days)</label>
                <input type="number" id="delay" name="Payment Delay" placeholder="e.g. 5" required min="0">
            </div>

            <div class="input-group">
                <label for="subscription">Subscription Type</label>
                <select id="subscription" name="Subscription Type" required>
                    <option value="0">Basic</option>
                    <option value="1">Standard</option>
                    <option value="2">Premium</option>
                </select>
            </div>

            <div class="input-group">
                <label for="contract">Contract Length</label>
                <select id="contract" name="Contract Length" required>
                    <option value="0">Monthly</option>
                    <option value="1">Quarterly</option>
                    <option value="2">Annual</option>
                </select>
            </div>

            <div class="input-group">
                <label for="spend">Total Spend ($)</label>
                <input type="number" step="0.01" id="spend" name="Total Spend" placeholder="e.g. 450.50" required min="0">
            </div>

            <div class="input-group">
                <label for="interaction">Last Interaction (Days ago)</label>
                <input type="number" id="interaction" name="Last Interaction" placeholder="e.g. 10" required min="0">
            </div>

            <button type="submit" class="submit-btn">Predict Churn Risk</button>
        </div>
    </form>

    <div id="resultBox" class="result-container">
        <div id="resultTitle" class="result-title"></div>
        <div id="resultDesc" class="result-desc"></div>
    </div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const formData = new FormData(this);
        const data = {};
        formData.forEach((value, key) => { data[key] = parseFloat(value); });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            const resultBox = document.getElementById('resultBox');
            const resultTitle = document.getElementById('resultTitle');
            const resultDesc = document.getElementById('resultDesc');

            resultBox.classList.remove('churn', 'no-churn', 'show');

            if (result.error) {
                resultTitle.innerText = "Error";
                resultDesc.innerText = result.error;
                resultBox.classList.add('churn', 'show');
                return;
            }

            if (result.prediction === 1) {
                resultTitle.innerText = "High Risk of Churn";
                resultDesc.innerText = "This customer is likely to cancel their subscription based on usage patterns.";
                resultBox.classList.add('churn');
            } else {
                resultTitle.innerText = "Low Risk of Churn";
                resultDesc.innerText = "This customer is likely to maintain an active subscription.";
                resultBox.classList.add('no-churn');
            }

            resultBox.classList.add('show');
        } catch (err) {
            console.error(err);
            alert("Prediction failed. Make sure Flask server is running.");
        }
    });
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file Adaboost_model.pkl not loaded.'}), 500

    try:
        data = request.json
        # Extract features in the exact sequence expected by AdaBoost
        features = [
            float(data['Age']),
            float(data['Gender']),
            float(data['Tenure']),
            float(data['Usage Frequency']),
            float(data['Support Calls']),
            float(data['Payment Delay']),
            float(data['Subscription Type']),
            float(data['Contract Length']),
            float(data['Total Spend']),
            float(data['Last Interaction'])
        ]

        prediction = model.predict([features])[0]
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
