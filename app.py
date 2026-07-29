import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# File name as uploaded
MODEL_FILENAME = 'AdaBoost_model_model.pkl'
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

# Load the trained AdaBoost model
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model from {MODEL_PATH}: {e}")

# HTML + CSS Template embedded directly inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-main: #0b1329;
            --card-bg: #1e293b;
            --input-bg: #0f172a;
            --border-color: #334155;
            --primary-accent: #3b82f6;
            --primary-hover: #2563eb;
            --emerald-accent: #10b981;
            --danger-accent: #ef4444;
            --text-primary: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background: radial-gradient(circle at top right, #1e293b, #0b1329 70%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            width: 100%;
            max-width: 880px;
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            padding: 2.5rem 2rem;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        form {
            padding: 2rem;
        }

        .grid-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 1.25rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .input-wrapper {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-wrapper i {
            position: absolute;
            left: 1rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .input-wrapper input,
        .input-wrapper select {
            width: 100%;
            padding: 0.75rem 1rem 0.75rem 2.5rem;
            background: var(--input-bg);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            color: var(--text-primary);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        .input-wrapper input:focus,
        .input-wrapper select:focus {
            border-color: var(--emerald-accent);
            box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.2);
        }

        .btn-submit {
            margin-top: 2rem;
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #10b981, #059669);
            border: none;
            border-radius: 10px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn-submit:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 20px -5px rgba(16, 185, 129, 0.4);
        }

        .result-card {
            margin: 2rem 2rem 0;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            animation: fadeIn 0.4s ease-in-out;
        }

        .result-card.churn {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #fca5a5;
        }

        .result-card.retained {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #6ee7b7;
        }

        .result-title {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .error-card {
            margin: 2rem 2rem 0;
            padding: 1rem;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--danger-accent);
            border-radius: 10px;
            color: #fca5a5;
            text-align: center;
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
            <h1><i class="fa-solid fa-chart-pie"></i> Customer Churn Intelligence</h1>
            <p>AdaBoost Classification & Risk Prediction Model</p>
        </div>

        {% if error %}
        <div class="error-card">
            <i class="fa-solid fa-triangle-exclamation"></i> {{ error }}
        </div>
        {% endif %}

        {% if prediction is defined %}
        <div class="result-card {{ 'churn' if prediction == 1 else 'retained' }}">
            <div class="result-title">
                {% if prediction == 1 %}
                    <i class="fa-solid fa-user-xmark"></i> High Risk: Customer Likely to Churn
                {% else %}
                    <i class="fa-solid fa-user-check"></i> Low Risk: Customer Likely Retained
                {% endif %}
            </div>
            {% if probability %}
            <p>Confidence: <strong>{{ probability }}%</strong></p>
            {% endif %}
        </div>
        {% endif %}

        <form action="/" method="POST">
            <div class="grid-container">
                
                <div class="input-group">
                    <label>Age</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-user"></i>
                        <input type="number" name="age" step="1" min="18" max="100" required value="{{ form_data.age if form_data else '' }}" placeholder="e.g. 35">
                    </div>
                </div>

                <div class="input-group">
                    <label>Gender</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-venus-mars"></i>
                        <select name="gender" required>
                            <option value="0" {% if form_data and form_data.gender == '0' %}selected{% endif %}>Female</option>
                            <option value="1" {% if form_data and form_data.gender == '1' %}selected{% endif %}>Male</option>
                        </select>
                    </div>
                </div>

                <div class="input-group">
                    <label>Tenure (Months)</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-calendar-days"></i>
                        <input type="number" name="tenure" step="1" min="0" required value="{{ form_data.tenure if form_data else '' }}" placeholder="e.g. 12">
                    </div>
                </div>

                <div class="input-group">
                    <label>Usage Frequency</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-chart-line"></i>
                        <input type="number" name="usage_frequency" step="1" min="0" required value="{{ form_data.usage_frequency if form_data else '' }}" placeholder="e.g. 15">
                    </div>
                </div>

                <div class="input-group">
                    <label>Support Calls</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-headset"></i>
                        <input type="number" name="support_calls" step="1" min="0" required value="{{ form_data.support_calls if form_data else '' }}" placeholder="e.g. 2">
                    </div>
                </div>

                <div class="input-group">
                    <label>Payment Delay (Days)</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-clock"></i>
                        <input type="number" name="payment_delay" step="1" min="0" required value="{{ form_data.payment_delay if form_data else '' }}" placeholder="e.g. 3">
                    </div>
                </div>

                <div class="input-group">
                    <label>Subscription Type</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-layer-group"></i>
                        <select name="subscription_type" required>
                            <option value="0" {% if form_data and form_data.subscription_type == '0' %}selected{% endif %}>Basic</option>
                            <option value="1" {% if form_data and form_data.subscription_type == '1' %}selected{% endif %}>Standard</option>
                            <option value="2" {% if form_data and form_data.subscription_type == '2' %}selected{% endif %}>Premium</option>
                        </select>
                    </div>
                </div>

                <div class="input-group">
                    <label>Contract Length</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-file-contract"></i>
                        <select name="contract_length" required>
                            <option value="0" {% if form_data and form_data.contract_length == '0' %}selected{% endif %}>Monthly</option>
                            <option value="1" {% if form_data and form_data.contract_length == '1' %}selected{% endif %}>Annual</option>
                            <option value="2" {% if form_data and form_data.contract_length == '2' %}selected{% endif %}>Quarterly</option>
                        </select>
                    </div>
                </div>

                <div class="input-group">
                    <label>Total Spend ($)</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-dollar-sign"></i>
                        <input type="number" name="total_spend" step="0.01" min="0" required value="{{ form_data.total_spend if form_data else '' }}" placeholder="e.g. 500.00">
                    </div>
                </div>

                <div class="input-group">
                    <label>Last Interaction (Days)</label>
                    <div class="input-wrapper">
                        <i class="fa-solid fa-handshake"></i>
                        <input type="number" name="last_interaction" step="1" min="0" required value="{{ form_data.last_interaction if form_data else '' }}" placeholder="e.g. 10">
                    </div>
                </div>

            </div>

            <button type="submit" class="btn-submit">
                <i class="fa-solid fa-bolt"></i> Run Risk Prediction
            </button>
        </form>
    </div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template_string(HTML_TEMPLATE)

    if model is None:
        return render_template_string(
            HTML_TEMPLATE,
            error=f"Model file '{MODEL_FILENAME}' could not be loaded."
        )

    try:
        # Extract the exact feature fields defined in your model pickle
        age = float(request.form.get('age', 0))
        gender = int(request.form.get('gender', 0))
        tenure = float(request.form.get('tenure', 0))
        usage_frequency = float(request.form.get('usage_frequency', 0))
        support_calls = float(request.form.get('support_calls', 0))
        payment_delay = float(request.form.get('payment_delay', 0))
        subscription_type = int(request.form.get('subscription_type', 0))
        contract_length = int(request.form.get('contract_length', 0))
        total_spend = float(request.form.get('total_spend', 0))
        last_interaction = float(request.form.get('last_interaction', 0))

        features = np.array([[
            age, gender, tenure, usage_frequency, support_calls,
            payment_delay, subscription_type, contract_length,
            total_spend, last_interaction
        ]])

        prediction = model.predict(features)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            probability = round(float(np.max(probs)) * 100, 2)

        return render_template_string(
            HTML_TEMPLATE,
            prediction=int(prediction),
            probability=probability,
            form_data=request.form
        )

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
