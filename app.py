import numpy as np
import pickle
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained model (Ensure your .pkl file is renamed to model.pkl or update the filename here)
model = pickle.load(open("liner2.pkl", "rb"))

# HTML Template with attractive styling embedded directly
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Cost Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: #ffffff;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 500px;
        }
        h2 {
            color: #333333;
            margin-bottom: 24px;
            font-weight: 700;
            text-align: center;
            font-size: 26px;
        }
        .form-group {
            margin-bottom: 18px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            color: #555555;
            font-weight: 500;
            font-size: 14px;
        }
        input, select {
            width: 100%;
            padding: 12px 14px;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            font-size: 15px;
            transition: all 0.3s ease;
            background-color: #f9fafb;
        }
        input:focus, select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
            outline: none;
            background-color: #ffffff;
        }
        button {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.3s ease, transform 0.1s ease;
            margin-top: 10px;
        }
        button:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }
        button:active {
            transform: translateY(0);
        }
        .result-card {
            margin-top: 25px;
            padding: 20px;
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 8px;
            text-align: center;
        }
        .result-card h3 {
            color: #166534;
            font-size: 16px;
            margin-bottom: 6px;
        }
        .result-card p {
            color: #15803d;
            font-size: 24px;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Insurance Cost Predictor</h2>
        <form method="POST">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" id="age" name="age" required placeholder="e.g., 30">
            </div>
            <div class="form-group">
                <label for="sex">Sex</label>
                <select id="sex" name="sex" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label for="bmi">BMI (Body Mass Index)</label>
                <input type="number" step="0.1" id="bmi" name="bmi" required placeholder="e.g., 25.4">
            </div>
            <div class="form-group">
                <label for="children">Number of Children</label>
                <input type="number" id="children" name="children" required placeholder="e.g., 1">
            </div>
            <div class="form-group">
                <label for="smoker">Smoker</label>
                <select id="smoker" name="smoker" required>
                    <option value="1">Yes</option>
                    <option value="0">No</option>
                </select>
            </div>
            <div class="form-group">
                <label for="region">Region</label>
                <select id="region" name="region" required>
                    <option value="0">Northeast</option>
                    <option value="1">Northwest</option>
                    <option value="2">Southeast</option>
                    <option value="3">Southwest</option>
                </select>
            </div>
            <button type="submit">Predict Cost</button>
        </form>

        {% if prediction %}
        <div class="result-card">
            <h3>Estimated Medical Insurance Cost</h3>
            <p>${{ prediction }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        try:
            # Collect and convert form input values
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            bmi = float(request.form["bmi"])
            children = float(request.form["children"])
            smoker = float(request.form["smoker"])
            region = float(request.form["region"])

            # Structure features as a 2D array matching model expectation
            features = np.array([[age, sex, bmi, children, smoker, region]])
            
            # Predict
            pred = model.predict(features)[0]
            prediction = round(float(pred), 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
