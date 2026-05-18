from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load everything
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return "✅ ML Model API is running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    input_data = pd.DataFrame([data])

    # ✅ Apply encoders (IMPORTANT)
    for col in ['Gender','Symptom_1','Symptom_2','Symptom_3','Treatment_Plan']:
        try:
            input_data[col] = encoders[col].transform(input_data[col])
        except:
            return jsonify({"error": f"Invalid value in {col}"})

    # Ensure column order (VERY IMPORTANT)
    expected_columns = [
        'Age','Gender','Symptom_1','Symptom_2','Symptom_3',
        'Body_Temperature_C','Oxygen_Saturation_%',
        'Heart_Rate_bpm','Treatment_Plan','Systolic_BP','Diastolic_BP'
    ]

    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Decode output
    result = encoders['Diagnosis'].inverse_transform(prediction)

    return jsonify({
        "prediction": result[0]
    })

if __name__ == "__main__":
    app.run(debug=True)
