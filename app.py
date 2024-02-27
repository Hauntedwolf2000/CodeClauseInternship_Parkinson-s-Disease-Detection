from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load('parkinsons_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect user input from the form
        tremors = float(request.form['tremors'])
        voice_recording = float(request.form['voice_recording'])
        # Add more parameters as needed

        # Make prediction using the loaded model
        prediction = model.predict([[tremors, voice_recording]])[0]

        # Render the result template with prediction
        return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
