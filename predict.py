from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

classifier = joblib.load('diabetic.joblib')

@app.route('/diabetic', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        Blood_Sugar = int(data.get('Blood_Sugar', 0))
        Carbohydrates = int(data.get('Carbohydrates', 0))
        Medicine_Type = int(data.get('Medicine_Type', ''))  # Note: Changing to string type
        Medicine_Dose = int(data.get('Medicine_Dose', 0))
        Exercise_Type = int(data.get('Exercise_Type', ''))  # Note: Changing to string type
        Exercise_Duration = float(data.get('Exercise_Duration', 0.0))
        Unusual_Event = int(data.get('Unusual_Event', '') ) # Note: Changing to string type
        
        # Create DataFrame with input data
        input_data = pd.DataFrame({
            'Blood_Sugar': [Blood_Sugar],
            'Carbohydrates': [Carbohydrates],
            'Medicine_Type': [Medicine_Type],
            'Medicine_Dose': [Medicine_Dose],
            'Exercise_Type': [Exercise_Type],
            'Exercise_Duration': [Exercise_Duration],
            'Unusual_Event': [Unusual_Event]
        })
        
        # Make predictions
        prediction = classifier.predict(input_data)
        print(prediction)
        
        return jsonify({'Diabetic_Status': int(prediction[0])})  # Returning only the predicted diabetes status
        
    except KeyError as e:
        return jsonify({'error': f'Missing required key: {e}'})
    except ValueError as e:
        return jsonify({'error': str(e)})
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {e}'})

if __name__ == '__main__':
    app.run(debug=True)




    
    #input_data =(1,67.0,0,1,5,27.32,6.5,200)

# changing the input_data to numpy array
#input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
#input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)




