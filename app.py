from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
import numpy as np
import sys
import keras
from keras.models import load_model
from keras.layers import Dense,Dropout
flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Heart Disease Prediction", 
		  description = "Predict the occurance of heart disease")

name_space = app.namespace('Predictor', description='Prediction using parameters API')

model = app.model('Prediction params', 
				  {'Age': fields.Float(required = False, 
				  							   description="Age", 
    					  				 	   help="Age of the specimen"),
				  'Sex': fields.Float(required = False, 
				  							   description="Sex", 
    					  				 	   help="1: male, 0: female"),
				  'ChestPain': fields.Float(required = False, 
				  							description="Chest Pain", 
    					  				 	help="chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic"),
				  'RestingBloodPressure': fields.Float(required = False, 
				  							description="RestingBloodPressure", 
    					  				 	help="resting blood pressure"),
				  'Cholestrol': fields.Float(required = False, 
				  							description="Cholestrol", 
    					  				 	help="Cholestrol"),						   
				  'FastingBloodSugar': fields.Float(required = False, 
				  							description="FastingBloodSugar", 
    					  				 	help="Fasting blood sugar"),						   
				  'RestingElectrocardiograph': fields.Float(required = False, 
				  							description="RestingElectrocardiograph", 
    					  				 	help="RestingElectrocardiograph"),						   
				  'MaximumHeartRateAcheived': fields.Float(required = False, 
				  							description="MaximumHeartRateAcheived", 
    					  				 	help="Maximu heart rate acheived "),						   
				  'Exang': fields.Float(required = False, 
				  							description="FastingBloodSugar", 
    					  				 	help="Angina raised due to exercise"),						   
				  'OldPeak': fields.Float(required = False, 
				  							description="OldPeak", 
    					  				 	help="OldPeak"),						   
				  'Slope': fields.Float(required = False, 
				  							description="Slope", 
    					  				 	help="Slope"),	
				  'OldPeak': fields.Float(required = False, 
				  							description="OldPeak", 
    					  				 	help="OldPeak"),								   	
				  'Ca': fields.Float(required = False, 
				  							description="ca", 
    					  				 	help="ca"),
				  'Thal': fields.Float(required = False, 
				  							description="thal", 
    					  				 	help="thal")								   		
											   })

#model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


@name_space.route("/")
class MainClass(Resource):


	@app.expect(model)		
	def post(self):
		try: 
			req = request.get_json()
			keras.backend.clear_session()
			model = load_model("HeartDiseasePredictionModel.h5")
			input=np.array([[req["Age"],req["Sex"],req["ChestPain"],req["RestingBloodPressure"],req["Cholestrol"],req["FastingBloodSugar"],req["RestingElectrocardiograph"],req["MaximumHeartRateAcheived"],req["Exang"],req["OldPeak"],req["Slope"],req["Slope"],req["Slope"]]])
			prediction=model.predict(input)
			response = jsonify({
				"statusCode": 200,
				"age":req,
				"input":input.tolist(),
				"status": "Predicted the heart disease",
				"result": prediction.tolist()
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
if __name__ == '__main__':
    flask_app.run()
