from flask import Flask, render_template, request
import numpy as np
import pickle  

app = Flask(__name__)

# Load trained ML model
mod_2 = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index_1.html')
def Output(pred):
    if(pred==0):
        return "Rainy"
    elif(pred==1):
        return "Cloudy"
    elif(pred==2):
        return "Sunny"
    else:
        return "Snow"


@app.predict('/predict',methods=['POST'])

def predict():
    fea=[int[i] for i in request.form.values()] #get the values from Weather_Output files

    final=np.array(fea).reshape[1,-1] #[[3,4,...]]
    pred=mod_2.predict(final)
    output = Output(round(pred))
    return render_template('index_1.html',pred_text = "The Weather will be {}".format(output))

if __name__ == '__main__':
    app.run(debug=true)
'''
def predict_weather():
    if request.method == "POST":
        try:
            # Get form data
            temp = int(request.form["temp"])
            hum = int(request.form["hum"])
            ws = int(request.form["ws"])
            pre = int(request.form["pre"])
            cc = int(request.form["cc"])
            at = int(request.form["at"])
            uv = int(request.form["uv"])
            sea = int(request.form["sea"])
            vis = int(request.form["vis"])
            ll = int(request.form["ll"])

            # Convert input to array
            cd = np.array([[temp, hum, ws, pre, cc, at, uv, sea, vis, ll]])

            # Predict
            result = mod_2.predict(cd)

            # Map prediction result
            weather_conditions = {0: "Rainy", 1: "Cloudy", 2: "Sunny", 3: "Snow"}
            prediction = weather_conditions.get(result[0], "Unknown Weather")

            return render_template("index.html", prediction=prediction)

        except ValueError:
            return render_template("index.html", error="Invalid input! Please enter numbers.")

    return render_template("index.html", prediction=None, error=None)

if __name__ == "__main__":
    app.run(debug=True)
'''