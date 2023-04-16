from flask import Flask, request, render_template
import pickle
import numpy as np

model = pickle.load(open('svm.pkl','rb'))
le = pickle.load(open('le.pkl','rb'))
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sepal_length = request.form['sl']
        sepal_width = request.form['sw']
        petal_length = request.form['pl']
        petal_width = request.form['pw']

        result = model.predict(np.array([sepal_length,sepal_width,petal_length,petal_width]).reshape(1,4))
        if result[0] == 0:
            iris = 'Iris-setosa'
        elif result[0] == 1:
            iris = 'Iris-versicolor'
        else:
            iris = 'Iris-virginica'        
    
        return f'<h1 style="text-align:center;margin-top:200px">The Iris is classified as {iris}</h1>'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)