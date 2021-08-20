from application import app
from flask import render_template, request, url_for, redirect
import pandas as pd
import json
import plotly
import plotly.express as px



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/line')
def line():
    return render_template('line.html')
@app.route('/scatter')
def scatter():
    return render_template('index.html')
@app.route('/threeD')
def threeD():
    return render_template('threeD.html')

@app.route('/chart1', methods=['POST'])
def chart1():
    if request.method == "POST":
        x_label = request.form.get("x_label", None)
        y_label = request.form.get("y_label", None)
        dataset_name = request.form.get("dataset_name", None)
        print(x_label)
        print(y_label)
        print(dataset_name)

    # Graph One
        if x_label and y_label and dataset_name:
            df = pd.read_csv("dataset/retrieve/"+str(dataset_name)+".csv")
            title = "Dataset Name:" + str(dataset_name)
            fig1 = px.scatter(x= df[x_label], y=df[y_label], color = df[y_label],title= title,labels=dict(x=x_label, y=y_label))
            graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('index.html', graph1JSON=graph1JSON)

@app.route('/chart2', methods=['POST'])
def chart2():
        if request.method == "POST":
            x_label = request.form.get("x_label", None)
            y_label = request.form.get("y_label", None)
            dataset_name = request.form.get("dataset_name", None)
            # Graph Two
            if x_label and y_label and dataset_name:
                df = pd.read_csv("dataset/retrieve/" + str(dataset_name) + ".csv")
                title = "Dataset Name:" + str(dataset_name)
                fig2 = px.line(x=df[x_label], y=df[y_label], color=df[y_label],title=title, labels=dict(x=x_label, y=y_label))
                graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('line.html', graph2JSON=graph2JSON)

@app.route('/chart3', methods=['POST'])
def chart3():
        if request.method == "POST":
            x_label = request.form.get("x_label", None)
            y_label = request.form.get("y_label", None)
            z_label = request.form.get("z_label", None)
            dataset_name = request.form.get("dataset_name", None)
            # Graph Three
            if x_label and y_label and dataset_name:
                df = pd.read_csv("dataset/retrieve/" + str(dataset_name) + ".csv")
                title = "Dataset Name:" + str(dataset_name)
                fig3 = px.scatter_3d(x=df[x_label], y=df[y_label], z=df[z_label],title=title,color=df[z_label], labels=dict(x=x_label, y=y_label, z=z_label))
                graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('threeD.html', graph3JSON=graph3JSON)


    # Graph two
    # df = px.data.iris()
    # fig2 = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
    #           color='species',  title="Iris Dataset")
    # graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    #
    # # Graph three
    # df = px.data.gapminder().query("continent=='Oceania'")
    # fig3 = px.line(df, x="year", y="lifeExp", color='country',  title="Life Expectancy")
    # graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)


    #return render_template('index.html', graph1JSON=graph1JSON,  graph2JSON=graph2JSON, graph3JSON=graph3JSON)
    #return render_template('index.html')