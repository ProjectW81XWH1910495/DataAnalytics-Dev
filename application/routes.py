from application import application
from flask import render_template, request, url_for, redirect
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
from werkzeug.utils import secure_filename


@application.route('/')
def index():
    return render_template('scatter.html')

@application.route('/trend')
def trend():
    return render_template('trend.html')
@application.route('/scatter')
def scatter():
    return render_template('scatter.html')
@application.route('/threeD')
def threeD():
    return render_template('threeD.html')
@application.route('/auc')
def auc():
    return render_template('auc.html')

@application.route('/chart1', methods=['POST','GET'])
def chart1():

    if request.method == "POST":
        x_label = request.form.get("x_label", None)
        y_label = request.form.get("y_label", None)

        if request.files:
            f = request.files['dataset']
            if str(secure_filename(f.filename)) != "":
                upload_path = "dataset/retrieve/" + str(secure_filename(f.filename))
                print(upload_path)
                f.save(upload_path)
                dataset_name = str(secure_filename(f.filename))
            else:
                dataset_name = request.form.get("dataset_name", None) + ".csv"


    # Graph One
        if x_label and y_label and dataset_name:
            df = pd.read_csv("dataset/retrieve/"+str(dataset_name))
            title = "Dataset Name:" + str(dataset_name)
            fig1 = px.scatter(x= df[x_label], y=df[y_label], color = df[y_label],title= title,labels=dict(x=x_label, y=y_label))
            graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('scatter.html', graph1JSON=graph1JSON)

@application.route('/chart2', methods=['POST','GET'])
def chart2():
        if request.method == "POST":
            x_label = request.form.get("x_label", None)
            y_label = request.form.get("y_label", None)
            if request.files:
                f = request.files['dataset']
                if str(secure_filename(f.filename)) != "":
                    upload_path = "dataset/retrieve/" + str(secure_filename(f.filename))
                    f.save(upload_path)
                    dataset_name = str(secure_filename(f.filename))
                else:
                    dataset_name = request.form.get("dataset_name", None) + ".csv"
            # Graph Two
            if x_label and y_label and dataset_name:
                df = pd.read_csv("dataset/retrieve/" + str(dataset_name))
                title = "Dataset Name:" + str(dataset_name)
                fig2 = px.line(x=df[x_label], y=df[y_label], title=title, labels=dict(x=x_label, y=y_label))
                graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('trend.html', graph2JSON=graph2JSON)

@application.route('/chart3', methods=['POST','GET'])
def chart3():
        if request.method == "POST":
            x_label = request.form.get("x_label", None)
            y_label = request.form.get("y_label", None)
            z_label = request.form.get("z_label", None)
            if request.files:
                f = request.files['dataset']
                if str(secure_filename(f.filename)) != "":
                    upload_path = "dataset/retrieve/" + str(secure_filename(f.filename))
                    f.save(upload_path)
                    dataset_name = str(secure_filename(f.filename))
                else:
                    dataset_name = request.form.get("dataset_name", None) + ".csv"
            if x_label and y_label and dataset_name:
                df = pd.read_csv("dataset/retrieve/" + str(dataset_name))
                title = "Dataset Name:" + str(dataset_name)
                fig3 = px.scatter_3d(x=df[x_label], y=df[y_label], z=df[z_label],title=title,color=df[z_label], labels=dict(x=x_label, y=y_label, z=z_label))
                graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('threeD.html', graph3JSON=graph3JSON)
@application.route('/chart4', methods=['POST','GET'])
def chart4():
        if request.method == "POST":
            x_label = request.form.get("x_label", None)
            y_label = request.form.get("y_label", None)
            distinct_thresholds = request.form.get("distinct_thresholds", None)
            print(distinct_thresholds.split(","))
            thresholds_number = request.form.get("thresholds_number", None)
            if request.files:
                f = request.files['dataset']
                if str(secure_filename(f.filename)) != "":
                    upload_path = "dataset/ROC/" + str(secure_filename(f.filename))
                    f.save(upload_path)
                    dataset_name = str(secure_filename(f.filename))
                else:
                    dataset_name = request.form.get("dataset_name", None) + ".csv"
            if x_label and y_label and dataset_name:
                df = pd.read_csv("dataset/ROC/" + str(dataset_name))
                pre = np.array(df['pre']).tolist()
                validation = np.array(df['validation']).tolist()
                if distinct_thresholds and thresholds_number == "distinct thresholds":
                    thresholds = distinct_thresholds.split(",")
                    thresholds = list(map(float,thresholds))
                    print(thresholds)
                else:
                    thresholds = sorted(np.linspace(0,1,int(thresholds_number)),reverse =True)
                fpr = []
                tpr = []
                #FPR = FP / (FP + TN)
                #TPR = TP / (TP + FN)

                for i in range(len(thresholds)):
                    FP,TP,TN,FN = 0,0,0,0
                    tmp = [0]*len(pre)
                    for j in range(len(validation)):
                        if pre[j]>thresholds[i]:
                            tmp[j] = 1
                    for a in range(len(tmp)):
                        if tmp[a] == 1 and tmp[a] == validation[a]:
                            TP+=1
                        elif tmp[a] == 0 and tmp[a]==validation[a]:
                            TN+=1
                        elif tmp[a] == 1 and tmp[a] != validation[a]:
                            FP+=1
                        else:
                            FN+=1
                    fpr.append(FP / (FP + TN))
                    tpr.append(TP / (TP + FN))
                print(fpr)
                print(tpr)

                title = "Dataset Name:" + str(dataset_name)
                fig4 = px.line(x=fpr, y=tpr,title=title, labels=dict(x=x_label, y=y_label))
                graph4JSON = json.dumps(fig4, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('auc.html', graph4JSON=graph4JSON)
# @app.route('/upload', methods = ['POST','GET'])
# def upload():
#     if request.method == 'POST':
#         f = request.files['file']
#         print(str(secure_filename(f.filename)))
#         upload_path = "dataset/retrieve/" + str(secure_filename(f.filename))
#         f.save(upload_path)





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


    #return render_template('scatter.html', graph1JSON=graph1JSON,  graph2JSON=graph2JSON, graph3JSON=graph3JSON)
    #return render_template('scatter.html')