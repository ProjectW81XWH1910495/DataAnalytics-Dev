from application import application
from flask import render_template, request, url_for, redirect
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import os
from werkzeug.utils import secure_filename
from scipy import stats
from scipy.stats import chi2_contingency
import collections
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from functools import partial




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
@application.route('/wil')
def wil():
    return render_template('wil.html')
@application.route('/chi')
def chi():
    return render_template('chi.html')
@application.route('/fi')
def fi():
    return render_template('fi.html')

@application.route('/chart1', methods=['POST','GET'])
def chart1():
    graphJSON = {}
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
                dataset_name = request.form.get("dataset_name", None)
                if ".csv" in dataset_name:
                    dataset_name = dataset_name
                else:
                    dataset_name = request.form.get("dataset_name", None) + ".csv"


    # Graph One
        if x_label and y_label and dataset_name:
            print(x_label)
            print(y_label)
            df = pd.read_csv("dataset/retrieve/" + str(dataset_name))
            heads= list(df.columns)
            table = df.values.tolist()[:100]
            # print(head)
            # print(table)
            title = "Dataset Name:" + str(dataset_name)
            if y_label == "mean_test_auc and percent_auc_diff":
                fig1 = px.scatter(x=df[x_label], y=df["mean_test_auc"], color=df["mean_test_auc"], title=title,
                                  labels=dict(x=x_label, y="mean_test_auc"))
                graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
                fig2 = px.scatter(x=df[x_label], y=df["percent_auc_diff"], color=df["percent_auc_diff"], title=title,
                                  labels=dict(x=x_label, y="percent_auc_diff"))
                graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
            else:

                fig1 = px.scatter(x= df[x_label], y=df[y_label], color = df[y_label],title= title,labels=dict(x=x_label, y=y_label))
                graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
                graph2JSON = "undefined"
        #os.remove("dataset/retrieve/" + str(dataset_name))
    return render_template('scatter.html', graph1JSON=graph1JSON, graph2JSON=graph2JSON, dataset_name = dataset_name, table = table, heads= heads)

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
                    dataset_name = request.form.get("dataset_name", None)
                    if ".csv" in dataset_name:
                        dataset_name = dataset_name
                    else:
                        dataset_name = request.form.get("dataset_name", None) + ".csv"
            # Graph Two
            if x_label and y_label and dataset_name:
                df = pd.read_csv("dataset/retrieve/" + str(dataset_name))
                title = "Dataset Name:" + str(dataset_name)
                if y_label == "mean_test_auc and percent_auc_diff":
                    fig1 = px.line(x=df[x_label], y=df["mean_test_auc"], title=title, labels=dict(x=x_label, y="mean_test_auc"))
                    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
                    fig2 = px.line(x=df[x_label], y=df["percent_auc_diff"], title=title, labels=dict(x=x_label, y="percent_auc_diff"))
                    graph2JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
                else:
                    fig1 = px.line(x=df[x_label], y=df[y_label], title=title,
                                   labels=dict(x=x_label, y=y_label))
                    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
                    graph2JSON = "None"

        #os.remove("dataset/retrieve/" + str(dataset_name))
        return render_template('trend.html', graph1JSON=graph1JSON,graph2JSON=graph2JSON,dataset_name = dataset_name)

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
                    dataset_name = request.form.get("dataset_name", None)
                    if ".csv" in dataset_name:
                        dataset_name = dataset_name
                    else:
                        dataset_name = request.form.get("dataset_name", None) + ".csv"
            if x_label and y_label and dataset_name:
                df = pd.read_csv("dataset/retrieve/" + str(dataset_name))
                title = "Dataset Name:" + str(dataset_name)
                if z_label == "mean_test_auc and percent_auc_diff":
                    fig1 = px.scatter_3d(x=df[x_label], y=df[y_label], z=df["mean_test_auc"],title=title,color=df["mean_test_auc"], labels=dict(x=x_label, y=y_label, z="mean_test_auc"))
                    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
                    fig2 = px.scatter_3d(x=df[x_label], y=df[y_label], z=df["percent_auc_diff"], title=title, color=df["percent_auc_diff"],
                                         labels=dict(x=x_label, y=y_label, z="percent_auc_diff"))
                    graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
                else:
                    fig1 = px.scatter_3d(x=df[x_label], y=df[y_label], z=df[z_label], title=title, color=df[z_label],
                                         labels=dict(x=x_label, y=y_label, z=z_label))
                    graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)
                    graph2JSON = "None"
            #os.remove("dataset/retrieve/" + str(dataset_name))
        return render_template('threeD.html', graph1JSON=graph1JSON,graph2JSON=graph2JSON, dataset_name = dataset_name)
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

@application.route('/chart5', methods=['POST','GET'])
def chart5():
    graphJSON = {}
    if request.method == "POST":
        Sample1 = request.form.get("Sample1", None)
        Sample2 = request.form.get("Sample2", None)

        if request.files:
            f = request.files['dataset']
            if str(secure_filename(f.filename)) != "":
                upload_path = "dataset/retrieve/" + str(secure_filename(f.filename))
                print(upload_path)
                f.save(upload_path)
                dataset_name = str(secure_filename(f.filename))
            else:
                dataset_name = request.form.get("dataset_name", None)
                if ".csv" in dataset_name:
                    dataset_name = dataset_name
                else:
                    dataset_name = request.form.get("dataset_name", None) + ".csv"

        if Sample1 and Sample2 and dataset_name:
            print("Sample 1"+Sample1)
            print("Sample 2" + Sample2)

            df = pd.read_csv("dataset/retrieve/" + str(dataset_name))
            heads= list(df.columns)
            table = df.values.tolist()[:100]
            # print(heads)
            # print(table)
            print(df[Sample1])
            print(df[Sample2])
            title = "Dataset Name:" + str(dataset_name)
            result = stats.wilcoxon(df[Sample1], df[Sample2])

            print(result)
    return render_template('wil.html', result=result, dataset_name=dataset_name, table=table, heads=heads)

@application.route('/chart6', methods=['POST','GET'])
def chart6():
    graphJSON = {}
    if request.method == "POST":
        Sample1 = request.form.get("Sample1", None)
        Sample2 = request.form.get("Sample2", None)
        condition = request.form.get("Condition", None)
        if request.files:
            f = request.files['dataset']
            if str(secure_filename(f.filename)) != "":
                upload_path = "dataset/retrieve/" + str(secure_filename(f.filename))
                print(upload_path)
                f.save(upload_path)
                dataset_name = str(secure_filename(f.filename))
            else:
                dataset_name = request.form.get("dataset_name", None)
                if ".csv" in dataset_name:
                    dataset_name = dataset_name
                else:
                    dataset_name = request.form.get("dataset_name", None) + ".csv"

        if Sample1 and Sample2 and dataset_name:
            print("Sample 1"+Sample1)
            print("Sample 2" + Sample2)

            df = pd.read_csv("dataset/retrieve/" + str(dataset_name))
            heads= list(df.columns)
            table = df.values.tolist()[:100]
            title = "Dataset Name:" + str(dataset_name)
            condition_variable_list = list(set(df[condition]))
            print(condition_variable_list)
            print(len(condition_variable_list))
            result_list = [[] for i in range(len(condition_variable_list))]
            total = []
            t,p = 0,0
            for i in range(len(condition_variable_list)):
                condition_table = df[df[condition] == condition_variable_list[i]]
                contingency = pd.crosstab(index=condition_table[Sample1], columns=condition_table[Sample2])
                chi_val, p_val, dof, expected = chi2_contingency(contingency, lambda_="log-likelihood", correction=False)

                result_list[i].append(condition_variable_list[i])
                result_list[i].append(chi_val)
                t += chi_val
                result_list[i].append(p_val)
                p += p_val


            print(result_list)
            total.append(p)
            total.append(t)
    return render_template('chi.html',total = total, dataset_name=dataset_name, table=table, result = result_list)

@application.route('/feature', methods=['POST','GET'])
def feature():
    if request.method == "POST":
        target = request.form.get("target", None)
        k = request.form.get("k", None)
        smethod = request.form.get("smethod", None)
        if request.files:
            f = request.files['dataset']
            if str(secure_filename(f.filename)) != "":
                upload_path = "dataset/original" + str(secure_filename(f.filename))
                f.save(upload_path)
                dataset_name = str(secure_filename(f.filename))
            else:
                dataset_name = request.form.get("dataset_name", None) + ".csv"
        if target and k and smethod and dataset_name:
            df = pd.read_csv("dataset/original/" + str(dataset_name))
            print(df)
            y = df[target]
            x = df.drop(target, 1)
            print(k)
            if smethod == "chi2":
                bestfeatures = SelectKBest(score_func=chi2, k=int(k))
            elif smethod == "f_classif":
                bestfeatures = SelectKBest(score_func=f_classif, k=int(k))
            else:
                bestfeatures = SelectKBest(score_func=mutual_info_classif, k=int(k))


            #bestfeatures = SelectKBest(score_func=partial(mutual_info_classif, random_state=0),k= int(k))
            #
            fit = bestfeatures.fit(x, y)
            dfscores = pd.DataFrame(fit.scores_)
            dfcolumns = pd.DataFrame(x.columns)
            # concat two dataframes for better visualization
            featureScores = pd.concat([dfcolumns, dfscores], axis=1)
            featureScores.columns = ['Feature', 'Score']  # naming the dataframe columns
            #print(featureScores.nlargest(10, 'Score'))
            print(featureScores)

            title = "Dataset Name:" + str(dataset_name)
            fig_fi = px.bar(featureScores.nlargest(int(k), 'Score'), y='Score', x='Feature', text_auto='.2s',
                         title="Feature Importance")
            fig_fi.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
            graphJSON_fi = json.dumps(fig_fi, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('fi.html', graphJSON_fi=graphJSON_fi)


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