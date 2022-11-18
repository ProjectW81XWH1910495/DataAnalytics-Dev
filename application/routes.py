from application import application
from flask import render_template, request, url_for, redirect, send_file, flash
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import matplotlib.pyplot as plt
import os
from werkzeug.utils import secure_filename
from scipy import stats
from scipy.stats import chi2_contingency
import collections
import sklearn.model_selection as model_selection
from sklearn.metrics import plot_confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from functools import partial
import shap
from retrieve_columns import expand_parameters_stored_in_one_column
from retrieve_columns import read_file,retrieve_target_columns_based_on_values
from retrieve_columns import find_columns_to_be_expanded, find_available_filters
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict



@application.route('/')
def index():
    return render_template('index.html')
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
@application.route('/feabar')
def feabar():
    return render_template('feabar.html')
@application.route('/he')
def he():
    return render_template('heatmap.html')

@application.route('/download/<path:data_path>')
def download(data_path):
    return send_file("../"+data_path, as_attachment=True)
# scatter
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
# trend
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
# 3d scatter
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
# ROC
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
# wil
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
# conditional chi-squared
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
# feature importance
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
@application.route('/featurebar', methods=['POST','GET'])
def featurebar():
    if request.method == "POST":
        target = request.form.get("target", None)
        if request.files:
            f = request.files['dataset']
            if str(secure_filename(f.filename)) != "":
                upload_path = "dataset/original" + str(secure_filename(f.filename))
                f.save(upload_path)
                dataset_name = str(secure_filename(f.filename))
            else:
                dataset_name = request.form.get("dataset_name", None) + ".csv"
        if target and dataset_name:
            df = pd.read_csv("dataset/original/" + str(dataset_name))
            print(df)
            y = df[target]
            X = df.drop(target, 1)
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
            cls = RandomForestClassifier(max_depth=7, random_state=0)
            cls.fit(X_train, y_train)
            # compute SHAP values
            explainer = shap.TreeExplainer(cls)
            shap_values = explainer.shap_values(X)
            class_names = list(set(list(y)))
            print(class_names)
            script_dir = os.path.dirname(__file__)
            results_dir = os.path.join(script_dir, "static/")
            con = plot_confusion_matrix(cls, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues, xticks_rotation='vertical')
            conmatrix = "confusion_matrix.png"
            plt.savefig(results_dir + conmatrix)
            plt.clf()
            fig_fi = shap.summary_plot(shap_values, X.values, plot_type="bar", class_names=class_names, feature_names=X.columns, show=False)
            featurebar = "feature_bar.png"
            plt.savefig(results_dir + featurebar)



    return render_template('feabar.html', featurebar = featurebar, conmatrix = conmatrix)
# heatmap
@application.route('/heat', methods=['POST','GET'])
def heat():
        if request.method == "POST":
            if request.files:
                f = request.files['dataset']
                if str(secure_filename(f.filename)) != "":
                    upload_path = "dataset/original/" + str(secure_filename(f.filename))
                    f.save(upload_path)
                    dataset_name = str(secure_filename(f.filename))
                else:
                    dataset_name = request.form.get("dataset_name", None)
                    if ".csv" in dataset_name:
                        dataset_name = dataset_name
                    else:
                        dataset_name = request.form.get("dataset_name", None) + ".csv"
            if dataset_name:
                df = pd.read_csv("dataset/original/" + str(dataset_name))
                title = "Dataset Name:" + str(dataset_name)
                #print(df.corr())
                heatmap_data = df.corr()
                fig1 = px.imshow(heatmap_data, text_auto=True, color_continuous_scale='RdBu_r')
                graph1JSON = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)


        #os.remove("dataset/retrieve/" + str(dataset_name))
        return render_template('heatmap.html', graph1JSON=graph1JSON,dataset_name = dataset_name)


@application.route('/upload_dataset', methods =['POST','GET'])
def upload_dataset():
    file_dir = "dataset/"
    files = os.listdir(file_dir)
    data, data_path, head = None, None, None
    if request.method == "POST":
        if request.files:
            f = request.files['dataset']
            if str(secure_filename(f.filename)) != "":
                data_path = 'dataset/' + secure_filename(f.filename)
                f.save(data_path)
                print('file uploaded successfully')
                dataset_name= f.filename
                data = read_file(data_path)
            if not data_path:
                flash("please upload a dataset in suitable format('.csv .xlsx .txt')")
                return render_template('upload_dataset.html',data=None, files=files)
            else:
                file_size = os.path.getsize(data_path)
                if file_size >= 5000000:
                    flash(f"the maximum file size limit is 500M! Your file is {file_size/1000000}M! Please upload another dataset")
                    return render_template('upload_dataset.html', data=None, files=files)
        else:
            print('existing dataset')
            dataset_name = request.form.get("dataset_name", None)
            data_path = 'dataset/' + str(dataset_name)
            data = read_file(data_path)
            print('show!!!')
            if not data_path:
                flash("the dataset you chose is not in a suitable format('.csv .xlsx .txt')")
                return render_template('upload_dataset.html',data=None, files=files)
            else:
                file_size = os.path.getsize(data_path)
                if file_size >= 300000000:
                    flash(
                        f"the maximum file size limit is 500M! Your file is {file_size / 1000000}M! Please upload another dataset")
                    return render_template('upload_dataset.html', data=None, files=files)
        return render_template('upload_dataset.html', data=data.head(100), heads=data.columns, data_path = data_path, files = files)

    return render_template('upload_dataset.html', data=None, files=files)

@application.route('/expand_data/', methods=['POST','GET'])
def expand_data():
    if request.method == "POST":
        data_path = request.form.get('data_path_1', None)
        check_columns = request.form.getlist('checkbox', None)
        data_path2 = request.form.get('data_path_2', None)
        print('1',data_path)
        if data_path:
            data = read_file(data_path)
            candidate_columns = find_columns_to_be_expanded(data)
            print(candidate_columns)
            if not candidate_columns:
                return render_template('retrieve_columns.html')
            else:
                return render_template('expand_data.html',data=data, data_path = data_path, candidate_columns = candidate_columns,new_data=None)

        if check_columns:
            old_df = read_file(data_path2)
            heads = old_df.columns
            new_df, sub_parameters = expand_parameters_stored_in_one_column(data_path2, check_columns)

            data_path_expand = data_path2[:-4] +'_expand'+data_path2[-4:]

            new_df.to_csv(data_path_expand,index=False)
            print(new_df.head())
            return render_template('expand_data.html', new_data=new_df, data_path = data_path_expand, heads = heads,sub_parameters=sub_parameters)




    print('aaaaaa')
    return render_template('retrieve_columns.html')

@application.route('/show_table/<path:data_path>/',methods=['POST','GET'])
def show_table(data_path):
    data = read_file(data_path)
    print(data.head())

@application.route('/retrieve_columns/<path:data_path>/', methods = ['POST','GET'])
def retrieve_columns(data_path):
    data = read_file(data_path)
    print(data.head())
    diversity_columns, continuous_columns = find_available_filters(data)
    diversity_filters, continuous_filters = {}, defaultdict(list)
    retrieved = False
    if request.method == 'POST':

        #diversity_filters = request.form.getlist('diversity_filters')

        for column in diversity_columns:
            if request.form.get(f"diversity_filters_{column}"):
                retrieved = True
                diversity_filters[column] = request.form.getlist(f"diversity_filters_{column}", None)
                if 'any' in diversity_filters[column]:
                    del diversity_filters[column]
        for column in continuous_columns:
            if request.form.get(f"continuous_filters_minimum_{column}"):
                retrieved = True
                continuous_filters[column].append(float(request.form.get(f"continuous_filters_minimum_{column}")))
                continuous_filters[column].append(float(request.form.get(f"continuous_filters_maximum_{column}")))
        print(diversity_filters,'diversity_filters')
        print(continuous_filters,'continuous_filters')
        return_columns = request.form.getlist('check_return_columns', None)  # return columns
        print('all',return_columns)
        if return_columns: retrieved = True
        if retrieved:
            output_path = data_path[:-4] +'_retrieve'+data_path[-4:]
            unique = False
            expand = False
            new_df = retrieve_target_columns_based_on_values(data_path, requirements=diversity_filters, range_requirements=continuous_filters,
                                                    target_columns=return_columns,output_path=output_path, unique=unique, expand=expand)

            print('retrieve succesfully!')
            return render_template('retrieve_columns.html', data_path = output_path, data = new_df, df_length=len(new_df), all_columns = new_df.columns,
                                        diversity_filters=diversity_filters, continuous_filters=continuous_filters,new_df=new_df)

    all_columns = list(data.columns)
    return render_template('retrieve_columns.html',data_path = data_path, data = data,df_length=len(data),
                           all_columns = data.columns, i = 0,
                           diversity_columns = diversity_columns, continuous_columns=continuous_columns, new_df = None)

# def retrieve_columns():
#     data = None
#     target_columns = []
#     download_path = ""
#     if request.method == "POST":
#         if request.files:
#             f = request.files['dataset']
#             f.save('dataset/' + secure_filename(f.filename))
#             print('file uploaded successfully')
#             dataset_name = f.filename
#             data = read_file('dataset/' + dataset_name)
#             download_path='dataset/' +dataset_name
#             target_columns = data.columns
#         else:
#             dataset_name = request.form.get("datasets", None)
#             print(dataset_name)
#             if dataset_name:
#                 data = read_file('dataset/'+dataset_name)
#                 print(data.head())
#                 data, target_columns_2 = expand_parameters_stored_in_one_column('dataset/'+dataset_name, data.columns)
#                 download_path = '././dataset/' + dataset_name + "_new"
#                 print('show!!!')
#
#         return render_template('retrieve_columns.html', download_path = download_path, dataset_name = dataset_name, data = data.head(50), target_columns = data.columns)
#     return render_template('retrieve_columns.html', data=data, target_columns = target_columns)

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