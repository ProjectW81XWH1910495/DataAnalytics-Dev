from datetime import datetime

import pandas as pd

# make sure your current path is \ProjectW81XWH1910495-keras then use python utils/retrievecolumns.py to run it
def retrieve_columns(originaltablePath,parameterList,outputPath,ifDeleteNull=False):
    """
this function is designed to retrieve a series important parameters from the original result table and merge into a new
table to make further data analysis
    @param originaltablePath:the path of the original result table as the input
    @param parameterList: the parameter you need to retrieve like ["ml_classifier_name","parameters_and_values","mean_test_auc","mean_train_auc","percent_auc_diff"],
    if the parameterList include parameter_and_values, it will separate to different columns respectively.
    @param outputPath:the path that you want to store this new retrieve table
    """
    originalTable = pd.read_csv(originaltablePath,low_memory=False)
    if 'parameters_and_values' in parameterList:
        parameter_list = originalTable['parameters_and_values'].tolist()
    parameter_label = list(eval(parameter_list[0]).keys())
    parameter_df = pd.DataFrame(columns=parameter_label)
    for item in parameter_list:
        item_dict = eval(item)
        if 'mstruct' in item_dict:
            layer = list(item_dict['mstruct'])[0:-1]
            layer_number = len(layer)
            layer.extend([0, 0, 0, 0])
            for i in range(4):
                item_dict["layer_number"] = layer_number
                item_dict[i+1] = layer[i]
        parameter_df=parameter_df.append(item_dict,ignore_index=True)

    parameterList.remove("parameters_and_values")
    rest_df = pd.DataFrame(originalTable, columns=parameterList)
    newTable = pd.concat([rest_df, parameter_df],axis=1)
    if ifDeleteNull:
        newTable.dropna(axis=0, how='any', inplace=True)
    newTable.to_csv(outputPath,index=False)
    print(outputPath + " generated successfully")


if __name__ == "__main__":
    #tablelist = {"20000.csv": "20000_retrieve.csv"}
    #tablelist = {"DNMStage1merged15Year.csv":"any.csv"}
    #tablelist = {"new_sk_model.csv":"anyML.csv"}
    tablelist = {

                 "15I240_h4_merged_s1.csv": "15I240_h4_merged_s1_retrieve.csv",
                 "15I240_h4_merged_s2.csv": "15I240_h4_merged_s2_retrieve.csv",
                 "15I240_h4_merged_s3.csv": "15I240_h4_merged_s3_retrieve.csv",
                 "15I240_h4_merged_s4.csv": "15I240_h4_merged_s4_retrieve.csv",
                 }
    # tablelist = {"5I240_h4_merged_s1.csv": "5I240_h4_merged_s1_retrieve.csv",
    #              "5I240_h4_merged_s2.csv": "5I240_h4_merged_s2_retrieve.csv",
    #              "5I240_h4_merged_s3.csv": "5I240_h4_merged_s3_retrieve.csv",
    #              "5I240_h4_merged_s4.csv": "5I240_h4_merged_s4_retrieve.csv",
    #              "10I240_h4_merged_s1.csv": "10I240_h4_merged_s1_retrieve.csv",
    #              "10I240_h4_merged_s2.csv": "10I240_h4_merged_s2_retrieve.csv",
    #              "10I240_h4_merged_s3.csv": "10I240_h4_merged_s3_retrieve.csv",
    #              "10I240_h4_merged_s4.csv": "10I240_h4_merged_s4_retrieve.csv",
    #              "15I240_h4_merged_s1.csv": "15I240_h4_merged_s1_retrieve.csv",
    #              "15I240_h4_merged_s2.csv": "15I240_h4_merged_s2_retrieve.csv",
    #              "15I240_h4_merged_s3.csv": "15I240_h4_merged_s3_retrieve.csv",
    #              "15I240_h4_merged_s4.csv": "15I240_h4_merged_s4_retrieve.csv",
    #              }
    #tablelist = {"year_15.csv":"year_15_retrieve.csv"}
    # tablelist = {}
    # for i in range(1,140):
    #     key = "h4_total-" + str((i-1)*1000) + "-" + str(i*1000) + "_1.csv"
    #     value = "h4_retrieve_total-" + str((i - 1) * 1000) + "-" + str(i * 1000) + "_1.csv"
    #     tablelist[key] = value
    for item in tablelist.keys():
        # retrieve_columns(originaltablePath = "DNM-RF/stage1/results/240/LSM-15Year-I-240_results/" + item,
        #                   parameterList = ["ml_classifier_name","parameters_and_values","mean_test_auc","mean_train_auc","percent_auc_diff"],
        #                  outputPath = "DNM-RF/stage1/retrieve/240/LSM-15Year-I-240_retrieve/" + tablelist[item],
        #                  ifDeleteNull=False)
        # retrieve_columns(originaltablePath="DNM-RF/stage1/mergedresult/240/" + item,
        #                  parameterList=["ml_classifier_name","host_name","running_time1(average sec)","parameters_and_values", "mean_test_auc",
        #                                 "mean_train_auc", "percent_auc_diff"],
        #                  outputPath="DNM-RF/stage1/mergedresult/240/" + tablelist[item],
        #                  ifDeleteNull=False)
        #Jiang test below (2021.8.9)
        # retrieve_columns(originaltablePath="DNM/stage1/results/" + item,
        #          parameterList=["ml_classifier_name","host_name","running_time1(average sec)","parameters_and_values", "mean_test_auc",
        #                         "mean_train_auc", "percent_auc_diff"],
        #          outputPath="testing/" + tablelist[item],
        #          ifDeleteNull=False)
        # retrieve_columns(originaltablePath="DNM/stage1/results/" + item,
        #          parameterList=["ml_classifier_name","host_name","running_time1(average sec)","parameters_and_values", "mean_test_auc",
        #                         "mean_train_auc", "percent_auc_diff"],
        #          outputPath="testing/" + tablelist[item],
        #          ifDeleteNull=False)
        retrieve_columns(originaltablePath="dataset/" + item,
                 parameterList=["ml_classifier_name","host_name","running_time1(average sec)","parameters_and_values", "mean_test_auc",
                                "mean_train_auc", "percent_auc_diff"],
                 outputPath="dataset/retrieve/" + tablelist[item],
                 ifDeleteNull=False)