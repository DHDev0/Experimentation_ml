import sys 
import numpy as np
import pandas as pd
from pickle import dump
from pickle import load
#transform
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
#model
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#metric
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score


class module_lr:
    def __init__(self, data_path=''):
        self.data = data_path
        if len(data_path) > 0:
            self.data = pd.read_csv(data_path).dropna()
        #add ML module here
        self.parameters_lr ={"MLPClassifier" : [MLPClassifier(max_iter=100,shuffle= True, early_stopping = True,),
                                        {'activation': ['relu', 'tanh', 'logistic', 'identity'],
                                        'hidden_layer_sizes': [(100,), (50,100,), (50,75,100,)],
                                        'solver': ['adam', 'sgd', 'lbfgs'],
                                        'learning_rate' : ['constant', 'adaptive', 'invscaling']}],
                        "MLPClassifier_0" : [MLPClassifier(max_iter=100,shuffle= True, early_stopping = True,),
                                            {}],
                        "LogisticRegression" : [LogisticRegression(),
                                               {'penalty':['l1','l2'],
                                                'C': [0.001, 0.01, 0.1, 1]}],
                        "LogisticRegression_0" : [LogisticRegression(),
                                                 {}],
                        }
        
        
    def pandas_data_bind(self, column_to_predict=["ordered"]):
        data_nun = self.data.nunique().tolist()
        data_shape = self.data.shape[0]
        data_column = list(self.data.columns)
        data_typer = dict(self.data.dtypes)

        training_column = [ data_column[i] for i in range(len(data_nun)) if data_nun[i] < data_shape*0.9 and data_nun[i] > 1 and i != len(data_nun)-1 ]
        try:
            predicted_column = column_to_predict
        except:
            predicted_column = []
            
        return data_typer,training_column,predicted_column


    def preprocessing(self, class_list,overall_type):
        overall_preprocess=[]
        num_cat = [ i for i in class_list if np.issubdtype(overall_type[i], float)]
        if len(num_cat) >= 1:
            numeric_features = num_cat
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
            )
            overall_preprocess.append(("scaler", numeric_transformer, numeric_features))
            
        str_cat = [ i for i in class_list if np.issubdtype(overall_type[i], float) == False]
        if len(str_cat) >= 1:
            categorical_features = str_cat
            categorical_transformer = OneHotEncoder(handle_unknown="ignore")
            overall_preprocess.append(("category", categorical_transformer, categorical_features))
        return ColumnTransformer( transformers=overall_preprocess )


    def inverse_preprocessing(self, preprocessed_transform,preprocessed_data):
        for i in range(len(preprocessed_transform.transformers_)):
            preprocessed_data = preprocessed_transform.named_transformers_[preprocessed_transform.transformers_[i][0]].inverse_transform(preprocessed_data)
        return preprocessed_data


    def model(self,x_train,y_train,model_choice="MLPClassifier"):
        models = GridSearchCV(estimator = self.parameters_lr[model_choice][0],
                            param_grid = self.parameters_lr[model_choice][1],
                            scoring = 'accuracy',
                            cv = 2,
                            n_jobs = -1,
                            verbose=3)
        models.fit(x_train, y_train)
        best_accuracy_lr = models.best_score_
        best_paramaeter_lr = models.best_params_
        return models ,  best_accuracy_lr, best_paramaeter_lr


    def metric(self):#TODO generalize for ml classification and regression
        prediction = self.train_model.predict(self.x_test)
        true = self.inverse_preprocessing(self.predicted_class, self.y_test)
        pred = np.array(self.inverse_preprocessing(self.predicted_class, prediction), dtype=true.dtype)
        print(true.dtype,pred.dtype)
        c_matrix = confusion_matrix(true, pred)
        # classes = np.unique(true).size
        # c_matrix = np.bincount(true * classes + pred).reshape((classes, classes))
        # acc_metric=accuracy_score(y_test, prediction)
        # roc_metric=roc_auc_score(y_test, prediction)
        f_p = c_matrix[0][1]/(c_matrix[0][1]+c_matrix[1][1])
        f_n = c_matrix[1][0]/(c_matrix[0][0]+c_matrix[1][0])
        stdoutOrigin = sys.stdout 
        sys.stdout = open("report.txt", "w")
        print("########### MODEL CHOICE###########")
        print(f"Best Accuracy of LR: {self.best_model_accuracy.mean()*100:.2f} %")
        print(f"Best Parameter of LR: {self.best_model_parameter}")
        print("########### MODEL METRIC ###########")
        print(f"confusion_matrix :")
        print(c_matrix)
        print(f"Pourcentage of false positive: {(f_p)*100:.3f} %")
        print(f"Pourcentage of false negative: {(f_n)*100:.3f} %")
        print(f"Positive accuracy: {(1-f_p)*100:.3f} %")
        print(f"Negative accuracy: {(1-f_n)*100:.3f} %")
        sys.stdout.close()
        print("create report.txt")


    def experiment(self,column_to_predict=["ordered"],model_choice="MLPClassifier_0"):
        #find csv structure                           pandas_data_bind
        data_typer,training_column,predicted_column = self.pandas_data_bind(column_to_predict=column_to_predict)
        #split training point and target
        x_var = self.data[training_column]
        y_var = self.data[predicted_column]
        #transform
        self.predictor_class = self.preprocessing(training_column,data_typer)
        self.predicted_class = self.preprocessing(predicted_column,data_typer)
        #split train dataset and test dataset
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.predictor_class.fit_transform(x_var),self.predicted_class.fit_transform(y_var),
                                                                             test_size=0.20,random_state=42)
        #train model
        self.train_model ,  self.best_model_accuracy, self.best_model_parameter = self.model(self.x_train,self.y_train,model_choice=model_choice)
        #measure performance
        self.metric()
        # save the model
        dump(self.train_model, open('model.pkl', 'wb'))
        # save the scaler
        dump(self.predictor_class, open('scaler_predictor.pkl', 'wb'))
        dump(self.predicted_class, open('scaler_predicted.pkl', 'wb'))
        

    def predictor(self,class_to_predict=["ordered",], csv_result_path="result.csv",prob_stat=False):
        #load model and scaler
        data_typer,data_train,data_pred = self.pandas_data_bind(column_to_predict=column_to_predict)
        x_var = self.predictor_class.transform(self.data[data_train])
        y_var = self.predicted_class.transform(self.data[data_pred])
        #predict
        dataset = self.data[data_train]
        pred = self.inverse_preprocessing(self.predicted_class,self.train_model.predict(self.y_test))
        print(pred.shape)
        dataset[class_to_predict[0]]=self.inverse_preprocessing(self.predicted_class,self.train_model.predict(self.y_test))
        #predict probability of class
        if prob_stat:
            prob = [[[self.inverse_preprocessing(self.predicted_class,np.array([self.train_model.classes_[i.tolist().index(h)]])),round(h*100,5)] for h in i] 
                    for i in self.train_model.predict_proba(test)]
            for i in model.classes_.tolist():
                dataset[class_to_predict[0] + f"_predict_classe_{i}_with_a_probability_of"] = [h[i][1] for h in prob]
        #save data
        dataset.data.to_csv("result.csv", index=False)


    def predictor_isolate(data_path ="training_sample.csv", model_path='model.pkl'
                ,scaler_predictor_path='scaler_predictor.pkl', scaler_predicted='scaler_predicted.pkl',
                class_to_predict=["ordered",], csv_result_path="result.csv",
                prob_stat=False):
        #load data
        data_pred = pd.read_csv(data_path).dropna()
        #load model and sclaer
        models = load(open(model_path, 'rb'))
        scaler_predictor = load(open(scaler_path, 'rb'))
        scaler_predicted = load(open(scaler_path, 'rb'))
        data_typer,data_train,_ = pandas_data_bind(data,column_to_predict=column_to_predict)
        #transform
        try:
            data = scaler_predictor.transform(data_pred.drop(class_to_predict, axis=1)[data_train])
        except: 
            data = scaler_predictor.transform(data_pred[[data_train]])
        #predict
        data_pred[class_to_predict[0]]=self.inverse_preprocessing(scaler_predicted,models.predict(data))
        #predict probability of class
        if prob_stat:
            prob = [[[self.inverse_preprocessing(scaler_predicted,np.array([model.classes_[i.tolist().index(h)]])),round(h*100,5)] for h in i] for i in models.predict_proba(test)]
            for i in model.classes_.tolist():
                data_pred[class_to_predict[0] + f"_predict_classe_{i}_with_a_probability_of"] = [h[i][1] for h in prob]
        #save data
        data_pred.to_csv("result.csv", index=False)



