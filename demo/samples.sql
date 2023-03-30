-- load sample datasets
SELECT sqml_load_dataset('iris');
SELECT sqml_load_dataset('digits');
SELECT sqml_load_dataset('wine');
SELECT sqml_load_dataset('breast_cancer');
SELECT sqml_load_dataset('diabetes');

-- train some models
SELECT sqml_train('Iris prediction', 'classification', 'logistic_regression', 'dataset_iris', 'target');
SELECT sqml_train('Iris prediction', 'classification', 'svc', 'dataset_iris', 'target');
SELECT sqml_train('Digits prediction', 'classification', 'logistic_regression', 'dataset_digits', 'target');
SELECT sqml_train('Digits prediction', 'classification', 'svc', 'dataset_digits', 'target');
SELECT sqml_train('Wine prediction', 'classification', 'logistic_regression', 'dataset_wine', 'target');
SELECT sqml_train('Wine prediction', 'classification', 'svc', 'dataset_wine', 'target');
SELECT sqml_train('Breast cancer prediction', 'classification', 'logistic_regression', 'dataset_breast_cancer', 'target');
SELECT sqml_train('Breast cancer prediction', 'classification', 'svc', 'dataset_breast_cancer', 'target');
SELECT sqml_train('Diabetes prediction', 'regression', 'linear_regression', 'dataset_diabetes', 'target');
SELECT sqml_train('Diabetes prediction', 'regression', 'svr', 'dataset_diabetes', 'target');
