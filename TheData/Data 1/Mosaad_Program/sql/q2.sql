use ltlf_data;
SELECT experiments.Model_ID , experiments.RMSE , experiments.MAPE , experiments.n_features , experiments.n_days  FROM ltlf_data.experiments , 
(SELECT Model_ID, min(MAPE) as MAPE FROM ltlf_data.experiments group by Model_ID ) as ff  where experiments.Model_ID =ff.model_ID and experiments.MAPE =ff.MAPE  ;