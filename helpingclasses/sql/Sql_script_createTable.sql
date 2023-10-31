SELECT  Model_ID , min(MAPE) ,  max(MAPE), count(*) FROM ltlf_data.experiments group by Model_ID order by Model_ID

CREATE TABLE `experiments` (
  `experiment_ID` varchar(14) NOT NULL,
  `n_days` int(11) DEFAULT NULL,
  `n_features` int(11) DEFAULT NULL,
  `n_traindays` int(11) DEFAULT NULL,
  `n_epochs` int(11) DEFAULT NULL,
  `n_batch` int(11) DEFAULT NULL,
  `n_neurons` int(11) DEFAULT NULL,
  `stacked_layers_num` int(11) DEFAULT NULL,
  `Dense_neurons_n2` int(11) DEFAULT NULL,
  `Dense_neurons_n1` int(11) DEFAULT NULL,
  `Dropout` double DEFAULT NULL,
  `earlystop` int(11) DEFAULT NULL,
  `RMSE` double DEFAULT NULL,
  `MAPE` double DEFAULT NULL,
  `min_train_loss` double DEFAULT NULL,
  `min_val_loss` double DEFAULT NULL,
  `optimizer` varchar(10) DEFAULT 'Adam',
  `Model_ID` int(11) DEFAULT NULL,
  `comment` text,
  `Model_summary` longtext,
  PRIMARY KEY (`experiment_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='experiment_ID, n_days, n_features, n_traindays, n_epochs, n_batch, \n        n_neurons,earlystop, RMSE, MAPE, min_train_loss, min_val_loss, Model_summary\n\nID like ''201811122127''';
