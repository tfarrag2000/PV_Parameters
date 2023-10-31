-- Create the 'pv_forecasting' database
CREATE DATABASE pv_forecasting;

-- Use the 'pv_forecasting' database
USE pv_forecasting;

-- Create the 'experiments' table to store experiment details
-- Create the 'experiments' table to store experiment details
CREATE TABLE experiments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_ID VARCHAR(255) NOT NULL,
    n_epochs INT NOT NULL,
    n_batch INT NOT NULL,
    outputIndex INT NOT NULL,
    ANN_arch VARCHAR(255) NOT NULL,
    Dropout FLOAT NOT NULL,
    earlystop BOOLEAN NOT NULL,
    RMSE FLOAT NOT NULL,
    MAPE FLOAT NOT NULL,
    min_train_loss FLOAT NOT NULL,
    min_val_loss FLOAT NOT NULL,
    Model_summary TEXT,
    comment TEXT,
    optimizer VARCHAR(255) NOT NULL,
    ActivationFunctions VARCHAR(255) NOT NULL,
    INDEX (experiment_ID)  -- Add an index to the 'experiment_ID' column
);

-- Create the 'experiment_results' table to store experiment results
CREATE TABLE experiment_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    experiment_ID VARCHAR(255) NOT NULL,
    test_RMSE FLOAT NOT NULL,
    test_MAPE FLOAT NOT NULL,
    median_ABE FLOAT,
    FOREIGN KEY (experiment_ID) REFERENCES experiments (experiment_ID)
);

-- Commit changes
COMMIT;

