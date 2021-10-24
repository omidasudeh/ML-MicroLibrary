# Run instruction:
## Note: 
- As the experiments were time consuming I have written a map-reduce like script to submit the code to the cluster (I ran on chpc) 
- I used pandas, numpy, and pickle libraries, make sure you have them installed (pip3 install --user pandas)
- for convenience .ipynp notebooks also provided

## For Part2: question 2.a Adaboost: 

- The main file is adaboost.py
- Run **"python3 ./adaboost.py num_iterations"** 
- e.g. python3 ./ adaboost.py 5 trains adaboost model with 5 iterations
- **__adaboost.ipynp__** is the notebook showing the results for 20 adaboost iterations
- outputs contains the train and test accuracies together with the list of the root nodes used in the decisions stumps and also the corresponding alpha
- if you Run **./adaboost_map.sh** it spawns 50 jobs on the CHPC cluster and each job does 10 iterations in a cyclic manner.
- make sure to change the account and partition in adaboost.sh based on your allocation on CHPC.
- After the cluster jobs completed, run "python3 adaboost_reduce.py job_log/ > adaboost_clean.csv" to reduce the results into .csv format

- commands for running on the cluster:
    - ./adaboost_map.sh  # start the jobs in cluster
    - python3 adaboost_reduce.py job_log/ > adaboost_clean.csv