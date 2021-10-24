# Run instruction:
## Note: 
- As the experiments were time consuming I have written a map-reduce like script to submit the code to the cluster (I ran on chpc) 
- I used pandas, numpy, and pickle libraries, make sure you have them installed (pip3 install --user pandas)
- for convenience .ipynp notebooks also provided

## For Part2: question 2.b Bagged trees:
- The main file is Bagged_tree_generator.py
- Run **"python3 ./Bagged_tree_generator.py"** 
- Bagged_tree.ipynp shows the notebook and a generated tree
- this will generate one single fully developed decision tree
- for all the 500 independent trees to be generated you need to run **./bagged_map.sh**
- After the cluster jobs completed, run "python3 bagged_reduce.py bagged_trees_predictions/ > bagged_clean.csv" to reduce the results into .csv format

- commands for running on the cluster:
    - ./bagged_map.sh  # start the jobs in cluster
    - python3 bagged_reduce.py bagged_trees_predictions/ > bagged_clean.csv

