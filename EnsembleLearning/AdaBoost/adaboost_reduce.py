# python3 adaboost_reduce.py job_log/ > adaboost_clean.csv
import sys,os

print("iterations, train_accuracy, test_accuracy")
for root, dirs, files in os.walk(sys.argv[1], topdown=False):
	if files:	
		for filename in files:
			if ".out" not in filename: 
				continue
			file_path = os.path.join(root, filename)
			fl = open(file_path).read()
			if "Failure" in fl: 
				continue
			for l in fl.split("\n"):
				l = l.strip()
				if l.startswith("adaboost iters:"):
					tuples = l.split()
					print(tuples[2], ",", tuples[4], ",", tuples[6])