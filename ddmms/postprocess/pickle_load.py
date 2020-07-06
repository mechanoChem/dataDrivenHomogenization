import pickle
import os, sys

filename = sys.argv[1]

Si = pickle.load(open(filename, "rb"))
print('Si: ', Si)
