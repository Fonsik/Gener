import random
import pickle
import numpy as np
random.seed()


tracks=open("tracks.pkl", "rb")

new=pickle.load(tracks)
print new
tracks.close()

