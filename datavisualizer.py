import pandas as pd 
import matplotlib.pyplot as plts

# class for visualizing data given a csv file
class DataVisualizer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
    
    def plot(self, x, y):
        self.data.plot(x=x, y=y)
        plts.show()