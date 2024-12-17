# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:34:46 2024

@author: 7muha
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

def PlotGraph(x_feature_1, label_1, x_feature_2,label_2, Y):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black', 'pink', 'gray', 'cyan']
    cmap = ListedColormap(colors)
    plt.figure(figsize=(10, 6))
    #scatter = plt.scatter(x_feature_1, x_feature_2, c=Y[:, 0], s=Y[:, 0] * 100, cmap='viridis', alpha=0.7)
    # Add color bar to represent the Y values
    scatter = plt.scatter(x_feature_1, x_feature_2, c=Y,  s=Y[:, 0] * 100, cmap=cmap, edgecolor='k')
    plt.colorbar(scatter, label='Y Value')
    plt.xlabel(label_1)
    plt.ylabel(label_2)
    plt.title('X Feature 1 vs X Feature 2 with Y Values')
    
    # Show the plot
    plt.show()

def Plot_2_Graph(x_feature_1, label_1, x_feature_2, label_2,y):
    
    # Define 9 distinct colors for the classes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan']
    
    cmap = cm.get_cmap('Set1', 9)
    
    scatter = plt.scatter(x_feature_1, x_feature_2, c=y, cmap=cmap)
    # Create a scatter plot
    #plt.figure(figsize=(10, 6))
    #for i in range(9):
        # Filter X and Y based on class i and plot them
    #    plt.scatter(x_feature_1, x_feature_2, color=colors[i], label=f'Class {i}', edgecolor='k')
    
    
    plt.colorbar(scatter, ticks=range(9), label='Y class')
    
    # Label the axes
    plt.xlabel(label_1)
    plt.ylabel(label_2)
    plt.title('X Feature 1 vs X Feature 2 with Discrete Colors')

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()          