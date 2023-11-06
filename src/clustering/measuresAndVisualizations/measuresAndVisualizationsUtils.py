import csv
import os.path as path  
import os

def saveDictToFile (data:dict, filePath, custom_headers=["", "value"]):
    # Open the CSV file for writing
    with open(filePath, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the custom headers to the CSV file
        writer.writerow(custom_headers)

        # Write the dictionary data to the CSV file
        for key, value in data.items():
            writer.writerow([key, value])

def getMetricsFolderPath ():
    return path.join (path.dirname(__file__), "../metrics")

def getPlotsFolderPath ():
    return path.join (path.dirname(__file__), "../plots")

def checkSubFoldersExists (path):
    os.makedirs(path, exist_ok=True)