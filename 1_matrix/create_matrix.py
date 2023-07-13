import pandas as pd
import random
import csv
import sys
import os

os.chdir('/home/getalp/mourasa/pipeline/1_matrix')

# Places list shuffle
def shuffle_items(item):
    random.shuffle(item)
    return item

# Reading names from People_spreadsheet.csv
people = []
with open('./datasets/People_spreadsheet.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        people.append(row['first_name'] + ' ' + row['last_name'])

# Places list
places_df=pd.read_csv('./datasets/Places_spreadsheet.csv')
city_names=places_df['Place']
shuffled_places=shuffle_items(city_names.tolist())

# Organizations list
organizations_df=pd.read_csv('./datasets/Organizations_spreadsheet.csv')
organization_column=organizations_df['Organizations']
shuffled_orgs=shuffle_items(organization_column.tolist() + organization_column.tolist())

data = []
# Example 20 rows
for _ in range(int(sys.argv[1])):
    person = random.choice(people)
    place=shuffled_places.pop(0)
    organization=shuffled_orgs.pop(0)
    row = [person, place, organization]
    data.append(row)

# matrix column names
fieldnames = ["PERSON", "PLACE", "ORGANIZATION"]
filename = "../2_prompt/datasets/matrix.csv"

# Output file
with open(filename, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(fieldnames)
    writer.writerows(data)
