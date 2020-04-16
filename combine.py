import pandas as pd
import os, glob

def excel_to_csv():
    for i in range(1, 21):
        r1 = pd.read_excel('./data/review/spread/reviews_labeled-'+ str(i)+'.xlsx')
        r1.to_csv('./data/review/spread/csv/reviews_labeled_' + str(i) +'.csv', encoding='utf-8', index=False)

def combine():
    fout=open("./data/review/review_merged.csv","a")
    # first file:
    for line in open("./data/review/spread/csv/reviews_labeled_1.csv"):
        fout.write(line)
    # now the rest:    
    for num in range(2, 21):
        f = open("./data/review/spread/csv/reviews_labeled_"+str(num)+".csv")
        next(f) # skip the header
        for line in f:
            fout.write(line)
        f.close() # not really needed
    fout.close()

# excel_to_csv()

# Run once or delete old file (it will append the data to the old file) 
# combine() 

df = pd.read_csv('./data/review/review_merged.csv')
df.drop(['Unnamed: 0'], axis=1, inplace=True)
df.to_csv('./data/review/review_merged.csv')
print(df.shape)

