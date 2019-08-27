# TCS-HumAIn-2019 (Taxonomy Creation)
To predict the tags (a.k.a. keywords, topics, summaries), given only the
question text and its title.

Problem Statement: - For the given content, come up 
with a solution to build the taxonomy. 

**High Level Solution Approach**


![Flow Diagram](https://user-images.githubusercontent.com/40590709/63788791-b46b4000-c913-11e9-81e0-a15f3b101667.jpg)

1. Do data preprocessing on raw data to make data suitable for the Machine
Learning model.
2. Train the model using data present in Train.csv
3. Once the model training is completed, apply the same model on data present
in Test.csv
4. The model will predict tags for test data.
5. Once tags are predicted, create a new CSV file with Id and Tags column.

