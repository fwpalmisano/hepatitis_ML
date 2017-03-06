# hepatitis_ML
First project utilizing Decision Tree and Random Forest methods! Applied to dataset on hepatitis from the UCI Machine Learning Repository.
___________________

For this project, I chose to work with a dataset from the UCI machine learning repository. This dataset contained information on patients with hepatitis. I chose this dataset because I was recently diagnosed with a rare genetic mutation that affects
my liver and initially presented with hepatitis-like symptoms. Most of the attributes were either Boolean classifiers 
or floating point values. The attributes contained information on symptoms displayed by patients (fatigue, malaise, liver 
palpable, spiders, etc), treatments pursued (steroids, antivirals, etc.), levels of different blood markers (bilirubin, 
albumin, etc), and survival. 

For my first experiment I used decision trees and random forest algorithms to predict survival based on 
symptoms displayed and treatments pursued. I was able to predict survival with 85%+ accuracy with DT and 90%+ with
Random Forests, but I believe that the accuracy of these models are overstated. In this dataset the VAST majority 
of individuals died (an algorithm could just say that every single person in my test group died and probably get 
80%+ accuracy). I would be more impressed with the accuracy of the models if the datasets were more balanced 
in terms of survival. 

For my second experiment I decided to use DT and RF Regressions to predict levels of albumin (protein in blood plasma) 
in the blood of individuals). For whatever reason, it was much easier to predict albumin levels than other 
floating point levels like bilirubin. I was able to predict albumin levels with a 50%+ accuracy with DT and 
70%+ accuracy with RF. 
