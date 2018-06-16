HYBRID KEYWORD EXTRACTION - MODEL 3 

MODEL 3, generates the best result as compared to our other models. Hence, it is our proposed model and our research paper is based on this model.

There are 4 models attached in this zip file. The zip file consists of the following:
1. Model 1: (model_1.py) Purely Statistical Keyword extraction , using termfrequency with Force Formula.
2. Model 2: (model_2.py) Preprocessing using NLTK funnctions, calculating Hybrid Score using statistical method.
3. Model 3: (model_3.py) Selected Model : Hybrid Approach, using statistical and graphical methods
4. Model 4: (model_4.py) Purely Graphical Keyword extraction method.

5. Input Folder: Contains 4 input text files from the Dataset. Dataset Link : http://alt.qcri.org/semeval2017/
6. Gold Folder: Contains gold annotated keywords for corresponding input files.
7. output_1 Folder: After model_1.py execution, the extracted keywords are stored in .out files in this folder for model_1
8. output_2 Folder: After model_2.py execution, the extracted keywords are stored in .out files in this folder for model_2
9. output_3 Folder: After model_3.py execution, the extracted keywords are stored in .out files in this folder for model_3
10. output_4 Folder: After model_4.py execution, the extracted keywords are stored in .out files in this folder for model_4
11. eval Folder: After evaluation_v3.py execution, all the results are stored in .csv file in this folder for each model.


NOTE: - Please change the current working directory path in all 4 model and the evaluation script before running the code.
      - For Model_4, please close the graph windown to resume running.

Execution Steps:
1. In python console, please open the current working directory where the models are stored.
2. run: python model_1.py
	python model_2.py
	python model_3.py
	python model_4.py
3. The output .out file will be created and stored in the output folder.
4. run: python evaluation_v3.py
5. The evaluation results are stored in .csv file in eval folder.

Install the follwing packages in the Anaconda Python. Installation can be done via the Anaconda command prompt with the following command

>> pip install 'package-name'

# REQUIRED PACKAGES
# ------------------------------------------------------------
# 1. csv
# 2. glob
# 3. os
# 4. re
# 5. numpy
# 6. sklearn





