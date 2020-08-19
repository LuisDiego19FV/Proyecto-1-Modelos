# Original by Luis Diego Fernandez
import math
import numpy as np
import pandas as pd
from optimization import opt_naive

# Function to find the R^2 of a model
def r_square(real, prediction, avg_y):
    sum_ssr = 0
    sum_sst = 0

    for index, val in enumerate(real):
        sum_ssr += (val - prediction[index])**2
        sum_sst += (val - avg_y)**2
    
    return 1 - sum_ssr/sum_sst

# Import original data
file_name = 'data.csv'
df = pd.read_csv(file_name)
global_avg_y = df.mean(axis=0)[0]

# Menu
flag = 0

while flag == 0:
    try:
        flag = input('''
Choose model to train:
    1. Model 1
    2. Model 2
    3. Model 3
    4. Model 4
    5. ALL
    6. Exit\n''')

        flag = int(flag)

    except:
        print("\nChoose a Numbre between 1 - 6")
        flag = 0

# Exit
if flag == 6: 
    exit()

# MODELO 1: q = a + xB, x = p & y = q
if flag == 1 or flag == 5:
    # data model
    df_m1 = df.copy()

    # get beta
    cov = df_m1.cov()
    var = df_m1.var()
    beta = cov["sales (q)"][1]/var["precio (p)"]

    # get alpha
    avg_y = df_m1.mean(axis=0)[0]
    avg_x = df_m1.mean(axis=0)[1]
    alpha = avg_y - beta * avg_x

    # function of the model
    model = lambda x: beta * x + alpha

    # data of the model
    model1_data = [[model(i), i] for i in df_m1["precio (p)"]]

    # out >> csv with data of model 2
    df_m1_pred = pd.DataFrame(data=model1_data, columns=["q", "p"])
    df_m1_pred.to_csv(path_or_buf="out model1.csv")

    # r^2
    r2 = r_square(df_m1["sales (q)"], df_m1_pred["q"], avg_y)

    # elasticity
    edp = beta * (avg_x/avg_y)

    # optimization
    optimal = opt_naive(model)

    print("\nMODEL 1: q ~ p | q = alpha + beta * p")
    print("--- Alpha: " + str(alpha))
    print("--- Beta:  " + str(beta))
    print("--- EDP:   " + str(edp))
    print("--- R^2:   " + str(r2))
    print("--- Opt.P: " + str(optimal))
    print("* data of predictions shown in \'out model1.csv\'")

# MODELO 2: ln(q) = a + xB, x = p & y = ln(q)
if flag == 2 or flag == 5:
    # data model
    df_m2 = df.copy()
    df_m2["sales (q)"] = df_m2["sales (q)"].apply(np.log)

    # get beta
    cov = df_m2.cov()
    var = df_m2.var()
    beta = cov["sales (q)"][1]/var["precio (p)"]

    # get alpha
    avg_y = df_m2.mean(axis=0)[0]
    avg_x = df_m2.mean(axis=0)[1]
    alpha = avg_y - beta * avg_x

    # function of the model
    model = lambda x: beta * x + alpha

    # data of the model
    model2_data = [[math.e**model(i),model(i), i] for i in df_m2["precio (p)"]]

    # out >> csv with data of model 2
    df_m2_pred = pd.DataFrame(data=model2_data, columns=["q" ,"ln(q)", "p"])
    df_m2_pred.to_csv(path_or_buf="out model2.csv")

    # r^2
    r2 = r_square(df_m2["sales (q)"], df_m2_pred["ln(q)"], avg_y)

    # elasticity
    edp = beta * avg_x

    # optimization (taking in account model is using ln(q))
    model_opt = lambda x: math.e**(beta * x + alpha)
    optimal = opt_naive(model_opt)

    print("\nMODEL 2: ln(q) ~ p | ln(q) = alpha + beta * p")
    print("--- Alpha: " + str(alpha))
    print("--- Beta:  " + str(beta))
    print("--- EDP:   " + str(edp))
    print("--- R^2:   " + str(r2))
    print("--- Opt.P: " + str(optimal))
    print("* data of predictions shown in \'out model2.csv\'")

# MODELO 3: y = a + xB, x = ln(p) & y = q
if flag == 3 or flag == 5:
    # data model
    df_m3 = df.copy()
    df_m3["precio (p)"] = df_m3["precio (p)"].apply(np.log)

    # get beta
    cov = df_m3.cov()
    var = df_m3.var()
    beta = cov["sales (q)"][1]/var["precio (p)"]

    # get alpha
    avg_y = df_m3.mean(axis=0)[0]
    avg_x = df_m3.mean(axis=0)[1]
    alpha = avg_y - beta * avg_x

    # function of the model
    model = lambda x: beta * x + alpha

    # data of the model
    model3_data = [[model(i), math.e**i, i] for i in df_m3["precio (p)"]]

    # out >> csv with data of model 2
    df_m3_pred = pd.DataFrame(data=model3_data, columns=["q" ,"p", "ln(p)"])
    df_m3_pred.to_csv(path_or_buf="out model3.csv")

    # r^2
    r2 = r_square(df_m3["sales (q)"], df_m3_pred["q"], avg_y)

    # elasticity
    edp = beta * (1/avg_y)

    # optimization (taking in account model is using ln(p))
    optimal = math.e**opt_naive(model)

    print("\nMODEL 3: q ~ ln(p) | q = alpha + beta * ln(p)")
    print("--- Alpha: " + str(alpha))
    print("--- Beta:  " + str(beta))
    print("--- EDP:   " + str(edp))
    print("--- R^2:   " + str(r2))
    print("--- Opt.P: " + str(optimal))
    print("* data of predictions shown in \'out model3.csv\'")

# MODELO 4: y = a + xB, x = ln(p) & y = ln(q)
if flag == 4 or flag == 5:
    # data model
    df_m4 = df.copy()
    df_m4 = df_m4.apply(np.log)

    # get beta
    cov = df_m4.cov()
    var = df_m4.var()
    beta = cov["sales (q)"][1]/var["precio (p)"]

    # get alpha
    avg_y = df_m4.mean(axis=0)[0]
    avg_x = df_m4.mean(axis=0)[1]
    alpha = avg_y - beta * avg_x

    # function of the model
    model = lambda x: beta * x + alpha

    # data of the model
    model4_data = [[math.e**model(i),model(i), math.e**i, i] for i in df_m4["precio (p)"]]

    # out >> csv with data of model 2
    df_m4_pred = pd.DataFrame(data=model4_data, columns=["q", "ln(q)" ,"p", "ln(p)"])
    df_m4_pred.to_csv(path_or_buf="out model4.csv")

    # r^2
    r2 = r_square(df_m4["sales (q)"], df_m4_pred["ln(q)"], avg_y)

    # elasticity
    edp = beta

    # optimization (taking in account model is using ln(q) & ln(p))
    optimal = math.e**opt_naive(model)

    print("\nMODEL 4: ln(q) ~ ln(p) | ln(q) = alpha + beta * ln(p)")
    print("--- Alpha: " + str(alpha))
    print("--- Beta:  " + str(beta))
    print("--- EDP:   " + str(edp))
    print("--- R^2:   " + str(r2))
    print("--- Opt.P: " + str(optimal))
    print("* data of predictions shown in \'out model4.csv\'")

