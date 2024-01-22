import pandas as pd
import openpyxl
import matplotlib.pyplot as plt



ppe_list = {
            0: 'Hardhat', 
            1: 'NO-Hardhat',
            2: 'Mask', 
            3: 'NO-Mask', 
            4: 'Safety Vest',
            5: 'NO-Safety Vest'
            }


def write_excel(cur_ppe_person: list):
    #loading the table
    path = "PPE_accounting.xlsx" 
    wb_obj = openpyxl.load_workbook(path)
    sheet = wb_obj.active

    prev_ppe_person = [sheet.cell(row = 2, column = col).value for col in range(2,8)] #we read from the table amount of each equipment

    for item in cur_ppe_person: #PPE per current person
        id = [key for key, obj in ppe_list.items() if obj == item][0] #find the ID of the current object in the list
        prev_ppe_person[id] +=1 #increase its amount


    #write new information into the table
    hardhat, no_hardhat, mask, no_mask, safety_vest, no_safety_vest = prev_ppe_person
    df = pd.DataFrame([[hardhat, no_hardhat, mask, no_mask, safety_vest, no_safety_vest]], 
                    columns=['Hardhat', 'NO_Hardhat', 'Mask', 'NO_Mask',  'Safety_Vest','NO_Safety_Vest'])

    df.to_excel('PPE_accounting.xlsx')


    #read the table and make a graph
    data = pd.read_excel(path)
    names = list(data.columns)[1:]
    
    x = ['Hardhat', 'NO_Hardhat', 'Mask', 'NO_Mask', 'Vest','NO_Vest']
    y = [int(data[cl].values[0]) for cl in names]

    plt.bar(x,y, color = ['#00FF00', '#FF6347', '#008000', '#800000', '#ADFF2F', '#F08080'])
    plt.savefig("res.png")

    
    