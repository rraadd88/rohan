import pandas as pd
def sheetname2df_to_excel(sheetname2df,datap,):
    writer = pd.ExcelWriter(datap)
    for sn in sheetname2df:
        sheetname2df[sn].to_excel(writer,sn)
    writer.save()
