import pandas as pd
import numpy as np

def normalize_core(core):
    core.columns = [c.replace(' ', '_') for c in core.columns]
    import re
    core["Ingreso_"] = core["Ingreso_"].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core["Ing_Disp"] = core["Ing_Disp"].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core["Ingreso_"] = [x.replace(',', '.') for x in core["Ingreso_"]]
    core["Ing_Disp"]= [x.replace(',', '.') for x in core["Ing_Disp"]]
    core["Ingreso_"] = [x.split(".")[0] for x in core["Ingreso_"]]
    core["Ing_Disp"] = [x.split(".")[0] for x in core["Ing_Disp"]]
    core["Ing_Disp"] = pd.to_numeric(core["Ing_Disp"])
    core["Ingreso_"]  = pd.to_numeric(core["Ingreso_"])
    core["Dependientes"] = core["Dependientes"].apply(lambda x: re.sub("[^\w]", "", str(x)))
    core["Dependientes"] = core["Dependientes"].replace("nan",0)
    core["Dependientes"] = core["Dependientes"].replace("None",0)
    core["Dependientes"] = pd.to_numeric(core["Dependientes"])
    core["Dependientes"]  = core["Dependientes"].fillna(0)
    core["descuentos"] =  core["Ingreso_"] - core["Ing_Disp"]
    core["Monto_de_la_mensualidad"] = core["Monto_de_la_mensualidad"].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core["Monto_de_la_mensualidad"] = [x.replace(',', '.') for x in core["Monto_de_la_mensualidad"]]
    core["Monto_de_la_mensualidad"] = pd.to_numeric(core["Monto_de_la_mensualidad"])
    core['BC_Score_'] = core['BC_Score_'].apply(lambda x: re.sub("[^\d\,]", "", str(x)))
    core['BC_Score_'] = pd.to_numeric(core['BC_Score_'])
    core["rule"]= np.where((core["BC_Score_"]>=680), 1,0)
    core["perfil"]= np.where((core["rule"]==0), "otros","Perfil_X")
    columnas = ["perfil","BC_Score_","Monto_de_la_mensualidad","descuentos","Dependientes","Ingreso_","Ing_Disp"]
    return core[columnas]



selected = ['perfil',
            'CAP_ing_declarado',
     'CAP_validated_final_model',
     'CAP_preds_declarado',
     'CAP_preds_sin_declarado',
     'CAP_real','flag_ingreso_neto_comprobado',
            'CAP_min_pred',
     'CAP_pred_promedio',
     'flag_validated_final_model',
     'flag_preds_declarado',
     'flag_preds_sin_declarado',
     'flag_min_pred',
     'flag_pred_promedio',
     'BC_Score_',
     'Monto_de_la_mensualidad',
     'descuentos',
     'Dependientes',
     'Ingreso_',
     'Ing_Disp',
           'real',
     'ing_declarado',
     'pred_promedio',
     'min_pred']

def create_dinamic(df):
    from IPython.display import HTML
    from pivottablejs import pivot_ui
    HTML('pivottablejs.html')
    return pivot_ui(df[selected])
    