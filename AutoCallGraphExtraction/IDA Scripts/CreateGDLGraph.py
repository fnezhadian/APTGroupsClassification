import idc

def create_gdl():
    path = idc.get_idb_path().rsplit('.')[0] + '.gdl'
    idc.gen_simple_call_chart(path, 'Call Gdl', idc.CHART_GEN_GDL)

create_gdl()
exit()

