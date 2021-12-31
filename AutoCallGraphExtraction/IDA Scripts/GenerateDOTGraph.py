import idaapi
import idc


def generate_simple_call_chart(flag):
    output_path = idc.get_idb_path().rsplit('.')[0]
    title = ''
    wait = ''
    idaapi.gen_simple_call_chart(output_path, wait, title, flag)
  

generate_simple_call_chart(idaapi.CHART_GEN_DOT)
exit()
