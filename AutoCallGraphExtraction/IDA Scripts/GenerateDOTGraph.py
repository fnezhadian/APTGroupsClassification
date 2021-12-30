import idaapi
import idc

def generate_simple_call_chart(start_ea, end_ea, flag):
    output_path = idc.get_idb_path().rsplit('.')[0]
    title = ''
    wait = ''
    idaapi.gen_simple_call_chart(output_path, wait, title, flag)
  

start_ea = ida_ida.inf_get_min_ea()
end_ea = ida_ida.inf_get_max_ea()

flag = idaapi.CHART_GEN_DOT
generate_simple_call_chart(start_ea, end_ea, flag)
exit()
