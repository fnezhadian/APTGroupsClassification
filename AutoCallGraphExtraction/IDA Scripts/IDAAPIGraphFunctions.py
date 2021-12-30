import idaapi
import idc

def generate_flow_graph(start_ea, end_ea, flag):
    output_path = idc.get_idb_path().rsplit('.')[0] + "_flow"
    title = ''
    pfn = None
    idaapi.gen_flow_graph(output_path, title, pfn, start_ea, end_ea, flag)
    

def generate_simple_call_chart(start_ea, end_ea, flag):
    output_path = idc.get_idb_path().rsplit('.')[0] + "_simple_call"
    title = ''
    wait = ''
    idaapi.gen_simple_call_chart(output_path, wait, title, flag)
    
    
def generate_complex_call_chart(start_ea, end_ea, flag):
    output_path = idc.get_idb_path().rsplit('.')[0] + "_complex_call"
    title = ''
    wait = ''
    recursion_depth = -1
    idaapi.gen_complex_call_chart(output_path, wait, title, start_ea, end_ea, flag, recursion_depth)
    

start_ea = ida_ida.inf_get_min_ea()
end_ea = ida_ida.inf_get_max_ea()

flag = idaapi.CHART_GEN_DOT
generate_flow_graph(start_ea, end_ea, flag)
generate_simple_call_chart(start_ea, end_ea, flag)

flag = idaapi.CHART_GEN_GDL
generate_flow_graph(start_ea, end_ea, flag)
generate_simple_call_chart(start_ea, end_ea, flag)

exit()
