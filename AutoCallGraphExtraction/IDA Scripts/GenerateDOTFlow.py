import idaapi
import idc


def generate_flow_graph(start_ea, end_ea, flag):
    output_path = idc.get_idb_path().rsplit('.')[0]
    title = ''
    pfn = None
    idaapi.gen_flow_graph(output_path, title, pfn, start_ea, end_ea, flag)
  

start_ea = ida_ida.inf_get_min_ea()
end_ea = ida_ida.inf_get_max_ea()

flag = idaapi.CHART_GEN_DOT
generate_flow_graph(start_ea, end_ea, flag)
exit()
