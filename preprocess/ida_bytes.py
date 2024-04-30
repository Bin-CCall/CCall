# -*- coding: utf-8 -*-
import idaapi
import idautils
import idc
import networkx
import pickle
import os
import platform
import copy
from itertools import islice
print(platform.python_version())

# endEA - the start addr of the next basic block
def get_bytes_padding(startEA,endEA):
    
    currentEA = startEA
    code = []
    
    while(currentEA < endEA):
        currentInst = idautils.DecodeInstruction(currentEA)
        temp = []
        normalizes = []
        for i in range(len(currentInst.Operands)):
            #normalizes = []
            if(currentInst.Operands[i].type == o_imm):
                imm_value = currentInst.Operands[i].value # 可用吗
                if(isEnabled(imm_value)):
                    #print('is_enabled:{imm_value}'.format(imm_value=hex(imm_value)))
                    hex_str = hex(imm_value).replace('0x','')
                    if(len(hex_str)%2==1):
                        hex_str.zfill(len(hex_str)+1)
                    imm_value_ = [int(hex_str[i:i+2],16) for i in range(0,len(hex_str),2)]
                    normalizes.extend(imm_value_)
                    normalizes.reverse()
        instr_temp = []
        for b in GetManyBytes(currentEA, currentInst.size):
            instr_temp.append(ord(b))
            '''
            if(ord(b) in normalizes):
                temp.append(-1)
            else:
                temp.append(ord(b))
            '''
        if(len(normalizes)>0):
            for i in range(len(instr_temp)-len(normalizes)+1):
                if(instr_temp[i:i+len(normalizes)]==normalizes):
                    instr_temp[i:i+len(normalizes)] = [-1 for k in range(len(normalizes))]
        temp.extend(instr_temp)
        length = currentInst.size
        while(length%4!=0):
            temp.append(-1)
            length+=1
        currentEA += currentInst.size
        code.extend(temp)
    
    return code
        
        
def get_blocks(f,func):

    all_blocks = {}
    all_blocks[func] = {}
    all_boundary = {}
    all_boundary[func] = {}
    for block in f:
        blocks = []
        if block.startEA != block.endEA:
            boundary = ((hex(block.startEA).replace("L", ""), hex(block.endEA - 1).replace("L", "")))
            #for b in GetManyBytes(block.startEA, block.endEA - block.startEA):  # youmeiyoujiayi
            blocks = get_bytes_padding(block.startEA, block.endEA)
            assert(len(blocks)%4==0)
            all_boundary[func][block.id] = boundary
            all_blocks[func][block.id] = blocks
    
    return all_blocks,all_boundary


def get_bytes_head_direct(ea,flow_chart,func_name,boundary):
    id = find_basic_block_id(boundary,ea,func_name)
    block = get_basic_block_by_id(flow_chart,id)
    code = []
    #inst = idautils.DecodeInstruction(int(ea,16))
    #nea = int(ea,16)+inst.size
    nea = int(ea,16)

    if(nea-block.startEA<=0):
        return [258,-1,-1,-1]
    #for b in GetManyBytes(block.startEA,nea-block.startEA):
        #code.append(ord(b))
    code = get_bytes_padding(block.startEA,nea)
    #code.extend([255,208,-1,-1])
    assert(len(code)%4==0)
    return code

def get_bytes_head(ea,flow_chart,func_name,boundary):
    id = find_basic_block_id(boundary,ea,func_name)
    block = get_basic_block_by_id(flow_chart,id)
    code = []
    inst = idautils.DecodeInstruction(int(ea,16))
    nea = int(ea,16)+inst.size
    assert(nea-block.startEA!=0)
    #for b in GetManyBytes(block.startEA,nea-block.startEA):
        #code.append(ord(b))
    code = get_bytes_padding(block.startEA,nea)
    assert(len(code)%4==0)
    return code


def get_bytes_tail(ea,flow_chart,func_name,boundary):
    id = find_basic_block_id(boundary,ea,func_name)
    block = get_basic_block_by_id(flow_chart,id)
    code = []
    inst = idautils.DecodeInstruction(int(ea,16))
    nea = int(ea,16)+inst.size
    if(block.endEA-nea!=0):
        #for b in GetManyBytes(nea,block.endEA-nea):
            #code.append(ord(b))
        code = get_bytes_padding(nea,block.endEA)
    else:
        code.extend([258,-1,-1,-1])
    assert(len(code)%4==0)
    return code

def combine_branch_sequence(func_branch_sequence,flow_chart,func_name,blocks,boundary,direct):
    all_branch = {}
    all_branch[func_name] = {}
    #{'map_over_members': {'0x1ae6': [0, 1, 2, 3],
    # '0x1bba': [0, 5, 6, 8, 9, 10, 11, 12, 14, 15],
    # '0x1c63': [0, 5, 6, 8, 9, 10, 11, 12, 14, 15, 19, 20, 21, 24]}}
    for x in func_branch_sequence[func_name].keys():#call_addr
        all_branch[func_name][x] = []
        if(direct):
            code_segment = get_bytes_head_direct(x,flow_chart,func_name,boundary)
        else:
            code_segment = get_bytes_head(x,flow_chart,func_name,boundary)
        all_branch[func_name][x].append(code_segment)
        #all_branch[func_name][x].extend(code_segment)
    return all_branch

def combine_branch_sequence_old_whole_sequence(func_branch_sequence,flow_chart,func_name,blocks,boundary,direct):
    all_branch = {}
    all_branch[func_name] = {}
    #{'map_over_members': {'0x1ae6': [0, 1, 2, 3],
    # '0x1bba': [0, 5, 6, 8, 9, 10, 11, 12, 14, 15],
    # '0x1c63': [0, 5, 6, 8, 9, 10, 11, 12, 14, 15, 19, 20, 21, 24]}}
    for x in func_branch_sequence[func_name].keys():#call_addr
        all_branch[func_name][x] = []
        if(direct):
            code_segment = get_bytes_head_direct(x,flow_chart,func_name,boundary)
        else:
            code_segment = get_bytes_head(x,flow_chart,func_name,boundary)
        for xx in func_branch_sequence[func_name][x]:#branch_choice
            temp_choice = []
            for xxx in xx:#block id
                if(xxx!=xx[-1]):
                    temp_choice.extend(blocks[func_name][xxx])
                    temp_choice.extend([256])
                    #all_branch[func_name][x].extend(blocks[func_name][xx])
                    #all_branch[func_name][x].extend([256])
                else:
                    temp_choice.extend(code_segment)
            all_branch[func_name][x].append(temp_choice)
        #all_branch[func_name][x].extend(code_segment)
    return all_branch

def get_direct_call(f,func,at):#!!!!!!!!!!!
    call_addr = {}
    call_addr[func] = []
    targets = {}
    targets[func] = {}
    direct_targets=[]
    for block in f:
        ea = block.startEA
        eea = block.endEA
        assert(ea is not None and eea is not None)

        while(ea < eea):
            inst = idautils.DecodeInstruction(ea)
            assert(inst is not None)
            if(inst.itype==idaapi.NN_call):
                temp = CodeRefsFrom(ea,False)
                temp_targets = []
                for x in temp:
                    name = idc.GetFunctionName(x) 
                    # if(name in at):
                    temp_targets.append(name)
                    direct_targets.append(x)

                if(len(temp_targets)>0):
                    call_addr[func].append(hex(ea).replace("L",""))
                    targets[func][hex(ea)]=temp_targets
                #else:
                    #targets[func]
            ea += inst.size
            #print(targets)
    return call_addr,targets,direct_targets

def get_indirect_call(f,func):
    call_addr = {}
    call_addr[func] = []
    for block in f:
        ea = block.startEA
        eea = block.endEA
        while(ea < eea):
            inst = idautils.DecodeInstruction(ea)
            assert(inst is not None)
            if(inst.itype in (idaapi.NN_callfi,idaapi.NN_callni)):
                call_addr[func].append(hex(ea).replace("L",""))
            ea += inst.size
    return call_addr


def find_basic_block_id(boundary,ea,name):
    for x in boundary[name].keys():
        #print(boundary[name][x])
        assert (int(boundary[name][x][0], 16) <= int(boundary[name][x][1], 16))
        if (int(ea,16)>=int(boundary[name][x][0],16) and int(ea,16)<=int(boundary[name][x][1],16)):
            return x
    #print("not find this block")
    #print(ea)
    return None


def get_basic_block_by_id(f,id):
    for block in f:
        if block.id == id:
            return block
    return None


def get_nx_graph(flowchart, ignore_external=False):
    """Convert an IDA flowchart to a NetworkX graph."""
    nx_graph = networkx.DiGraph()
    for block in flowchart:
        # Make sure all nodes are added (including edge-less nodes)
        nx_graph.add_node(block.id)
        #assert(block.preds() is not None)
        #print("preds:")
        #print(block.preds())
        #for pred in block.preds():
            #assert(pred.0)
            #print(pred)
            #nx_graph.add_edge(pred.id, block.id)
        for succ in block.succs():
            nx_graph.add_edge(block.id, succ.id)
    #assert(networkx.is_weakly_connected(nx_graph))
    return nx_graph


def get_simple_path(graph, k, source_id, target_id):
    if(networkx.has_path(graph,source_id,target_id)==False):
        return None
    return list(
        islice(networkx.shortest_simple_paths(graph, source_id, target_id), k)
    )



'''
def get_simple_path(graph,k,source_id,target_id):
    if(networkx.has_path(graph,source_id,target_id)==False):
        return None
    shortest_path_length = networkx.dijkstra_path_length(graph, source_id, target_id)
    path_length_range = shortest_path_length
    simple_paths = list(networkx.all_simple_paths(graph, source_id, target_id, cutoff=path_length_range))
    simple_paths.sort(key = lambda i:len(i),reverse=False)
    for i in simple_paths:
        assert(i[0]==source_id and i[-1]==target_id)
    if(len(simple_paths)>k):
        return simple_paths[0:k]
    else:
        return simple_paths
'''

def get_head_and_ret(call_addr,cfg,flow_chart,func_name,function,func_bounds,k,func_blocks):
    function_start_ea = function.startEA
    entry_block_id = find_basic_block_id(func_bounds, hex(function_start_ea).replace("L", ""), func_name)
    block_sequence = {}
    ret_block = {}
    ret_block_seq = {}
    block_sequence[func_name] = {}
    ret_block[func_name] = {}
    ret_block_seq[func_name] = {}
    assert (entry_block_id == 0)
    for x in call_addr[func_name]:#call_addr
        code_segment = get_bytes_tail(x,flow_chart,func_name,func_bounds)
        block_sequence[func_name][x]=[]
        ret_block[func_name][x]=[]
        ret_block_seq[func_name][x]=[]
        call_id = find_basic_block_id(func_bounds,x,func_name)
        call_block = get_basic_block_by_id(flow_chart,call_id)
        assert(call_id is not None and call_block is not None) 
        ret_block[func_name][x]=list(call_block.succs())
        temp = []
        for b in ret_block[func_name][x]:
            temp.extend(code_segment)
            temp.extend([256])
            if(b.id in func_blocks[func_name].keys()):
                temp.extend(func_blocks[func_name][b.id])
            else:
                temp.extend([258])
            ret_block_seq[func_name][x].append(temp)
            temp = []
        if call_id == entry_block_id:
            block_sequence[func_name][x].append([call_id])
            continue
        block_sequence[func_name][x]=get_simple_path(cfg,k,entry_block_id,call_id)
        if(block_sequence[func_name][x] is None):
            block_sequence[func_name][x] = []
            block_sequence[func_name][x].append([i for i in range(call_id+1)])

    return block_sequence,ret_block_seq

def get_num_insns(func_ea):

    if func_ea == idc.BADADDR:
        iter = func_ea
        backward_count = 0
        while backward_count < 100:
            backward_count += 1
            iter = idc.prev_head(iter)
            if iter in ALL_FUNCTIONS or idc.print_insn_mnem(iter) == 'retn':
                break
        func_start = idc.next_head(iter)

        iter = func_ea
        forward_count = 0
        while forward_count < 100:
            forward_count += 1
            iter = idc.next_head(iter)
            if iter in ALL_FUNCTIONS or idc.print_insn_mnem(iter) == 'retn':
                break
        func_end = idc.prev_head(iter)

        num_insns = backward_count + forward_count

    else:
        num_insns = len(list(idautils.FuncItems(func_ea)))

    return num_insns


def isAddressTaken(func_addr):
    min_ea  = idc.MinEA()
    max_ea = idc.MaxEA()
    #idc.FindText(当前地址,flag,从当前地址开始搜索的行数,#行中的坐标,searchar)
    addr = 0
    hex_func_addr = hex(func_addr) #type:str
    strip_hex_func_addr = hex_func_addr.replace("0x","")
    length = len(strip_hex_func_addr)
    if(length%2!=0): 
        length+=1
    strip_hex_func_addr.zfill(length)
    assert(len(strip_hex_func_addr)==length)
    hex_func_addr_list = [strip_hex_func_addr[i:i+2] for i in range(0, length, 2)]
    hex_func_addr_list.reverse()
    final_hex_func_addr = ' '.join(hex_func_addr_list)
    assert(final_hex_func_addr[0]==strip_hex_func_addr[-2] and final_hex_func_addr[-2]==strip_hex_func_addr[0])
    addr = idc.FindBinary(min_ea, SEARCH_DOWN | SEARCH_NEXT, final_hex_func_addr)
    
    if(addr>=min_ea and addr<=max_ea):
        print("find_at_function!!!!!!!!!!!!!!!!1")
        print(hex(func_addr))
        return True
    else:
        return False
	
    return False


def getFunctionSequence(func_name,function,func_boundary,func_blocks,k,flow_chart,cfg):
    if(function.flags & idaapi.FUNC_THUNK == idaapi.FUNC_THUNK):
        print("func_thunk!!!!!!!!!!!!!!!!!!!!")
        print(func_name)
        return None
    function_start_ea = function.startEA
    entry_block_id = find_basic_block_id(func_boundary, hex(function_start_ea).replace("L", ""), func_name)
    block_sequence = {}
    byte_sequence = {}
    block_sequence[func_name] = []
    byte_sequence[func_name]=[]
    assert (entry_block_id == 0)
    func_end = idc.FindFuncEnd(function_start_ea)
    func_end = hex(func_end-1)
    end_block_id = find_basic_block_id(func_boundary,func_end, func_name)
    if(end_block_id is None):
        return None
    #assert(end_block_id is not None)
    #assert(end_block_id!=entry_block_id)
    block_sequence[func_name]=get_simple_path(cfg,k,entry_block_id,end_block_id)
    if(block_sequence[func_name] is None):
        block_sequence[func_name] = []
        block_sequence[func_name].append([i for i in func_blocks[func_name].keys()])
        #return None
    assert(block_sequence[func_name] is not None)
    for x in block_sequence[func_name]:
        temp_choice = []
        for xx in x:
            temp_choice.extend(func_blocks[func_name][xx])
            if(xx!=x[-1]):
                temp_choice.extend([256])
        byte_sequence[func_name].append(temp_choice)
    return byte_sequence


def indirect_execute():

    k = 3
    dir_file = idaapi.get_root_filename()
    #print(dir_file)
    cur_dir = os.getcwd()
    root_dir = cur_dir.replace("binary","temp_data")+"/"+dir_file+"/"
    dir = {}
    dir['raw'] = root_dir + "SVFOutput.pkl"
    dir['calladdr'] = root_dir + "calladdr.pkl"
    dir['blocks'] = root_dir + "blocks.pkl"
    #dir['branch'] = root_dir + "branch.pkl"
    dir['head'] = root_dir+"head.pkl"
    dir['ret'] = root_dir+"tail.pkl"
    dir['func_seq'] = root_dir+"func_seq.pkl"
    dir['boundary'] = root_dir + "boundary.pkl"
    dir['address_taken'] = root_dir+"AddressTaken.pkl"
    raw_file = open(dir['raw'],'rb')
    all_blocks = {}
    all_boundary = {}
    all_call_addr = {}
    all_head_addr = {}
    all_head_branch = {}
    all_ret = {}
    at_functions = []
    all_at_seq = {}
    #all_branch = {}
    #{'map_over_members': {'188': [('print_descr', 'ar.c'), ('print_contents', 'ar.c'), ('extract_file', 'ar.c')],
    # '247': [('print_descr', 'ar.c'), ('print_contents', 'ar.c'), ('extract_file', 'ar.c')]}}
    #ats = pickle.load(open(dir['address_taken'],'rb'))

    indtemp = pickle.load(raw_file)
    for func_addr in Functions():
        func_name = idc.GetFunctionName(func_addr)
        if func_addr != BADADDR:
            function = idaapi.get_func(func_addr)
            f = idaapi.FlowChart(function)
            cfg = get_nx_graph(f)
            func_blocks,func_boundary = get_blocks(f,func_name)
            #print("blocks and boundary done")
            all_blocks.update(func_blocks)
            all_boundary.update(func_boundary)
            if(isAddressTaken(func_addr)):
            #if(func_name in ats):
				
                at_functions.append(func_name)
                function_sequence = getFunctionSequence(func_name,function,func_boundary,func_blocks,k,f,cfg) #fs[func_name] = [1,2,3,...,4]
                if(function_sequence is None):
                    at_functions.remove(func_name)
                else:
                    all_at_seq.update(function_sequence)

            if func_name in indtemp.keys():
                func_call_addr = get_indirect_call(f,func_name)
                print(func_call_addr)
                all_call_addr.update(func_call_addr)
                func_head,func_ret = get_head_and_ret(func_call_addr,cfg,f,func_name,function,func_boundary,k,func_blocks)
                all_head_addr.update(func_head)
                all_ret.update(func_ret)
                func_head_branch = combine_branch_sequence_old_whole_sequence(func_head,f,func_name,func_blocks,func_boundary,0)
#combine_branch_sequence(func_head,f,func_name,func_blocks,func_boundary,0)
                all_head_branch.update(func_head_branch)
                #func_branch_sequence,ccc = get_branch_sequence(func_call_addr,f,func_name,function,func_boundary)
                #c+=ccc
                #print(func_branch_sequence)
                #func_branch = combine_branch_sequence(func_branch_sequence,f,func_name,func_blocks,func_boundary)
                #all_branch.update(func_branch)
    '''
    print(all_head_addr['add_exclude_fp'])
    print(all_head_branch['add_exclude_fp'])
    print(all_blocks['add_exclude_fp'])
    print(all_ret['add_exclude_fp'])
    '''
    assert(len(all_blocks)==len(all_boundary))
    #print("before end")
    if(len(at_functions)>0):

        output = open(dir['blocks'], 'wb')
        pickle.dump(all_blocks,output)
        output.close()
        output = open(dir['boundary'],'wb')
        pickle.dump(all_boundary,output)
        output.close()
        output = open(dir['head'],'wb')
        pickle.dump(all_head_branch,output)
        output.close()
        output = open(dir['ret'],'wb')
        pickle.dump(all_ret,output)
        output.close()
        output = open(dir['func_seq'],'wb')
        pickle.dump(all_at_seq,output)
        output.close()
        output = open(dir['address_taken'],'wb')
        pickle.dump(at_functions,output)
        output.close()


    '''
    output = open(dir['branch'], 'wb')
    print(len(indtemp))
    print(len(all_branch))
    for d in all_branch.keys():
        if(len(all_branch[d])==0):
            dd = copy.copy(d)
            del all_branch[dd]
    for d in all_branch.keys():
        if(len(all_branch[d])<len(indtemp[d])):
            print(d)
            print(len(all_branch[d]),len(indtemp[d]))
    print(len(all_branch))
    pickle.dump(all_branch, output)
    output.close()
    '''
    output = open(dir['calladdr'], 'wb')
    pickle.dump(all_call_addr, output)
    output.close()


def direct_execute():

    k = 3
    dir_file = idaapi.get_root_filename()
    print(dir_file)
    cur_dir = os.getcwd()
    root_dir = cur_dir.replace("binary","new_temp_data_direct")+"/"+dir_file+"/"
    dir = {}
    dir['raw'] = root_dir + "SVFOutput.pkl"
    dir['calladdr'] = root_dir + "calladdr.pkl"
    dir['dcalladdr'] = root_dir + "dcalladdr.pkl"
    dir['blocks'] = root_dir + "blocks.pkl"
    #dir['branch'] = root_dir + "branch.pkl"
    dir['head'] = root_dir+"dhead.pkl"
    dir['ret'] = root_dir+"dtail.pkl"
    #dir['func_seq'] = root_dir+"func_seq.pkl"
    dir['boundary'] = root_dir + "boundary.pkl"
    dir['address_taken'] = root_dir+"AddressTaken.pkl"
    dir['direct_targets'] = root_dir+"direct_targets.pkl"
    dir['direct_target_sequence'] = root_dir+"direct_target_sequence.pkl"

    with open(dir['address_taken'],'rb') as at_file:
        at_functions = pickle.load(at_file)
    
    with open(dir['calladdr'],'rb') as ca_file:
        all_incall_addr = pickle.load(ca_file)

    with open(dir['blocks'],'rb') as b_file:
        all_blocks = pickle.load(b_file)
    
    with open(dir['boundary'],'rb') as bry_file:
        all_boundary = pickle.load(bry_file)

    raw_file = open(dir['raw'],'rb')

    all_call_addr = {}
    all_head_addr = {}
    all_head_branch = {}
    all_ret = {}
    all_target = {}
    all_direct_target_seq = {}
    indtemp = pickle.load(raw_file)
    
    for func_addr in Functions():
        direct_targets=[]
        func_name = idc.GetFunctionName(func_addr)
        assert(func_name is not None)
        #if (func_addr != BADADDR) and (func_name in all_incall_addr.keys()):
        if(func_addr!=BADADDR):
            function = idaapi.get_func(func_addr)
            f = idaapi.FlowChart(function)
            cfg = get_nx_graph(f)
            func_blocks = all_blocks#!!!!!!!!!!!!!!!!!!!!!!
            func_boundary = all_boundary #!!!!!!!!!!!!!!!!!!!!1111

            func_call_addr,targets,direct_targets = get_direct_call(f,func_name,at_functions)

            all_call_addr.update(func_call_addr)
            if(targets is not None):
                all_target.update(targets)
            func_head,func_ret = get_head_and_ret(func_call_addr,cfg,f,func_name,function,func_boundary,k,func_blocks)
            all_head_addr.update(func_head)
            all_ret.update(func_ret)
            func_head_branch = combine_branch_sequence(func_head,f,func_name,func_blocks,func_boundary,1)
            all_head_branch.update(func_head_branch)
            
            for addr in direct_targets:
                fname=idc.GetFunctionName(addr)
                function = idaapi.get_func(addr)
                f = idaapi.FlowChart(function)
                cfg = get_nx_graph(f)
                func_blocks,func_boundary = get_blocks(f,fname)
                function_sequence = getFunctionSequence(fname,function,func_boundary,func_blocks,k,f,cfg)
                if(function_sequence is not None):
                    all_direct_target_seq.update(function_sequence)


    output = open(dir['head'],'wb')
    pickle.dump(all_head_branch,output)
    output.close()

    output = open(dir['ret'],'wb')
    pickle.dump(all_ret,output)
    output.close()

    output = open(dir['dcalladdr'], 'wb')
    pickle.dump(all_call_addr, output)
    output.close()


    output = open(dir['direct_target_sequence'], 'wb')
    pickle.dump(all_direct_target_seq, output)
    output.close()


    with open(dir['direct_targets'],'wb') as f:
        pickle.dump(all_target,f)

    

    

if __name__ == "__main__":
    idc.Wait()
    # cur_dir = os.getcwd()
    # print(cur_dir)
    indirect_execute()
    #direct_execute()
    idc.Exit(0)

