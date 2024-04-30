# -*- coding: utf-8 -*-
import os
import pickle
import sys
import random

def handle_indirect_call_at(file,at):
    res = {}
    (pre_func,pre_line) = (0 , 0)
    cur_targets = []
    while(True):
        data = file.readline()
        if not data:
            break
        print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(data)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_split = data.split(" ")
        (cur_func,cur_line) = (data_split[0],data_split[1])
        if(cur_func!=pre_func or cur_line!=pre_line):
            if (pre_func != cur_func and cur_func not in res.keys()):
                res[cur_func] = {}
            if(pre_func,pre_line)!=(0,0):
                res[pre_func][pre_line] = cur_targets
            (pre_func,pre_line) = (cur_func,cur_line)
            cur_targets = []
        #file_name = ((data_split[3].replace('\n','')).split('/'))[-1]
        if(data_split[2] not in cur_targets):
            cur_targets.append((data_split[2]).split('.')[0])
            #assert data_split[2].split('.')[0] in at
    assert pre_func==cur_func and pre_line==cur_line
    res[pre_func][pre_line] = cur_targets
    return res

def handle_indirect_call(file):
    res = {}
    (pre_func,pre_line) = (0 , 0)
    (cur_func,cur_line) = (-1,-1)
    cur_targets = []
    while(True):
        data = file.readline()
        if not data:
            break
        print("!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(data)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        data_split = data.split(" ")
        (cur_func,cur_line) = (data_split[0],data_split[1])
        if(cur_func!=pre_func or cur_line!=pre_line):
            if (pre_func != cur_func and cur_func not in res.keys()):
                res[cur_func] = {}
            if(pre_func,pre_line)!=(0,0):
                if(pre_line in res[pre_func].keys()):
                    res[pre_func][pre_line].extend(cur_targets)
                else:
                    res[pre_func][pre_line]=cur_targets
            (pre_func,pre_line) = (cur_func,cur_line)
            cur_targets = []
        #file_name = ((data_split[3].replace('\n','')).split('/'))[-1]
        if((data_split[2]).split('.')[0] not in cur_targets):
            cur_targets.append((data_split[2]).split('.')[0])
            #assert data_split[2].split('.')[0] in at
    if(not (pre_func==cur_func and pre_line==cur_line)):
        print((pre_func,cur_func,pre_line,cur_line)) 
        print(cur_targets)
        return None
        assert 0
    res[pre_func][pre_line] = cur_targets
    return res

def handle_AT(input):
    at = []
    while(True):
        data = input.readline()
        if not data:
            break
        if((data.strip()).split(".")[0]not in at):
            at.append((data.strip()).split(".")[0])
    #print(at)
    return at

def handle_data(root_dir):
    '''
    dir = {}
    dir['input'] = root_dir+"indcall\\"
    dir['filter'] = root_dir + "AT\\"
    dir['output'] = root_dir+"indtemp\\"
    dir['addresstaken'] = root_dir+"AddressTaken\\"
    '''
    indirect_call_list = os.listdir(root_dir)
    #indirect_call_list=['nginx']
    print(indirect_call_list)
    for x in indirect_call_list:
        #input = open(dir['filter']+x,"r")
        #input = open(root_dir + x +'/AddressTaken.data',"r")
        #at = handle_AT(input)
        #print(len(at))
        #input.close()
        input_file = open(root_dir+x+"/SVFOutput.data","r")
        #res = handle_indirect_call_at(input_file,at)
        res = handle_indirect_call(input_file)
        if(res==None):
            print(x,root_dir)
            assert(0)
        input_file.close()
        output_file = open(root_dir+x+'/SVFOutput.pkl',"wb")
        pickle.dump(res,output_file,protocol=2)
        print(len(res))
        for d in res.keys():
            print(d)
            print(len(res[d]))
            for dd in res[d].keys():
                print(len(res[d][dd]))
        #print(res)
        output_file.close()
        #output = open(root_dir+x+"/AddressTaken.pkl", "wb")
        #pickle.dump(at, output, protocol=2)
        #output.close()
if __name__ == "__main__":
    root_dir = "./temp_data/"#"../new_dataset/temp_data/"
    op_list = os.listdir(root_dir)
    for op in op_list:
        handle_data(root_dir+op+"/")

