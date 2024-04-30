# coding=utf-8
import os
import subprocess
import pickle
import random
from itertools import product

def get_negative_data(root_dir,f_list,label = 257):
	for f in f_list:
		dir = get_dir(root_dir,f)
		if(os.path.exists(dir['indfinal'])):
			indfinal = pickle.load(open(dir['indfinal'],"rb"))
			head_seq = pickle.load(open(dir['head'],"rb"))
			tail_seq = pickle.load(open(dir['tail'],"rb"))
			func_seq = pickle.load(open(dir['func_seq'],"rb"))
			blocks = pickle.load(open(dir['blocks'],"rb"))
			at = pickle.load(open(dir['filter'],'rb'))
			negative = {}
			call_target = {}
			#get_call_target
			for x in indfinal.keys():#x:func_name
				#call_target[x] = {}
				for xx in indfinal[x].keys():#xx:indirect_call_address
					call_target[(x,xx)] = [xxx for xxx in indfinal[x][xx]]

			for x in indfinal.keys():
				#negative[x] = {}
				for xx in indfinal[x].keys():
					negative[(x,xx)] = {}
					if x in head_seq.keys() and xx in head_seq[x].keys():
						head = head_seq[x][xx]
						tail = tail_seq[x][xx]
						if(len(at)>4*len(call_target[(x,xx)])):
							at_sample = random.sample(at,4*len(call_target[(x,xx)]))
						else:
							at_sample = at
						for xxx in at_sample:
							if xxx not in call_target[(x,xx)]:
								# print("function:"+xxx)
								middle = func_seq[xxx]
								negative[(x,xx)][xxx]=get_sample(head,middle,tail)
					else:
						#print(f)
						#print(x)
						print("**--------------------------------------")

			#print(negative)
			output = open(dir['negative']+f+".pkl","wb")
			pickle.dump(negative,output,protocol=2)
			print(f+' neg done')
			output.close()

def get_positive_data(root_dir,f_list,label = 257):
	for f in f_list:
		dir = get_dir(root_dir,f)
		if(os.path.exists(dir['indfinal'])):
			indfinal = pickle.load(open(dir['indfinal'],"rb"))
			head_seq = pickle.load(open(dir['head'],"rb"))
			tail_seq = pickle.load(open(dir['tail'],"rb"))
			func_seq = pickle.load(open(dir['func_seq'],"rb"))
			blocks = pickle.load(open(dir['blocks'],"rb"))
			positive = {}
			for x in indfinal.keys():#x:func_name
				#positive[x]={}
				for xx in indfinal[x].keys():#xx:indirect_call_address
					positive[(x,xx)]={}
					if x in head_seq.keys() and xx in head_seq[x].keys():
						head = head_seq[x][xx]
						tail = tail_seq[x][xx]
						for xxx in indfinal[x][xx]:
							if(xxx not in func_seq.keys()):
								# print("----------------------------Not_in_address_taken_list----------------------------")
								# print(xxx)
								#print(dir['filter'])
								# print("----------------------------Not_in_address_taken_list----------------------------")
								continue
							#assert(xxx in func_seq.keys())
							# print("function:"+xxx)
							middle = func_seq[xxx]
							positive[(x,xx)][xxx]=get_sample(head,middle,tail)

					# else:
						#print(f)
						#print(x)
						# print("*--------------------------------------")

			#print(positive)
			output = open(dir['positive']+f+".pkl","wb")
			pickle.dump(positive,output,protocol=2)
			print(f + ' pos done')
			output.close()

def get_sample(head,middle,tail):
	final = []
	list1 = [i for i in range(len(head))]
	list2 = [i2 for i2 in range(len(middle))]
	list3 = [i3 for i3 in range(len(tail))]
	l = [list1,list2,list3]
	temp = []
	for item in product(*l):
		temp.extend(head[item[0]])
		temp.extend([257])
		temp.extend(middle[item[1]])
		temp.extend([257])
		temp.extend(tail[item[2]])
		final.append(temp)
		temp = []
	return final

def cleaning_combining(list,ndir,root,name,label):#deduplication
    # root = "../data_append/results/"+op
    # dir = "../data_append/positive_negative/"+op
	final_data = []
	final_index = []
	length = 0
	for f in list:
		if(not os.path.exists(ndir+f)):
			continue
		input = open(ndir+f,"rb")
		data = pickle.load(input)
		# print(data)
		for (x,xx) in data.keys():
			for xxx in data[(x,xx)].keys():
				for d in data[(x,xx)][xxx]:
					length+=1
					if d not in final_data and len(d)>4:
						# print(length)
						final_data.append(d)
						final_index.append('@'.join([x,xx,xxx]))
					# else:
					# 	print("@")
		input.close()
	#print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
	#print(len(final_data))
	#print(length)
	#print(final)
	#print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
	output = open(root+label+'_'+name+".pkl","wb")
	pickle.dump(final_data,output,protocol=2)
	output.close()
	if('test' in label):
		tput = open(root+label+'_idx_'+name+".pkl","wb")
		pickle.dump(final_index,tput,protocol=2)
		tput.close()

def make_dirs(dir):
	if(not os.path.exists(dir)):
		os.makedirs(dir)
	else:
		pass
	return True


def get_dir(root_dir,f):
	dir = {}
	dir['head'] = root_dir +f+ "/head.pkl"
	dir['tail'] = root_dir +f+ "/tail.pkl"
	dir['func_seq'] = root_dir +f+ "/func_seq.pkl"
	dir['blocks'] = root_dir + f + "/blocks.pkl"
	dir['boundary'] = root_dir + f+"/boundary.pkl"
	dir['indcall'] = root_dir + f+ "/SVFOutput.pkl"
	dir['indfinal'] = root_dir +f+ "/final.pkl"
	dir['filter'] = root_dir + f + "/AddressTaken.pkl"
	dir['positive'] = root_dir.replace("temp_data","positive")
	dir['negative'] = root_dir.replace("temp_data","negative")
	make_dirs(dir['negative'])
	make_dirs(dir['positive'])
	return dir

def handle_combine(root_dir,final_dir,f_list,label):
    # root_dir = "../data_append/temp_data/"+op
	# final_dir = "../data_append/results/"+op
	dir = {}
	dir['positive'] = root_dir.replace("temp_data","positive")
	dir['negative'] = root_dir.replace("temp_data","negative")
	#f_list = os.listdir(root_dir)
	get_positive_data(root_dir,f_list)
	get_negative_data(root_dir,f_list)
	#pos_list = os.listdir(dir['positive'])
	#neg_list = os.listdir(dir['negative'])
	pos_list = [x+'.pkl' for x in f_list]
	neg_list = [x+'.pkl' for x in f_list]
	print('cleaning')
	cleaning_combining(pos_list,dir['positive'],final_dir,'positive',label)
	cleaning_combining(neg_list,dir['negative'],final_dir,'negative',label)
	print('cleaning done')

def combine_together(final_dir,name):
    # final_dir = "../data_append/results/"
	# op_list = ['clang_O2_m32']
	label = ['negative','positive']
	for i in label:
		print(label)
		output_data = []
		output_index = []
		output_file_data = final_dir+name+'_'+i+".pkl"
		output_file_index = final_dir+name+'_idx_'+i+".pkl"
		for op in op_list:
			print(op)
			cur_dir_data = final_dir+op+"/"+name+'_'+i+".pkl"
			cur_data = pickle.load(open(cur_dir_data,'rb'))
			output_data.extend(cur_data)
			if('test' in name):
				cur_dir_idx = final_dir+op+"/"+name+'_idx_'+i+".pkl"
				cur_index = pickle.load(open(cur_dir_idx,'rb'))
				output_index.extend(cur_index)
		output = open(output_file_data,"wb")
		pickle.dump(output_data,output,protocol=2)
		output.close()
		if('test' in name):
			tput = open(output_file_index,"wb")
			assert(len(output_index)==len(output_data))
			pickle.dump(output_index,tput,protocol=2)
			tput.close()

if __name__ == "__main__":
	root_dir = "./temp_data/"
	final_dir = "./results_indirect_train/"
	make_dirs(final_dir)
	op_list = os.listdir(root_dir)
	# op_list = ['clang_O'+str(i) for i in range(0,4)]
	train_file = ['bitcoin-cli', 'busybox_unstripped', 'bzip2', 'dealII', 'diff', 'fzputtygen', 'gcc', 'gobmk',
				   'h264ref', 'hmmer', 'milc', 'namd', 'nginx', 'omnetpp', 'openssl', 'perlbench']
	test_file = ['povray', 'sjeng', 'soplex', 'Xalan']

	for op in op_list:
		make_dirs(final_dir+op)
		print(op)
		C_train_list=os.listdir(root_dir+op)
		# handle_combine(root_dir+op+"/",final_dir+op+"/",C_test_list,'C_test')
		handle_combine(root_dir+op+"/",final_dir+op+"/",C_train_list,'C_train')

	#handle_combine(root_dir+"clang_O2/",final_dir+"clang_O2/")

	#combine_together(final_dir,'C_test_m32')
	combine_together(final_dir,'C_train')

	#'''


