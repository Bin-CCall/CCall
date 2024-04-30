import os
import subprocess
import pickle
import copy
def check_exist(call,blocks):
	call_new = []
	for x in call:
		print("This is x in call in check exist")
		print(x)
		#assert(x[0].split('.')[0] in blocks.keys())
		#if(x[0].split('.')[0] in blocks.keys()):
		if(x.strip('\n') in blocks.keys()):
			if(len(blocks[x.strip('\n')])!=0):
				print(x)
				call_new.append(x.strip('\n'))
		else:
			print("---------------error blank--------------")
			print(x.strip('\n'))
			#print(blocks.keys())
			print("---------------error blank--------------")
	'''
	print("###############################")
	print(len(call))
	print(len(call_new))
	print("###############################")
	'''
	return call_new


def remove_item(call_final):
	call_temp = copy.deepcopy(call_final)
	for x in call_final.keys():
		for xx in call_final[x].keys():
			if(len(call_temp[x][xx])==0):
				call_temp[x].pop(xx)
		if(len(call_temp[x])==0):
			call_temp.pop(x)
	print("This is call_temp")
	print(call_temp)
	return call_temp


def batch_direct(dir,elf_list):
	for e in elf_list:
		print(e)	
		blocks = clean_blocks(dir+e+"/blocks.pkl")
		#blocks = clean_instructions(dir+e+"/instructions.pkl")
		calladdr = pickle.load(open(dir+e+"/dcalladdr.pkl",'rb'))
		print(calladdr)
		dcall_target = pickle.load(open(dir+e+"/direct_targets.pkl",'rb'))
		
		call_final = {}
		for x in calladdr.keys():
			print(x)
			call_final[x] = {}
			for xx in calladdr[x]:
				dcall_target[x][xx] = check_exist(dcall_target[x][xx],blocks)
				call_final[x][xx] = dcall_target[x][xx]
		call_final = remove_item(call_final)
		if(len(call_final)!=0):
			print(dir+e+"/dfinal.pkl")
			output = open(dir+e+"/dfinal.pkl",'wb')
			pickle.dump(call_final,output)
			output.close()
			#print(call_final)


def batch_addr2line(dir,elf_list):
	for e in elf_list:
		print(e)
		blocks = clean_blocks(dir+e+"/blocks.pkl")
		#blocks = clean_instructions(dir+e+"/instructions.pkl")
		calladdr = pickle.load(open(dir+e+"/calladdr.pkl",'rb'))
		print(calladdr)
		callsvf = pickle.load(open(dir+e+"/SVFOutput.pkl",'rb'))
		binary_file = dir.replace("temp_data","binary_dbg")+e
		call_final = {}
		for x in calladdr.keys():
			print(x)
			cur_line = {}
			flag = 0
			call_final[x] = {}
			for xx in calladdr[x]:
				cmd = "addr2line -e "+binary_file+" "+xx
				#print(cmd)
				s = os.popen(cmd)
				data = s.read()
				line = data.split(":")[1].replace("\n","")
				print("This is line")
				print(line)
				if(line in cur_line.keys()):
					#'''
					print("********************************")
					print(binary_file)
					print(xx)
					print(line)
					print(x)
					print(calladdr[x])
					print(callsvf[x])
					print("********************************")
					#'''
					if x in callsvf.keys() and x in call_final.keys():
						if line in callsvf[x].keys():
							abandon = callsvf[x].pop(line)
						if cur_line[line] in call_final[x].keys():
							abandon = call_final[x].pop(cur_line[line])
					continue
				else:
					cur_line[line]=xx
				#file = data.split(":")[0].split("/")[-1]
				if flag==0 and line in callsvf[x].keys():
					#call_final[x][xx] = []
					'''
					print("---------------------------------")
					print("callsvf[x][line]")
					print(len(callsvf[x][line]))
					print("callsvf[x]")
					print(len(callsvf[x]))
					'''
					print("This is callsvf")
					print(callsvf[x][line])
					callsvf[x][line] = check_exist(callsvf[x][line],blocks)
					'''
					if(len(callsvf[x][line])==0):
						abandon = callsvf[x].pop(line)
						if(len(callsvf[x])==0):
							abandon = callsvf.pop(x)
					'''
					if (x in callsvf.keys() and line in callsvf[x].keys()):
						print("yes!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
						call_final[x][xx] = callsvf[x][line]
						print(call_final[x][xx])
					if(len(callsvf)==0):
						flag = 1
		call_final = remove_item(call_final)
		if(len(call_final)!=0):
			print(dir+e+"/final.pkl")
			output = open(dir+e+"/final.pkl",'wb')
			pickle.dump(call_final,output)
			output.close()
			#print(call_final)

def clean_blocks(dir):
	blocks = pickle.load(open(dir,'rb'))
	new_blocks = {}
	#print(blocks)
	for x in blocks.keys():
		find_flag = 0
		if(len(blocks[x].keys())>1):
			new_blocks[x] = blocks[x]
			find_flag = 1
			continue
		else:
			for xx in blocks[x].keys():
				for i in blocks[x][xx]:
					if i != 0:
						new_blocks[x] = blocks[x]
						find_flag = 1
						continue
		if find_flag == 0:
			print(x)
	print("+++++++++++++++++++++")
	print(len(new_blocks.keys()))
	print(len(blocks.keys()))
	print("+++++++++++++++++++++")
	output = open(dir,'wb')
	pickle.dump(new_blocks,output)
	output.close()
	return new_blocks


def clean_instructions(dir):
	blocks = pickle.load(open(dir,'rb'))
	new_blocks = {}
	#print(blocks)
	for x in blocks.keys():
		find_flag = 0
		if(len(blocks[x].keys())>2):
			new_blocks[x] = blocks[x]
			find_flag = 1
			continue
	
		else:
			for xx in blocks[x].keys():
				for i in blocks[x][xx]:
					if (i not in ['<ins>','</ins>']):
						new_blocks[x] = blocks[x]
						find_flag = 1
						continue
		if find_flag == 0:
			print(x)
		#'''
	print("+++++++++++++++++++++")
	print(len(new_blocks.keys()))
	print(len(blocks.keys()))
	print("+++++++++++++++++++++")
	output = open(dir,'wb')
	pickle.dump(new_blocks,output)
	output.close()
	return new_blocks

def handle_addr2line(root_dir):
	'''
	dir = {}
	dir['branch'] = root_dir + "branch/"
	dir['calladdr'] = root_dir + "calladdr/"
	dir['blocks'] = root_dir + "blocks/"
	dir['boundary'] = root_dir + "boundary/"
	dir['indcall'] = root_dir + "indtemp/"
	dir['binary'] = root_dir + "binary/"
	dir['out'] = root_dir + "indfinal/"
	'''
	elf_list = os.listdir(root_dir)
	#elf_list=['memcached']
	print(elf_list)
	batch_addr2line(root_dir,elf_list)

def handle_direct(root_dir):
	'''
	dir = {}
	dir['branch'] = root_dir + "branch/"
	dir['calladdr'] = root_dir + "calladdr/"
	dir['blocks'] = root_dir + "blocks/"
	dir['boundary'] = root_dir + "boundary/"
	dir['indcall'] = root_dir + "indtemp/"
	dir['binary'] = root_dir + "binary/"
	dir['out'] = root_dir + "indfinal/"
	'''
	elf_list = os.listdir(root_dir)
	print(elf_list)
	batch_direct(root_dir,elf_list)

if __name__ == "__main__":
	root_dir = "./temp_data/"
	op_list = os.listdir(root_dir)
	#op_list = ['clang_O'+str(i) for i in range(4)]
	op_list=['clang_O1']
	for op in op_list:	
		handle_addr2line(root_dir+op+"/")
		#handle_direct(root_dir+op+"/")

	

	'''
					}
{'shaxxx_stream': 
{'0x4ef': [85, 72, 137, 229, 72, 129, 236, 32, 1, 0, 0, 72, 137, 125, 240, 72, 137, 117, 232, 72, 137, 85, 224, 72, 137, 77, 216, 76, 137, 69, 208, 76, 137, 77, 200, 72, 139, 125, 240, 72, 139, 117, 232, 72, 139, 85, 224, 72, 139, 77, 216, 232, 168, 82, 0, 0, 65, 137, 194, 131, 232, 251, 68, 137, 149, 252, 254, 255, 255, 137, 133, 248, 254, 255, 255, 15, 132, 36, 0, 0, 0, 256, 233, 0, 0, 0, 0, 256, 139, 133, 252, 254, 255, 255, 133, 192, 15, 133, 29, 0, 0, 0, 256, 184, 72, 128, 0, 0, 137, 199, 232, 251, 85, 0, 0, 72, 137, 69, 192, 72, 131, 125, 192, 0, 15, 133, 12, 0, 0, 0, 256, 72, 141, 189, 16, 255, 255, 255, 255, 85, 208], '0x612': [85, 72, 137, 229, 72, 129, 236, 32, 1, 0, 0, 72, 137, 125, 240, 72, 137, 117, 232, 72, 137, 85, 224, 72, 137, 77, 216, 76, 137, 69, 208, 76, 137, 77, 200, 72, 139, 125, 240, 72, 139, 117, 232, 72, 139, 85, 224, 72, 139, 77, 216, 232, 168, 82, 0, 0, 65, 137, 194, 131, 232, 251, 68, 137, 149, 252, 254, 255, 255, 137, 133, 248, 254, 255, 255, 15, 132, 36, 0, 0, 0, 256, 233, 0, 0, 0, 0, 256, 139, 133, 252, 254, 255, 255, 133, 192, 15, 133, 29, 0, 0, 0, 256, 184, 72, 128, 0, 0, 137, 199, 232, 251, 85, 0, 0, 72, 137, 69, 192, 72, 131, 125, 192, 0, 15, 133, 12, 0, 0, 0, 256, 72, 141, 189, 16, 255, 255, 255, 255, 85, 208, 256, 72, 199, 133, 8, 255, 255, 255, 0, 0, 0, 0, 256, 72, 139, 125, 240, 232, 162, 85, 0, 0, 131, 248, 0, 15, 132, 5, 0, 0, 0, 256, 233, 202, 0, 0, 0, 256, 72, 131, 189, 8, 255, 255, 255, 0, 15, 134, 23, 0, 0, 0, 256, 72, 139, 69, 200, 72, 139, 117, 224, 72, 141, 189, 16, 255, 255, 255, 255, 208]}}

{'sha224_init_ctx': {0: ('0x90', '0x112')}, 
'sha256_finish_ctx': {0: ('0x240', '0x26b')}, 
'sha256_read_ctx': {0: ('0x120', '0x13e'), 1: ('0x13f', '0x148'), 2: ('0x149', '0x14c'), 3: ('0x14d', '0x18c'), 4: ('0x18d', '0x196')}, 'sha256_conclude_ctx': {0: ('0x270', '0x2cb'), 1: ('0x2cc', '0x2d8'), 2: ('0x2d9', '0x3bc')}, 'free': {0: ('0x5ac0', '0x5ac7')}, 'sha256_process_block': {0: ('0x9d0', '0xaba'), 1: ('0xabb', '0xac8'), 2: ('0xac9', '0xad2'), 3: ('0xad3', '0xadf'), 4: ('0xae0', '0xb36'), 5: ('0xb37', '0xb3b'), 6: ('0xb3c', '0xc1d'), 7: ('0xc1e', '0xd02'), 8: ('0xd03', '0xdea'), 9: ('0xdeb', '0xecf'), 10: ('0xed0', '0xfb4'), 11: ('0xfb5', '0x1096'), 12: ('0x1097', '0x1175'), 13: ('0x1176', '0x1257'), 14: ('0x1258', '0x1339'), 15: ('0x133a', '0x141e'), 16: ('0x141f', '0x1506'), 17: ('0x1507', '0x15eb'), 18: ('0x15ec', '0x16d0'), 19: ('0x16d1', '0x17b2'), 20: ('0x17b3', '0x1891'), 21: ('0x1892', '0x1973'), 22: ('0x1974', '0x1ab7'), 23: ('0x1ab8', '0x1bfe'), 24: ('0x1bff', '0x1d48'), 25: ('0x1d49', '0x1e8f'), 26: ('0x1e90', '0x1fd6'), 27: ('0x1fd7', '0x211a'), 28: ('0x211b', '0x225b'), 29: ('0x225c', '0x239f'), 30: ('0x23a0', '0x24e3'), 31: ('0x24e4', '0x262a'), 32: ('0x262b', '0x2774'), 33: ('0x2775', '0x28bb'), 34: ('0x28bc', '0x2a02'), 35: ('0x2a03', '0x2b46'), 36: ('0x2b47', '0x2c87'), 37: ('0x2c88', '0x2dcb'), 38: ('0x2dcc', '0x2f0f'), 39: ('0x2f10', '0x3056'), 40: ('0x3057', '0x31a0'), 41: ('0x31a1', '0x32e7'), 42: ('0x32e8', '0x342e'), 43: ('0x342f', '0x3572'), 44: ('0x3573', '0x36b3'), 45: ('0x36b4', '0x37f7'), 46: ('0x37f8', '0x393b'), 47: ('0x393c', '0x3a82'), 48: ('0x3a83', '0x3bcc'), 49: ('0x3bcd', '0x3d13'), 50: ('0x3d14', '0x3e5a'), 51: ('0x3e5b', '0x3f9e'), 52: ('0x3f9f', '0x40df'), 53: ('0x40e0', '0x4223'), 54: ('0x4224', '0x4367'), 55: ('0x4368', '0x44ae'), 56: ('0x44af', '0x45f8'), 57: ('0x45f9', '0x473f'), 58: ('0x4740', '0x4886'), 59: ('0x4887', '0x49ca'), 60: ('0x49cb', '0x4b0b'), 61: ('0x4b0c', '0x4c4f'), 62: ('0x4c50', '0x4d93'), 63: ('0x4d94', '0x4eda'), 64: ('0x4edb', '0x5024'), 65: ('0x5025', '0x516b'), 66: ('0x516c', '0x52b2'), 67: ('0x52b3', '0x53f6'), 68: ('0x53f7', '0x5537'), 69: ('0x5538', '0x5711'), 70: ('0x5712', '0x5717')}, 'shaxxx_stream': {0: ('0x440', '0x490'), 1: ('0x491', '0x495'), 2: ('0x496', '0x4a3'), 3: ('0x4a4', '0x4a8'), 4: ('0x4a9', '0x4b4'), 5: ('0x4b5', '0x4c0'), 6: ('0x4c1', '0x4db'), 7: ('0x4dc', '0x4e7'), 8: ('0x4e8', '0x4f1'), 9: ('0x4f2', '0x4fc'), 10: ('0x4fd', '0x50e'), 11: ('0x50f', '0x513'), 12: ('0x514', '0x57d'), 13: ('0x57e', '0x582'), 14: ('0x583', '0x590'), 15: ('0x591', '0x5a2'), 16: ('0x5a3', '0x5b7'), 17: ('0x5b8', '0x5bc'), 18: ('0x5bd', '0x5c1'), 19: ('0x5c2', '0x5dd'), 20: ('0x5de', '0x5eb'), 21: ('0x5ec', '0x602'), 22: ('0x603', '0x62a'), 23: ('0x62b', '0x636')}, 'sha256_stream': {0: ('0x3f0', '0x437')}, 'ferror_unlocked': {0: ('0x5ab0', '0x5ab7')}, 'sha256_buffer': {0: ('0x690', '0x6df')}, 'memcpy': {0: ('0x5ad0', '0x5ad7')}, 'sha256_process_bytes': {0: ('0x6e0', '0x702'), 1: ('0x703', '0x723'), 2: ('0x724', '0x730'), 3: ('0x731', '0x73f'), 4: ('0x740', '0x782'), 5: ('0x783', '0x7e7'), 6: ('0x7e8', '0x802'), 7: ('0x803', '0x80d'), 8: ('0x80e', '0x81f'), 9: ('0x820', '0x824'), 10: ('0x825', '0x82f'), 11: ('0x830', '0x883'), 12: ('0x884', '0x888'), 13: ('0x889', '0x8bf'), 14: ('0x8c0', '0x8c4'), 15: ('0x8c5', '0x8cf'), 16: ('0x8d0', '0x90e'), 17: ('0x90f', '0x960'), 18: ('0x961', '0x96c'), 19: ('0x96d', '0x972')}, 'sha224_finish_ctx': {0: ('0x3c0', '0x3eb')}, 'afalg_stream': {0: ('0x5720', '0x573a')}, 'malloc': {0: ('0x5ac8', '0x5acf')}, 'sha224_read_ctx': {0: ('0x1c0', '0x1de'), 1: ('0x1df', '0x1e8'), 2: ('0x1e9', '0x22c'), 3: ('0x22d', '0x236')}, 'set_uint32': {0: ('0x1a0', '0x1b5')}, 'sha256_init_ctx': {0: ('0x0', '0x2b'), 1: ('0x2c', '0x82')}, 'sha224_stream': {0: ('0x640', '0x687')}, 'feof_unlocked': {0: ('0x5aa8', '0x5aaf')}, 'sha224_buffer': {0: ('0x980', '0x9cf')}, 'fread_unlocked': {0: ('0x5ab8', '0x5abf')}}

{'shaxxx_stream': ['0x4ef', '0x612']}

{'shaxxx_stream': {'196': [('sha256_init_ctx', 'sha256.c'), ('sha224_init_ctx', 'sha256.c')], '253': [('sha256_finish_ctx', 'sha256.c'), ('sha224_finish_ctx', 'sha256.c')]}}

goal:{'shaxxx_stream': {'0x4ef': blabla
	'''


