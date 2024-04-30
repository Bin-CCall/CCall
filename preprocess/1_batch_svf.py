import os
import subprocess
import sys
import time
def make_dir(path):
	if(os.path.exists(path)==False):
		cmd_mkdir ="mkdir " + path
		print(cmd_mkdir)
		p=os.popen(cmd_mkdir,"r")
		print(p.read())
		p.close()

def batch_svf():
	op_path  = "./binary/"
	bc_path  = "./bitcode/"
	temp_path = "./temp_data/"
	plugin_path = "./bin/spa"
	#op_list = ["clang_O"+str(i) for i in range(4)]
	op_list = os.listdir(op_path)
	#op_list = ["clang_O3"]
	#op_list = ['util-linux']
	for op in op_list:
		print(op)
		bin_path = op_path + op
		bin_list = os.listdir(bin_path)
		'''
		for bina in bin_list:
			#if('.bc' in bina):
				#continue
			#cmd_get_bc = "extract-bc " + bin_path +"/"+ bina
			cmd_strip = "strip -g "+ bin_path +"/"+ bina
			print(cmd_strip)
			p = os.popen(cmd_strip,"r")
			print(p.read())
			p.close()
		'''
		#make_dir(bc_path+op)
		#make_dir(temp_path+op)
		#cmd_batch = "find " + bin_path +"/ -name \"*.bc\" | xargs -I file  mv file "+ bc_path+op +"/"
		#print(cmd_batch)
		#p=os.popen(cmd_batch,"r")
		#print(p.read())
		#p.close()
		
		bc_list  = os.listdir(bc_path + op)
		print(bc_list)
		for bc in bin_list:
			print(bc)
			bbc = bc+".bc"
			#bc= bbc.replace(".bc","")
			#make_dir(temp_path+op+'/'+bc)
			os.makedirs(temp_path+op+'/'+bc, exist_ok=True)
			if(os.path.exists(temp_path+op+"/"+bc+"/SVFOutput.data")==False):
				start = time.time()
				cmd = ""+plugin_path + " " + bc_path + op +'/'+bbc
				print(cmd)
				p=os.popen(cmd,"r")
				print(p.read())
				p.close()
				end = time.time()
				print('processing time = {time}s'.format(time=end-start))
				cmd_mv = "mv SVFOutput.data "+temp_path+op+"/"+bc
				print(cmd_mv)
				p=os.popen(cmd_mv,"r")
				print(p.read())
				p.close()
		

if __name__ == "__main__":
	out = open('svf.log','w')
	sys.stdout = out
	batch_svf()
	out.close()
