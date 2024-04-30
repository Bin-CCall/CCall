# Preprocess
1. Use **gclang** to compile code, use **extract-bc** to get bitcode, and put it in **bitcode**
2. Put stripped binaries and unstripped binaries in **binary** and **binary_dbg**
3. Run *1_batch_svf.py* to get source-level indirect call targtes
4. Run *2_handle_indcall.py* to handle indirect call targets
5. Run *3_batch_ida.py* to get binary-level information from stripped binaries
6. Run *4_addr2line.py* to map source line to binary-level address
7. Run *5_combine.py* to get dataset
