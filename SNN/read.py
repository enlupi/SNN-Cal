
from dataset import readfile
import os
import glob

archivo1 = os.path.join("merged_data_kaon_pruebas", "merged_kaon_pruebas_2.dat")
#archivo1 = os.path.join("merged_data","merged_kaon.dat")
ph_list, E_list, ct_list , sE_list, N_list, p_class, primary_list = readfile(archivo1, False)


with open("resultados_merged_2.txt", "w") as file:
    for E, sE, N, clase in zip(E_list, sE_list, N_list, p_class):
        file.write(f"{E}\t{sE}\t{N}\t{clase}\n")

'''
pattern = os.path.join("kaon_pruebas", "kaon_1_*.dat")
files = sorted(glob.glob(pattern))
print(len(files))


for i in files:
    ph_list2, E_list2, ct_list, sE_list2, N_list2, clase2, primary_list = readfile(i, False)
    out_name = f"resultados_buenos_{os.path.basename(i)}.txt"
    with open(out_name, "w") as file2:
        for E, sE, N, clase in zip(E_list2, sE_list2, N_list2, clase2):
            file2.write(f"{E}\t{sE}\t{N}\t{clase}\n")  
'''
