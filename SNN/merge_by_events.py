import os
import glob
import re

'''def extract_numbers(filename):
    """
    Función auxiliar para extraer los números de un nombre de archivo.
    """
    # Busca todos los grupos de dígitos en el nombre del archivo
    s = re.findall(r'\d+', filename)
    # Devuelve una lista de enteros para que Python pueda comparar numéricamente
    return [int(x) for x in s] if s else []'''


def merge_by_events(input_path, num_events, particle_name):
    """
    Agrupa archivos .dat individuales en archivos fusionados de 'num_events' cada uno.
    """

    # Carpeta de salida 
    output_dir = f"merged_data_{particle_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 2. Buscar archivos 
    pattern = os.path.join(f"{input_path}", f"{particle_name}_*.dat")
    files = glob.glob(pattern)
    
    total_files = len(files)
    if not files:
        print(f"AVISO: No se encontraron archivos en la ruta {pattern}")
        return

    print(f"Total archivos encontrados: {total_files}")

    # 3. Procesamiento por bloques 
    counter = 1
    delimiter = b'EOE '

    for i in range(0, total_files, num_events):
        divided_files = files[i : i + num_events]
        
        # Nombre del archivo de salida
        output_filename = os.path.join(output_dir, f"merged_{particle_name}_{counter}.dat")
        
        try:
            with open(output_filename, 'wb') as outfile:
                event = 1
                for filepath in divided_files:
                    with open(filepath, 'rb') as infile:
                        data = infile.read()
                        outfile.write(data)
                        outfile.write(delimiter)
                        outfile.write(struct.pack('i', event))
                        event +=1
            counter += 1
            
        except Exception as e:
            print(f"  -> [ERROR] Fallo al crear {output_filename}: {e}")

    print(f"--- Proceso finalizado. Se crearon {counter - 1} archivos fusionados. ---")



if __name__ == "__main__":
    full_path_to_data = r"/lhome/ext/uovi123/uovi123l/SNN-Cal/kaon_pruebas" 
    
    events_per_file = 5
    
    merge_by_events(full_path_to_data, events_per_file, "kaon")



