import pickle
import os

def readData(  file_path):
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    data2 = pickle.loads(bytes_in)
    return data2

def saveData( data,file_path):
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    # data = bytearray(n_bytes)
    bytes_out = pickle.dumps(data)
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])