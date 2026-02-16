import pickle

class NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Fix for NumPy >=2.0 pickles loaded in NumPy <2.0
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        elif module == "numpy.core.multiarray" or module == "numpy.multiarray":
            module = "numpy.core.multiarray"
        return super().find_class(module, name)

buffer_file_path = '/home/metamobility2/Changseob/online_adaptation_GPE/Controller_Online_Adaptation_GPE_indoor/buffer_2.pkl'

with open(buffer_file_path, "rb") as f:
    buffer_data = NumpyCompatUnpickler(f).load()

print(buffer_data['bin_state'][-5][0.9]['L'][0])
print(buffer_data['bin_state'][0][0.9]['L'][0])
print(buffer_data['bin_state'][5][0.9]['L'][0])