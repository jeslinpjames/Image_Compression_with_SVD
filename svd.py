import numpy as np

class SVD:
    def __init__(self, matrix):
        self.matrix = matrix   
        self.U, self.S, self.Vt = np.linalg.svd(matrix, full_matrices=False)
        
    def get_matrix(self):
        return self.matrix
    
    def get_U(self):
        return self.U
    
    def get_S(self):
        return self.S
    
    def get_Vt(self):
        return self.Vt
    
    def reconstruct_matrix(self, rank=None):
        if rank is None:
            rank = len(self.S)

        reconstructed_matrix = np.dot(self.U[:, :rank], np.dot(np.diag(self.S[:rank]), self.Vt[:rank, :]))

        return reconstructed_matrix
    