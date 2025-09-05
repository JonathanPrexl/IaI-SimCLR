import numpy as np
import torch
import torch.nn as nn

class NT_Xent_SingGPU(nn.Module):

    def __init__(self, batch_size, temperature):
        super(NT_Xent_SingGPU, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):

        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size

        # z_i and z_j shape: (BS,512)
        # --> z.shape = (2*BS,512)
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()

        # shape here: (2*BS,2*BS-2)
        correct_order = positive_samples > negative_samples
        correct_order_sum = correct_order.sum()/N
        correct_order_sum = correct_order_sum/(2*self.batch_size)

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss, correct_order_sum

if __name__ == "__main__":

    l = NT_Xent_SingGPU(128,0.5)
    x1 = torch.rand(128,512)
    x2 = torch.rand(128,512)
    l(x1,x1)