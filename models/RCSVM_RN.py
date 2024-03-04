import torch


def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(torch.sum(tensor1 - tensor2) ** 2)

class RCSVM_RN:
    def __init__(self, data, positives, unlabeled, alpha, beta):
        self.data = data
        self.positives = positives
        self.unlabeled = unlabeled
        self.alpha = alpha
        self.beta = beta


    # def train(self):
    #     positive_mask = torch.zeros(len(self.data))
    #     for i in self.positives:
    #         positive_mask[i] = 1
    #     positive_mask = torch.tensor(positive_mask, dtype = torch.bool)
    #     unlabeled_mask = torch.tensor([not x for x in positive_mask], dtype = torch.bool)

    #     self.c_positive = self.alpha * 1/len(self.positives) * torch.mean(self.data[positive_mask]) - self.beta * 1/len(self.unlabeled) * torch.mean(self.data[unlabeled_mask])

    #     self.c_negative = self.alpha * 1/len(self.unlabeled) * torch.mean(self.data[unlabeled_mask]) - self.beta * 1/len(self.positives) * torch.mean(self.data[positive_mask])



    def train(self):
        soma_positive = 0
        for element in self.positives:
            soma_positive += self.data[element] / torch.norm(self.data[element], p = 2)
            
        soma_unlabeled = 0
        for element in self.unlabeled:
            soma_unlabeled += self.data[element] / torch.norm(self.data[element], p = 2)

        self.c_positive = (self.alpha / len(self.positives)) * soma_positive - (self.beta / len(self.unlabeled)) * soma_unlabeled 
        self.c_negative = (self.alpha/ len(self.unlabeled)) *  soma_unlabeled - (self.beta / len(self.positives)) * soma_positive



    def negative_inference(self, num_neg = None, similarity = torch.dist):
        RN = list()
        for element in self.unlabeled:
            if similarity(self.c_positive, self.data[element]) <= similarity(self.c_negative,self.data[element]):
                RN.append(element)
        
        if not num_neg:
            return torch.tensor(RN)
        else:
            return torch.tensor(RN[:num_neg])
    