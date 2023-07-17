
import snntorch as snn
from snntorch import surrogate, functional, utils
import torch
from dataset_and_model.mnist import forward_pass

acc_dict = {'rate': functional.accuracy_rate, 
            'temporal': functional.accuracy_temporal}
loss_dict = {
    'mse': functional.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2) ,
    'cross': functional.ce_count_loss() 
}
def get_acc_function(name):
    return acc_dict[name]

def get_loss_function(name):
    return loss_dict[name]

class SNN_ATTACK():
    def __init__(self, cfg, model) -> None:
        
        general_setup = cfg.parameters.general_setup
        self.num_pic = general_setup.num_pic
        
        algorithm_setup = cfg.parameters.algorithm
        self.num_samples = algorithm_setup.num_samples
        self.lamda = algorithm_setup.lamda
        self.loss_function= algorithm_setup.loss_fn
        self.is_target = algorithm_setup.is_target
        self.spd = algorithm_setup.samples_per_draw
        
        
        optimize_setup = cfg.parameters.optimizer 
        self.lr = optimize_setup.learning_rate
        
        
        data_setup = cfg.dataset_and_model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
       

    def run(self, data, model):
        for i, (image, label) in enumerate(iter(data)):
            if i == self.num_pic:
                return 
            print("No.",i,"/", self.numpic)
            
            self.attack(model, image, label)
            

            
    def attack(self, model, image, label):
        
        for i in range(self.num_samples):
            
            gradient_list = []
            for i in range(self.samples_per_draw):
                output = forward_pass(model, image)
                acc = get_acc_function('rate')(output, label)
                if acc > 0:
                    return image
                loss = get_loss_function(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                gradient_list.append(image.grad)

            gradient_mean = torch.mean(gradient_list)
            image = image - self.lr * gradient_mean
            