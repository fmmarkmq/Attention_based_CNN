import torch
import torch.nn.functional as F
import torchvision
from utils.tools import dotdict
from sklearn.metrics import accuracy_score
from autoattack import AutoAttack
import foolbox as fb
from attack.fb_PyTorchModel_revised import PyTorchModel
from attack.autoattack_model import Model_for_Autoattack


class Attack(object):
    def __init__(self, args: dotdict, bounds, preprocessing, data_loader, device=None):
        super(Attack, self).__init__()
        self.args = args
        self.bounds = bounds
        self.preprocessing = preprocessing
        self.data_loader = data_loader
        self.device = device

    def __call__(self, attacker_name: str, model) -> float:
        model.eval()
        if attacker_name in ['fgsm', 'pgd', 'deepfool']:
            return self.fb_attack(attacker_name, model)
        elif attacker_name in ['apgd-ce']:
            return self.auto_attack(attacker_name, model)

    
    def fb_attack(self, attacker_name: str, model):
        fmodel = PyTorchModel(model, bounds=self.bounds, preprocessing=self.preprocessing, device=self.device)
        if attacker_name == 'fgsm':
            epsilon, = self.args.attack['fgsm']
            attacker = fb.attacks.FGSM()
        elif attacker_name == 'pgd':
            epsilon, alpha, num_iter = self.args.attack['pgd']
            attacker =  fb.attacks.LinfPGD(rel_stepsize=alpha, steps=num_iter)
        elif attacker_name == 'deepfool':
            epsilon, stepsize, num_iter = self.args.attack['deepfool']
            attacker = fb.attacks.LinfDeepFoolAttack(overshoot=stepsize, steps=num_iter)

        suc_all = torch.empty((0), device=self.device)
        for _, (images, labels) in enumerate(self.data_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            _, _, is_adv = attacker(fmodel, images, labels, epsilons=epsilon)
            suc_all = torch.concat([suc_all, is_adv])
        attack_accuracy = 1 - suc_all.mean()
        return float(attack_accuracy)
    
    def auto_attack(self, attacker_name: str, model):
        if attacker_name == 'apgd-ce':
            epsilon, = self.args.attack['apgd-ce']
            transform = torchvision.transforms.Normalize(mean=self.preprocessing['mean'], std=self.preprocessing['std'])
            amodel = Model_for_Autoattack(model, transform)
            adversary = AutoAttack(amodel, norm='Linf', eps=epsilon, version='standard')
            adversary.attacks_to_run = ['apgd-ce']

        images = self.data_loader.dataset.data
        images_shape = images.shape
        images = images.reshape(images_shape[0], -1, images_shape[-2],images_shape[-1])
        images = (images-images.min())/images.max()
        labels = self.data_loader.dataset.targets

        _, y_adv = adversary.run_standard_evaluation(images, labels, bs=100, return_labels=True)
        return accuracy_score(labels, y_adv)


    # def fgsm(self, model, X, y):
    #     epsilon, = self.args.attack['fgsm']
    #     delta = torch.zeros_like(X, requires_grad=True)
    #     loss = torch.nn.CrossEntropyLoss()(model(X + delta), y)
    #     loss.backward()
    #     return epsilon * delta.grad.detach().sign()

    # def pgd(self, model, X, y):
    #     epsilon,alpha,num_iter = self.args.attack['pgd']

    #     delta = torch.zeros_like(X, requires_grad=True)
    #     for _ in range(num_iter):
    #         loss = torch.nn.CrossEntropyLoss()(model(X + delta), y)
    #         loss.backward()
    #         delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
    #         delta.grad.zero_()
    #         fb.attacks.LinfPGD()
    #     return delta.detach()
        
