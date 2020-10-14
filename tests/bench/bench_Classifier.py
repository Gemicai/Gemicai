from torchvision import models
import gemicai as gem
from datetime import datetime
import torch
import numpy as np


def bench_classify():
    relevant_labels = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
    resnet18 = models.resnet18(pretrained=True)
    ds = gem.DicomoDataset.get_dicomo_dataset('examples', labels=relevant_labels)
    net = gem.Classifier(resnet18, ds.classes('BodyPartExamined'), enable_cuda=False)

    ds_iter = iter(ds)
    tensors = torch.cat((torch.unsqueeze(next(ds_iter)[0], 0), torch.unsqueeze(next(ds_iter)[0], 0)))
    cls = net.classify(tensors)




