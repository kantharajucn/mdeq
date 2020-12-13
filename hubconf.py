import torch
import os

dependencies = ['torch', 'yacs', 'scipy', 'termcolor']


def mdeq(pretrained=False, **kwargs):
    """
    What Makes for Good Views for Contrastive Learning?
    :param pretrained:
    :param kwargs:
    :return:
    """
    from lib.models.mdeq import MDEQClsNet
    from lib.config import config
    config.defrost()
    print(os.getcwd())
    config.merge_from_file(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments/imagenet/cls_mdeq_SMALL.yaml'))

    model = MDEQClsNet(config, **kwargs)

    if pretrained:
        url = 'https://github.com/kantharajucn/mdeq/releases/download/v1.0/MDEQ_Small_Cls.pkl'
        device = torch.device("cpu")
        torch.hub.download_url_to_file(url, dst=os.path.join(os.path.dirname(os.path.realpath(__file__)),"pretrained_models/MDEQ_Small_Cls.pkl"))
        state_dict = torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"pretrained_models/MDEQ_Small_Cls.pkl"), map_location=device)
        model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    mdeq(True)