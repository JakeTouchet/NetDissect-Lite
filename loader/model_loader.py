import settings
import torch
import torchvision
import timm

def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        if settings.MODEL != "resnet50":
            checkpoint = torch.load(settings.MODEL_FILE)
            if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
                if settings.MODEL_PARALLEL:
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                        'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
            else:
                model = checkpoint
        else:
            # TODO - don't use parallel, at least for test, don't know if can replace
            model = timm.models.create_model("resnet50", checkpoint_path=settings.MODEL_FILE)
            # Replace with gelu
            def replace_layers(model, old, new):
                for n, module in model.named_children():
                    if len(list(module.children())) > 0:
                        ## compound module, go inside it
                        replace_layers(module, old, new)
                    if isinstance(module, old):
                        ## simple module
                        setattr(model, n, new)
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
