import numpy as np
import pytest
import torch

from ml.training.model import Detector


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Detector("efficientnet_b0", pretrained=False).eval()


def test_output_shapes_and_feature_map(model):
    x = torch.randn(2, 3, 224, 224)
    logit, fmap = model.forward_with_features(x)
    assert logit.shape == (2, 1) and fmap.shape == (2, 1280, 7, 7)
    assert torch.equal(model(x), logit)


def test_head_is_exactly_pool_then_linear(model):
    x = torch.randn(3, 3, 224, 224)
    logit, fmap = model.forward_with_features(x)
    manual = fmap.mean(dim=(2, 3)) @ model.fc.weight.T + model.fc.bias
    assert torch.allclose(logit, manual, atol=1e-5)


def test_cam_from_feature_map_equals_autograd_gradcam(model):
    """The serving layer's CAM (weights x feature map) must equal textbook Grad-CAM before ReLU."""
    x = torch.randn(1, 3, 224, 224, requires_grad=False)
    logit, fmap = model.forward_with_features(x)
    fmap.retain_grad()
    logit.sum().backward()
    gradcam = (fmap.grad.mean(dim=(2, 3), keepdim=True) * fmap).sum(dim=1)[0].detach()  # alpha_c = mean gradient
    cam = torch.einsum("c,chw->hw", model.fc.weight[0], fmap[0]).detach()
    # d(logit)/dA_c = w_c / (H*W): the two maps differ only by that constant factor.
    assert torch.allclose(gradcam * 49, cam, rtol=1e-3, atol=1e-4)


def test_eval_mode_is_deterministic_and_train_mode_uses_dropout(model):
    x = torch.randn(2, 3, 224, 224)
    assert torch.equal(model(x), model(x))
    m = Detector("efficientnet_b0", pretrained=False, dropout=0.9).train()
    assert not torch.equal(m(x), m(x))


def test_freezing_controls_trainable_parameters():
    m = Detector("efficientnet_b0", pretrained=False)
    total = sum(p.numel() for p in m.backbone.parameters())
    assert m.set_backbone_trainable(False) == 0
    assert not any(p.requires_grad for p in m.backbone.parameters()) and all(p.requires_grad for p in m.fc.parameters())
    assert m.set_backbone_trainable(True) == total
    partial = m.set_backbone_trainable(10)
    assert 0 < partial < total and sum(p.requires_grad for p in m.backbone.parameters()) == 10
    last = list(m.backbone.parameters())[-1]
    assert last.requires_grad and not next(iter(m.backbone.parameters())).requires_grad


def test_frozen_batchnorm_stats_do_not_update_in_train_mode():
    m = Detector("efficientnet_b0", pretrained=False)
    m.set_backbone_trainable(False)
    m.train()
    bn = next(x for x in m.backbone.modules() if isinstance(x, torch.nn.BatchNorm2d))
    before = bn.running_mean.clone()
    m(torch.randn(4, 3, 224, 224))
    assert torch.equal(before, bn.running_mean)
    m.set_backbone_trainable(True)
    m.train()
    m(torch.randn(4, 3, 224, 224))
    assert not torch.equal(before, bn.running_mean)


def test_pretrained_weights_load_and_differ_from_random_init():
    pre = Detector("efficientnet_b0", pretrained=True)
    rnd = Detector("efficientnet_b0", pretrained=False)
    w_pre = next(pre.backbone.parameters())
    w_rnd = next(rnd.backbone.parameters())
    assert not torch.allclose(w_pre, w_rnd)
    assert np.isfinite(w_pre.detach().numpy()).all()


@pytest.mark.parametrize("name", ["mobilenetv3_large_100", "resnet50"])
def test_other_candidates_follow_the_same_contract(name):
    m = Detector(name, pretrained=False).eval()
    logit, fmap = m.forward_with_features(torch.randn(1, 3, 224, 224))
    assert logit.shape == (1, 1) and fmap.shape[0] == 1 and fmap.shape[1] == m.fc.in_features


def test_construction_leaves_pretrained_batchnorm_statistics_untouched():
    import timm

    from ml.training.model import CANDIDATES

    raw = timm.create_model(CANDIDATES["efficientnet_b0"], pretrained=True, num_classes=0, global_pool="")
    ours = Detector("efficientnet_b0", pretrained=True)
    raw_bn = next(m for m in raw.modules() if isinstance(m, torch.nn.BatchNorm2d))
    our_bn = next(m for m in ours.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    assert torch.equal(raw_bn.running_mean, our_bn.running_mean) and torch.equal(raw_bn.running_var, our_bn.running_var)
