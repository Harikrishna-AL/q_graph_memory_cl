import torch
import timm

def test_siglip():
    model = timm.create_model('vit_so400m_patch14_siglip_384.webli', pretrained=False, num_classes=0)
    data_config = timm.data.resolve_data_config(model.default_cfg)
    print("Data Config:", data_config)
    
    dummy_img = torch.randn(2, 3, 384, 384)
    out = model(dummy_img)
    print("Output Shape:", out.shape)

if __name__ == "__main__":
    test_siglip()
