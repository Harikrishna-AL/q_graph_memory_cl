import timm
print("Loading CLIP...")
m1 = timm.create_model('resnet50_clip.openai', pretrained=True, num_classes=0)
print("CLIP Loaded!", m1.default_cfg if hasattr(m1, 'default_cfg') else "No cfg")

print("Loading SimCLR...")
try:
    m2 = timm.create_model('hf_hub:lightly-ai/simclrv1-imagenet1k-resnet50-1x', pretrained=True, num_classes=0)
    print("SimCLR Loaded via TIMM HF Hub!")
except Exception as e:
    print("Failed via TIMM:", e)

