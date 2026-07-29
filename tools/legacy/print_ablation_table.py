import json
import glob
import os

def generate_full_ablation_table():
    files = glob.glob('results/ablation_*.json')
    
    # Organize data: dict[dataset][backbone][component] = accuracy
    data = {}
    
    for f in files:
        basename = os.path.basename(f)
        # Format is ablation_{backbone}_{dataset}.json
        # e.g., ablation_dinov3_imagenet_r.json
        parts = basename.replace('ablation_', '').replace('.json', '').split('_')
        
        if 'resnet50' in parts:
            backbone = 'ResNet50'
            dataset = basename.replace('ablation_resnet50_', '').replace('.json', '')
        elif 'dinov3' in parts:
            backbone = 'DINOv3'
            dataset = basename.replace('ablation_dinov3_', '').replace('.json', '')
        elif 'siglip2' in parts:
            backbone = 'SigLIP2'
            dataset = basename.replace('ablation_siglip2_', '').replace('.json', '')
        else:
            continue
            
        dataset = dataset.replace('_', '-').title().replace('-R', '-R').replace('Tinyimagenet', 'TinyImageNet').replace('Objectnet', 'ObjectNet')
        
        with open(f, 'r') as fp:
            try:
                j = json.load(fp)
            except:
                continue
                
        if dataset not in data:
            data[dataset] = {}
        if backbone not in data[dataset]:
            data[dataset][backbone] = {}
            
        for comp in j.get('component_ablation', []):
            step = comp['step']
            aia = comp['aia'] * 100
            if step == "Standard NCM":
                data[dataset][backbone]['NCM'] = f"{aia:.1f}%"
            elif step == "+Analytic ETF":
                data[dataset][backbone]['+ETF'] = f"{aia:.1f}%"
            elif step == "Full MAYA (Hybrid)":
                data[dataset][backbone]['Full MAYA'] = f"{aia:.1f}%"
                
    print("| Dataset | Backbone | Standard NCM | + Analytic ETF | Full MAYA (ETF + Graph) |")
    print("| :--- | :--- | :---: | :---: | :---: |")
    
    for dataset in sorted(data.keys()):
        for backbone in ['ResNet50', 'DINOv3', 'SigLIP2']:
            if backbone in data[dataset]:
                ncm = data[dataset][backbone].get('NCM', '-')
                etf = data[dataset][backbone].get('+ETF', '-')
                maya = data[dataset][backbone].get('Full MAYA', '-')
                print(f"| **{dataset}** | {backbone} | {ncm} | {etf} | **{maya}** |")

if __name__ == "__main__":
    generate_full_ablation_table()
