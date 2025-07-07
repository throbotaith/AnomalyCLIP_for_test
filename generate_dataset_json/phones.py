import os
import json

class PhonesSolver(object):
    CLSNAMES = ['phone']

    def __init__(self, root='datasets_phones'):
        self.root = root
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['train', 'test']:
                cls_info = []
                phase_dir = f'{cls_dir}/{phase}'
                if not os.path.exists(phase_dir):
                    continue
                species = os.listdir(phase_dir)
                for specie in species:
                    is_abnormal = True if specie not in ['good'] else False
                    img_dir = f'{cls_dir}/{phase}/{specie}'
                    if not os.path.isdir(img_dir):
                        continue
                    img_names = os.listdir(img_dir)
                    mask_dir = f'{cls_dir}/ground_truth/{specie}' if is_abnormal else None
                    mask_names = os.listdir(mask_dir) if mask_dir and os.path.isdir(mask_dir) else None
                    img_names.sort()
                    if mask_names is not None:
                        mask_names.sort()
                    for idx, img_name in enumerate(img_names):
                        mask_path = ''
                        if mask_names is not None and idx < len(mask_names):
                            mask_path = f'{cls_name}/ground_truth/{specie}/{mask_names[idx]}'
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path=mask_path,
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples += 1
                            else:
                                normal_samples += 1
                info[phase][cls_name] = cls_info
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + '\n')
        print('normal_samples', normal_samples, 'anomaly_samples', anomaly_samples)

if __name__ == '__main__':
    runner = PhonesSolver(root='datasets_phones')
    runner.run()
