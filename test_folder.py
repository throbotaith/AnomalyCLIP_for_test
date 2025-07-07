import argparse
import os
from datetime import datetime

# heavy dependencies (torch, cv2, etc.) are imported lazily in main



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def visualizer(path, anomaly_map, img_size, output_dir, threshold=0.5):
    filename = os.path.basename(path)
    dirname = os.path.dirname(path)

    # read the original image without resizing so output matches input size
    vis = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = vis.shape[:2]

    # resize anomaly map to the original resolution before overlaying
    mask = normalize(anomaly_map[0])
    mask = cv2.resize(mask, (orig_w, orig_h))
    mask[mask < threshold] = 0

    vis = apply_ad_scoremap(vis, mask)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    save_vis = os.path.join(output_dir, f'anomaly_map_{filename}')
    print(save_vis)
    cv2.imwrite(save_vis, vis)


def load_model(args, device):
    AnomalyCLIP_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=AnomalyCLIP_parameters)
    model.eval()
    preprocess, _ = get_transform(args)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), AnomalyCLIP_parameters)
    checkpoint = torch.load(args.checkpoint_path)
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return model, text_features, preprocess


def process_image(model, text_features, preprocess, image_path, args, device, output_dir):
    img = Image.open(image_path)
    img = preprocess(img)
    image = img.reshape(1, 3, args.image_size, args.image_size).to(device)
    with torch.no_grad():
        image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_probs = image_features @ text_features.permute(0, 2, 1)
        text_probs = (text_probs / 0.07).softmax(-1)
        text_probs = text_probs[:, 0, 1]
        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            if idx >= args.feature_map_layer[0]:
                patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
                similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
                anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
                anomaly_map_list.append(anomaly_map)
        anomaly_map = torch.stack(anomaly_map_list)
        anomaly_map = anomaly_map.sum(dim=0)
        anomaly_map = torch.stack([
            torch.from_numpy(gaussian_filter(i, sigma=args.sigma))
            for i in anomaly_map.detach().cpu()
        ], dim=0)
    visualizer(image_path, anomaly_map.detach().cpu().numpy(), args.image_size, output_dir, args.threshold)


def test_folder(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, text_features, preprocess = load_model(args, device)
    img_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    for fname in sorted(os.listdir(args.folder_path)):
        if not fname.lower().endswith(img_extensions):
            continue
        image_path = os.path.join(args.folder_path, fname)
        process_image(model, text_features, preprocess, image_path, args, device, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("AnomalyCLIP folder test", add_help=True)
    parser.add_argument("--folder_path", type=str, required=True, help="folder containing test images")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--depth", type=int, default=9, help="model depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="context length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="text context length")
    parser.add_argument("--feature_map_layer", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--sigma", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5, help="heatmap detection threshold")
    args = parser.parse_args()

    # create output directory based on timestamp and checkpoint name
    model_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = "anomaly_detection_results"
    os.makedirs(result_root, exist_ok=True)
    args.output_dir = os.path.join(result_root, f"{timestamp}_{model_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "inference_settings.txt"), "w") as f:
        f.write(f"timestamp: {timestamp}\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")

    # heavy imports after argument parsing so `--help` works without dependencies
    import torch
    import random
    import numpy as np
    import cv2
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from utils import get_transform, normalize
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner

    globals().update(locals())  # make imported modules available to functions

    setup_seed(args.seed)
    test_folder(args)
