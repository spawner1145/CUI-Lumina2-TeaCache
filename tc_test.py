import torch
import numpy as np
import time
import json
import os
from unittest.mock import patch
import folder_paths

try: # lpips分析图像差异距离
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try: # 贝叶斯
    from skopt import Optimizer
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

# 全局状态管理
class TeaCacheStateManager:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TeaCacheStateManager, cls).__new__(cls)
            cls._instance.runs = {}
            cls._instance.lpips_model = None
            cls._instance.baseline_image = None
        return cls._instance

    def register_run(self, run_id, params):
        self.runs[run_id] = {"start_time": time.time(), "params": params, "results": {}, "cache": {}, "cnt": 0}

    def get_run_data(self, run_id):
        return self.runs.get(run_id)

    def cleanup_run(self, run_id):
        if run_id in self.runs:
            del self.runs[run_id]

    def set_lpips_model(self, model):
        self.lpips_model = model

    def get_lpips_model(self):
        return self.lpips_model

    def set_baseline_image(self, image_tensor):
        self.baseline_image = image_tensor

    def get_baseline_image(self):
        return self.baseline_image

STATE_MANAGER = TeaCacheStateManager()

# 缓存逻辑
def teacache_forward_analysis(self, x, timesteps, context, num_tokens, **kwargs):
    transformer_options = kwargs.get('transformer_options', {})
    run_id = transformer_options.get('teacache_run_id')
    if not run_id:
        if hasattr(self, "_forward_original"):
            return self._forward_original(x, timesteps, context, num_tokens, **kwargs)
        return self.forward_original(x, timesteps, context, num_tokens, **kwargs)
    
    run_data = STATE_MANAGER.get_run_data(run_id)
    if not run_data:
        if hasattr(self, "_forward_original"):
            return self._forward_original(x, timesteps, context, num_tokens, **kwargs)
        return self.forward_original(x, timesteps, context, num_tokens, **kwargs)

    params = run_data['params']
    
    if run_data.get("num_steps") is None and transformer_options.get("num_steps") is not None:
        run_data["num_steps"] = transformer_options.get("num_steps")

    cap_feats, cap_mask = context, kwargs.get('attention_mask')
    ref_latents = kwargs.get("ref_latents", [])
    ref_contexts = kwargs.get("ref_contexts", [])
    siglip_feats = kwargs.get("siglip_feats", [])
    bs, c_channels, h_img, w_img = x.shape
    if hasattr(self, 'pad_to_patch_size'): x = self.pad_to_patch_size(x, (self.patch_size, self.patch_size))
    else: 
        ph = (self.patch_size - h_img % self.patch_size) % self.patch_size
        pw = (self.patch_size - w_img % self.patch_size) % self.patch_size
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    t = (1.0 - timesteps).to(dtype=x.dtype)
    t_in = t * getattr(self, "time_scale", 1.0)
    t_emb = self.t_embedder(t_in, dtype=x.dtype)
    adaln_input = t_emb
    if getattr(self, "clip_text_pooled_proj", None) is not None:
        pooled = kwargs.get("clip_text_pooled", None)
        if pooled is not None:
            pooled = self.clip_text_pooled_proj(pooled)
        else:
            clip_text_dim = getattr(self, "clip_text_dim", None)
            if clip_text_dim is None:
                clip_text_dim = t_emb.shape[-1]
            pooled = torch.zeros((x.shape[0], clip_text_dim), device=x.device, dtype=x.dtype)
        adaln_input = self.time_text_embed(torch.cat((t_emb, pooled), dim=-1))
    try:
        patchify_out = self.patchify_and_embed(
            x,
            cap_feats,
            cap_mask,
            adaln_input,
            num_tokens,
            ref_latents=ref_latents,
            ref_contexts=ref_contexts,
            siglip_feats=siglip_feats,
            transformer_options=transformer_options,
        )
    except TypeError:
        patchify_out = self.patchify_and_embed(x, cap_feats, cap_mask, adaln_input, num_tokens)

    if len(patchify_out) == 6:
        x, mask, img_size, cap_size, freqs_cis, timestep_zero_index = patchify_out
    else:
        x, mask, img_size, cap_size, freqs_cis = patchify_out
        timestep_zero_index = None
    freqs_cis = freqs_cis.to(x.device)
    max_seq_len = x.shape[1]
    
    should_calc = True
    num_steps_in_state = run_data.get("num_steps")
    cnt = run_data.get('cnt', 0)
        
    if num_steps_in_state is None or num_steps_in_state == 0 or cnt == 0 or cnt == num_steps_in_state - 1:
        should_calc = True
    else:
        run_data.setdefault('cache', {})
        current_cache = run_data['cache'].setdefault(max_seq_len, {"accumulated_rel_l1_distance": 0.0, "previous_modulated_input": None, "previous_residual": None})
        if current_cache.get("previous_modulated_input") is not None:
            mod_result = self.layers[0].adaLN_modulation(adaln_input.clone())
            if isinstance(mod_result, (list, tuple)) and len(mod_result) > 0:
                modulated_inp = mod_result[0]
            elif torch.is_tensor(mod_result):
                modulated_inp = mod_result
            else:
                raise ValueError("adaLN_modulation returned unexpected type or empty list/tuple")
            coefficients = params.get("coefficients_to_use", [])
            rescale_func = np.poly1d(coefficients)
            prev_mod_input = current_cache["previous_modulated_input"]
            prev_mean = prev_mod_input.abs().mean().item()
            rel_l1_change = ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).cpu().item() if prev_mean > 1e-9 else float('inf')
            current_cache["accumulated_rel_l1_distance"] += rescale_func(rel_l1_change)
            
            if current_cache["accumulated_rel_l1_distance"] < params.get('rel_l1_thresh', 0.3): should_calc = False
            else:
                should_calc = True
                current_cache["accumulated_rel_l1_distance"] = 0.0
            current_cache["previous_modulated_input"] = modulated_inp.clone()
        else: should_calc = True
    
    if should_calc:
        if max_seq_len != run_data.get('uncond_seq_len'):
            run_data['results'].setdefault("total_inferences", 0)
            run_data['results']["total_inferences"] += 1
        original_x = x.clone()
        processed_x = x
        for layer in self.layers:
            try:
                processed_x = layer(processed_x, mask, freqs_cis, adaln_input, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)
            except TypeError:
                processed_x = layer(processed_x, mask, freqs_cis, adaln_input)
        run_data.setdefault('cache', {})
        current_cache = run_data['cache'].setdefault(max_seq_len, {})
        current_cache["previous_residual"] = processed_x - original_x
        current_cache["accumulated_rel_l1_distance"] = 0.0
        if current_cache.get("previous_modulated_input") is None:
            mod_result = self.layers[0].adaLN_modulation(adaln_input.clone())
            if isinstance(mod_result, (list, tuple)) and len(mod_result) > 0:
                current_cache["previous_modulated_input"] = mod_result[0]
            elif torch.is_tensor(mod_result):
                current_cache["previous_modulated_input"] = mod_result
            else:
                raise ValueError("adaLN_modulation returned unexpected type or empty list/tuple")
    else:
        current_cache = run_data['cache'].get(max_seq_len, {})
        processed_x = x + current_cache.get("previous_residual", 0)
        if max_seq_len != run_data.get('uncond_seq_len'):
            run_data['results'].setdefault("cache_hits", 0)
            run_data['results']["cache_hits"] += 1

    if num_steps_in_state is not None and max_seq_len != run_data.get('uncond_seq_len'):
        run_data['cnt'] += 1

    try:
        output = self.final_layer(processed_x, adaln_input, timestep_zero_index=timestep_zero_index)
    except TypeError:
        output = self.final_layer(processed_x, adaln_input)
    if hasattr(self, 'unpatchify'): output = self.unpatchify(output, img_size, cap_size, return_tensor=True)[:, :, :h_img, :w_img]
    else: raise NotImplementedError("Model does not have an 'unpatchify' method.")
    return -output

# TeaCache Patcher
class TeaCache_Patcher:
    DEFAULT_COEFFS = [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344]
    
    @classmethod
    def INPUT_TYPES(cls):
        modes = ["手动输入", "自动微调", "贝叶斯优化"] if SKOPT_AVAILABLE else ["手动输入", "自动微调"]
        metrics = ["速度-命中率权衡", "质量-命中率权衡 (LPIPS)"] if LPIPS_AVAILABLE else ["速度-命中率权衡"]
        
        optional_inputs = {
            "coefficients_str": ("STRING", {"multiline": True, "default": json.dumps(cls.DEFAULT_COEFFS)}),
        }
        if LPIPS_AVAILABLE:
            optional_inputs["max_lpips_thresh"] = ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.00001, "display": "number"})

        return {
            "required": {
                "model": ("MODEL",),
                "mode": (modes,),
                "evaluation_metric": (metrics,),
                "rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "step": 0.001}),
            },
            "optional": optional_inputs
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "run_id")
    FUNCTION = "patch_model"
    CATEGORY = "utils/analysis"
    EXPERIMENTAL = True
    
    def _load_history(self, file_path):
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f: return json.load(f)
        except Exception: pass
        return []

    def _get_score(self, run, baseline_time, metric, max_lpips_thresh):
        inferences = run.get("total_inferences", 0)
        if inferences == 0: return 0
        hit_ratio = run.get("cache_hits", 0) / inferences

        if "LPIPS" in metric:
            lpips_dist = run.get("lpips_distance")
            if lpips_dist is None: return 0
            
            if max_lpips_thresh > 0 and lpips_dist > max_lpips_thresh:
                return 0

            if lpips_dist < 1e-6: return 0
            
            # 公式：命中率 / LPIPS距离
            return hit_ratio / lpips_dist
        else: # "速度-命中率权衡"
            time_saved = baseline_time - run.get('generation_time', 0)
            if time_saved <= 0: return 0
            return time_saved * hit_ratio

    def _find_best_run(self, history, baseline_time, metric, max_lpips_thresh):
        eff_runs = [r for r in history if r.get("rel_l1_thresh") != 0]
        if not eff_runs: return None
        
        best_run, max_score = None, -1.0
        for run in eff_runs:
            score = self._get_score(run, baseline_time, metric, max_lpips_thresh)
            if score > max_score:
                max_score, best_run = score, run
        return best_run

    def patch_model(self, model, mode, evaluation_metric, rel_l1_thresh, coefficients_str=None, max_lpips_thresh=0.6):
        history = self._load_history(os.path.join(folder_paths.get_output_directory(), "teacache_analysis.json"))
        baseline_runs = [r for r in history if r.get("rel_l1_thresh") == 0]
        # 基准时间只在“速度”相关的评估中需要
        baseline_time = min([r['generation_time'] for r in baseline_runs]) if baseline_runs else None
        
        coeffs_to_use = []
        if mode == "贝叶斯优化":
            if not SKOPT_AVAILABLE: raise Exception("'贝叶斯优化' 模式需要 'scikit-optimize' 库。")
            
            space = [Real(-1000, 1000), Real(-1000, 1000), Real(-1000, 1000), Real(-200, 200), Real(-50, 50)]
            optimizer = Optimizer(dimensions=space, random_state=int(time.time()), acq_func="gp_hedge")
            
            if history:
                valid_history = [r for r in history if "coefficients" in r and r.get("rel_l1_thresh") != 0 and len(r["coefficients"]) == len(space)]
                if "LPIPS" in evaluation_metric:
                    valid_history = [r for r in valid_history if r.get("lpips_distance") is not None]
                elif "速度" in evaluation_metric and not baseline_time:
                    print("警告: 缺少基准时间，无法进行基于速度的贝叶斯优化。")
                    valid_history = []
                
                if valid_history:
                    y_iters = [-self._get_score(r, baseline_time, evaluation_metric, max_lpips_thresh) for r in valid_history]
                    optimizer.tell([r["coefficients"] for r in valid_history], y_iters)
            coeffs_to_use = optimizer.ask()

        elif mode == "自动微调":
            best_efficient_run = self._find_best_run(history, baseline_time, evaluation_metric, max_lpips_thresh)
            base_coeffs = best_efficient_run["coefficients"] if best_efficient_run else self.DEFAULT_COEFFS
            idx_to_tweak = len(history) % len(base_coeffs)
            perturb_factor = np.random.uniform(0.8, 1.2)
            coeffs_to_use = list(base_coeffs)
            coeffs_to_use[idx_to_tweak] *= perturb_factor
        else: # 手动输入
            coeffs_to_use = json.loads(coefficients_str) if coefficients_str else self.DEFAULT_COEFFS
        
        run_id = str(time.time_ns())
        params_for_run = { 
            "rel_l1_thresh": rel_l1_thresh, 
            "coefficients_to_use": coeffs_to_use,
            "max_lpips_thresh": max_lpips_thresh
        }
        STATE_MANAGER.register_run(run_id, params_for_run)
        
        new_model = model.clone()
        diffusion_model = new_model.get_model_object("diffusion_model")
        
        if hasattr(diffusion_model, "_forward"):
            if not hasattr(diffusion_model, "_forward_original"):
                diffusion_model._forward_original = diffusion_model._forward
            diffusion_model._forward = teacache_forward_analysis.__get__(diffusion_model, diffusion_model.__class__)
        else:
            if not hasattr(diffusion_model, 'forward_original'):
                diffusion_model.forward_original = diffusion_model.forward
            diffusion_model.forward = teacache_forward_analysis.__get__(diffusion_model, diffusion_model.__class__)

        old_wrapper = new_model.model_options.get("model_function_wrapper")

        def unet_wrapper_function(model_function, kwargs):
            c_dict = kwargs.get("c", {})
            c_dict.setdefault("transformer_options", {})
            c_dict["transformer_options"]["teacache_run_id"] = run_id
            if "sample_sigmas" in c_dict["transformer_options"]:
                c_dict["transformer_options"]["num_steps"] = len(c_dict["transformer_options"]["sample_sigmas"])
            if old_wrapper:
                return old_wrapper(model_function, {"input": kwargs["input"], "timestep": kwargs["timestep"], "c": c_dict, "cond_or_uncond": kwargs.get("cond_or_uncond")})
            return model_function(kwargs["input"], kwargs["timestep"], **c_dict)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        print(f"[TeaCache Patcher] 已准备好运行，ID: {run_id}, 评估模式: {evaluation_metric}")
        return (new_model, run_id)

# 结果收集
class TeaCache_Result_Collector:
    @classmethod
    def INPUT_TYPES(cls):
        return { "required": { "latent": ("LATENT",), "run_id": ("STRING", {"forceInput": True}), "analysis_file": ("STRING", {"default": "teacache_analysis.json"}), }, "optional": { "trigger": ("STRING", {"forceInput": True}) } }
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("LATENT", "status")
    FUNCTION = "collect_and_save"
    CATEGORY = "utils/analysis"
    EXPERIMENTAL = True
    def _load_history(self, file_path):
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f: return json.load(f)
        except Exception: pass
        return []
    def collect_and_save(self, latent, run_id, analysis_file, trigger=None):
        run_data = STATE_MANAGER.get_run_data(run_id)
        if not run_data: return (latent, f"错误: 未找到ID为 {run_id} 的运行记录。")
        params = run_data['params']
        results = run_data['results']
        generation_time = time.time() - run_data.get('start_time', time.time())
        final_run_data = {
            "timestamp": time.time(), "generation_time": generation_time,
            "cache_hits": results.get('cache_hits', 0), "total_inferences": results.get('total_inferences', 0),
            "rel_l1_thresh": params.get('rel_l1_thresh'), "coefficients": params.get('coefficients_to_use'),
            "lpips_distance": results.get('lpips_distance', None), "max_lpips_thresh": params.get('max_lpips_thresh', None)
        }
        full_path = os.path.join(folder_paths.get_output_directory(), analysis_file)
        try:
            history_data = self._load_history(full_path)
            history_data.append(final_run_data)
            with open(full_path, 'w') as f: json.dump(history_data, f, indent=4)
            STATE_MANAGER.cleanup_run(run_id)
            summary = f"结果已成功保存到 {analysis_file}。\n"
            summary += f"耗时: {generation_time:.2f}s, 缓存命中/总数: {results.get('cache_hits', 0)}/{results.get('total_inferences', 0)}"
            if final_run_data["lpips_distance"] is not None:
                summary += f", LPIPS距离: {final_run_data['lpips_distance']:.4f}"
            print(summary)
            return (latent, summary)
        except Exception as e:
            return (latent, f"写入文件时出错: {e}")

# LPIPS模型加载
class LPIPS_Model_Loader:
    def __init__(self): self.model = None
    @classmethod
    def INPUT_TYPES(cls): return {"required": {}}
    RETURN_TYPES = ("LPIPS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "utils/analysis"
    EXPERIMENTAL = True
    def load_model(self):
        if not LPIPS_AVAILABLE: raise Exception("LPIPS库未安装。请执行 'pip install lpips'")
        if STATE_MANAGER.get_lpips_model() is None:
            print("正在加载LPIPS模型 (vgg)...")
            lpips_model = lpips.LPIPS(net='vgg').cpu()
            STATE_MANAGER.set_lpips_model(lpips_model)
            print("LPIPS模型加载完成。")
        return (STATE_MANAGER.get_lpips_model(),)

# 基准图像存储
class Store_Baseline_Image:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"image": ("IMAGE",)}}
    RETURN_TYPES = ("BASELINE_IMG",)
    FUNCTION = "store_image"
    CATEGORY = "utils/analysis"
    EXPERIMENTAL = True
    def store_image(self, image):
        baseline_tensor = image.permute(0, 3, 1, 2).contiguous()
        STATE_MANAGER.set_baseline_image(baseline_tensor)
        return (baseline_tensor.clone(),)

# LPIPS评估
class TeaCache_LPIPS_Evaluator:
    @classmethod
    def INPUT_TYPES(cls): return {"required": { "test_image": ("IMAGE",), "baseline_image": ("BASELINE_IMG",), "lpips_model": ("LPIPS_MODEL",), "run_id": ("STRING", {"forceInput": True}), }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "evaluate"
    CATEGORY = "utils/analysis"
    EXPERIMENTAL = True
    def _preprocess_image(self, image_tensor): return image_tensor * 2.0 - 1.0
    def evaluate(self, test_image, baseline_image, lpips_model, run_id):
        if baseline_image is None: return ("错误: 未提供基准图像 (Baseline Image)。",)
        run_data = STATE_MANAGER.get_run_data(run_id)
        if not run_data: return (f"错误: 未找到ID为 {run_id} 的运行记录。",)
        test_image_t = test_image.permute(0, 3, 1, 2)
        test_img_proc = self._preprocess_image(test_image_t)
        base_img_proc = self._preprocess_image(baseline_image)
        device = 'cpu'
        lpips_model.to(device)
        distance = lpips_model(test_img_proc.to(device), base_img_proc.to(device))
        lpips_score = distance.item()
        run_data.setdefault('results', {})['lpips_distance'] = lpips_score
        return (f"LPIPS距离计算完成: {lpips_score:.4f}",)
