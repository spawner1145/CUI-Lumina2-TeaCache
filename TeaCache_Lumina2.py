import torch
import numpy as np
from comfy.ldm.common_dit import pad_to_patch_size  # noqa
from unittest.mock import patch
import re

#DEFAULT_COEFFICIENTS = [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344]
DEFAULT_COEFFICIENTS = [225.7042019806413, -608.8453716535591, 304.1869942338369, 124.21267720116742, -1.4089066892956552]

# referenced from https://github.com/spawner1145/TeaCache/blob/main/TeaCache4Lumina2/teacache_lumina2.py
# firstly transplanted by @fexli https://github.com/fexli
# retransplanted by @spawner1145 https://github.com/spawner1145
def teacache_forward_working(
    self, x, timesteps, context, num_tokens, attention_mask=None, transformer_options={}, **kwargs
):
    if not hasattr(self, 'teacache_state'):
        self.teacache_state = {
            "cnt": 0,
            "num_steps": transformer_options.get("num_steps"),
            "cache": transformer_options.get("cache", {}),
            "uncond_seq_len": transformer_options.get("uncond_seq_len")
        }
    if not isinstance(self.teacache_state.get("cache"), dict):
        self.teacache_state["cache"] = {}

    if self.teacache_state.get("num_steps") is None and transformer_options.get("num_steps") is not None:
        self.teacache_state["num_steps"] = transformer_options.get("num_steps")

    cap_feats = context
    cap_mask = attention_mask
    bs, c_channels, h_img, w_img = x.shape
    x = pad_to_patch_size(x, (self.patch_size, self.patch_size))
    t = (1.0 - timesteps).to(dtype=x.dtype)

    t_emb = self.t_embedder(t, dtype=x.dtype)
    adaln_input = t_emb

    if cap_feats is not None:
        cap_feats = self.cap_embedder(cap_feats)

    x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t_emb, num_tokens)
    freqs_cis = freqs_cis.to(x.device)
    max_seq_len = x.shape[1]
    should_calc = True

    enable_teacache = transformer_options.get('enable_teacache', False)
    current_cache = None
    modulated_inp = None
    
    if enable_teacache:
        cache_key = max_seq_len
        if cache_key not in self.teacache_state['cache']:
            self.teacache_state['cache'][cache_key] = {
                "accumulated_rel_l1_distance": 0.0,
                "previous_modulated_input": None,
                "previous_residual": None,
            }
        current_cache = self.teacache_state['cache'][cache_key]
        
        try:
           if self.layers and hasattr(self.layers[0], 'adaLN_modulation'):
               mod_result = self.layers[0].adaLN_modulation(adaln_input.clone())
               if isinstance(mod_result, (list, tuple)) and len(mod_result) > 0:
                    modulated_inp = mod_result[0]
               elif torch.is_tensor(mod_result):
                   modulated_inp = mod_result
               else:
                    raise ValueError("adaLN_modulation returned unexpected type or empty list/tuple")
           else:
                raise AttributeError("Layer 0 or adaLN_modulation not found")
        except Exception as e:
             print(f"Warning: TeaCache - Failed to get modulated_inp: {e}. Disabling cache for this step.")
             enable_teacache = False
             should_calc = True
             modulated_inp = None
             if current_cache:
                  current_cache["previous_modulated_input"] = None
                  current_cache["accumulated_rel_l1_distance"] = 0.0

    if enable_teacache and modulated_inp is not None and current_cache is not None:
        num_steps_in_state = self.teacache_state.get("num_steps")
        if num_steps_in_state is None or num_steps_in_state == 0:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        elif self.teacache_state['cnt'] == 0 or self.teacache_state['cnt'] == num_steps_in_state - 1:
            should_calc = True
            current_cache["accumulated_rel_l1_distance"] = 0.0
        else:
            if current_cache.get("previous_modulated_input") is not None:
                # coefficients = [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344]
                coefficients = transformer_options.get('coefficients', DEFAULT_COEFFICIENTS)
                if not isinstance(coefficients, (list, tuple)):
                      coefficients = DEFAULT_COEFFICIENTS
                
                try:
                    rescale_func = np.poly1d(coefficients)
                except Exception as e:
                    print(f"Warning: TeaCache np.poly1d failed with coefficients {coefficients}: {e}. Using default for this step.")
                    rescale_func = np.poly1d(DEFAULT_COEFFICIENTS)

                prev_mod_input = current_cache["previous_modulated_input"]
                if prev_mod_input.shape != modulated_inp.shape:
                     print(f"Warning: TeaCache - modulated input shape mismatch: prev={prev_mod_input.shape}, curr={modulated_inp.shape}. Forcing recalculation.")
                     should_calc = True
                     current_cache["accumulated_rel_l1_distance"] = 0.0
                     rel_l1_change = float('inf')
                else:
                    prev_mean = prev_mod_input.abs().mean()
                    if prev_mean.item() > 1e-9:
                        rel_l1_change = ((modulated_inp - prev_mod_input).abs().mean() / prev_mean).cpu().item()
                    else:
                        rel_l1_change = 0.0 if modulated_inp.abs().mean().item() < 1e-9 else float('inf')
                    
                    rescaled_value = rescale_func(rel_l1_change)
                    if np.isnan(rescaled_value) or np.isinf(rescaled_value):
                         current_cache["accumulated_rel_l1_distance"] = float('inf')
                    else:
                         current_cache["accumulated_rel_l1_distance"] += rescaled_value

                    if current_cache["accumulated_rel_l1_distance"] < transformer_options.get('rel_l1_thresh', 0.3):
                        should_calc = False
                    else:
                        should_calc = True
                        current_cache["accumulated_rel_l1_distance"] = 0.0
            else:
                 should_calc = True
                 current_cache["accumulated_rel_l1_distance"] = 0.0

        current_cache["previous_modulated_input"] = modulated_inp.clone()

        if self.teacache_state.get('uncond_seq_len') is None:
            self.teacache_state['uncond_seq_len'] = cache_key

        if num_steps_in_state is not None and cache_key != self.teacache_state.get('uncond_seq_len'):
            self.teacache_state['cnt'] += 1
            if self.teacache_state['cnt'] >= num_steps_in_state:
                self.teacache_state['cnt'] = 0

    can_reuse_residual = (enable_teacache and 
                         not should_calc and 
                         current_cache and 
                         current_cache.get("previous_residual") is not None and
                         current_cache["previous_residual"].shape == x.shape) # 检查形状

    if can_reuse_residual:
         processed_x = x + current_cache["previous_residual"]
    else:
        if enable_teacache and not should_calc and current_cache and current_cache.get("previous_residual") is not None and current_cache["previous_residual"].shape != x.shape:
             print(f"Warning: TeaCache - Residual shape mismatch: cache={current_cache['previous_residual'].shape}, input={x.shape}. Forcing recalculation.")
             if current_cache:
                  current_cache["accumulated_rel_l1_distance"] = 0.0

        original_x = x.clone()
        current_x_for_processing = x
        for layer in self.layers:
            current_x_for_processing = layer(current_x_for_processing, mask, freqs_cis, adaln_input)

        if enable_teacache and current_cache:
            current_cache["previous_residual"] = current_x_for_processing - original_x
            current_cache["accumulated_rel_l1_distance"] = 0.0
        processed_x = current_x_for_processing

    output = self.final_layer(processed_x, adaln_input)
    output = self.unpatchify(output, img_size, cap_size, return_tensor=True)[:, :, :h_img, :w_img]

    return -output


class TeaCache_Lumina2:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "rel_l1_thresh": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 10.0, "step": 0.001}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "The start percentage of the steps that will apply TeaCache. / TeaCache开始应用的步数百分比。"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "The end percentage of the steps that will apply TeaCache. / TeaCache停止应用的步数百分比。"}),
                 "coefficients_string": ("STRING", {
                    "multiline": True, 
                    "default": str(DEFAULT_COEFFICIENTS),
                    "tooltip": "Coefficients for np.poly1d. Format: 393.7, -603.5, 209.1, -23.0, 0.86 (with or without brackets []) / 用于 np.poly1d 的系数。格式: 393.7, -603.5, 209.1, -23.0, 0.86 (可带或不带方括号[])"
                    }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_teacache"
    CATEGORY = "utils"

    def patch_teacache(self, model, rel_l1_thresh, start_percent, end_percent, coefficients_string):
        if rel_l1_thresh == 0:
            try:
               diffusion_model = model.get_model_object("diffusion_model")
               if hasattr(diffusion_model, 'teacache_state'):
                   delattr(diffusion_model, 'teacache_state')
            except:
                pass
            return (model,)

        parsed_coefficients = DEFAULT_COEFFICIENTS
        try:
            s = re.sub(r'[\[\]\s]', '', coefficients_string.strip())
            if not s:
                 parsed_coefficients = DEFAULT_COEFFICIENTS
            else:
                coeff_list = [float(item) for item in s.split(',') if item]
                if not coeff_list:
                     parsed_coefficients = DEFAULT_COEFFICIENTS
                else:
                    parsed_coefficients = coeff_list
        except ValueError:
            print(f"Warning: TeaCache - Could not parse coefficients string: '{coefficients_string}'. Using default coefficients: {DEFAULT_COEFFICIENTS}")
            parsed_coefficients = DEFAULT_COEFFICIENTS
        except Exception as e:
             print(f"Warning: TeaCache - Error parsing coefficients '{coefficients_string}': {e}. Using default coefficients: {DEFAULT_COEFFICIENTS}")
             parsed_coefficients = DEFAULT_COEFFICIENTS


        new_model = model.clone()
        if 'transformer_options' not in new_model.model_options:
            new_model.model_options['transformer_options'] = {}

        new_model.model_options["transformer_options"]["cache"] = {}
        new_model.model_options["transformer_options"]["uncond_seq_len"] = None
        new_model.model_options["transformer_options"]["rel_l1_thresh"] = rel_l1_thresh
        new_model.model_options["transformer_options"]["coefficients"] = parsed_coefficients
        
        diffusion_model = new_model.get_model_object("diffusion_model")

        if hasattr(diffusion_model, 'teacache_state'):
            delattr(diffusion_model, 'teacache_state')

        context_patch_manager = patch.multiple(
            diffusion_model,
            forward=teacache_forward_working.__get__(diffusion_model, diffusion_model.__class__)
        )

        def unet_wrapper_function(model_function, kwargs):
            input_val = kwargs["input"]
            timestep = kwargs["timestep"]
            c_condition_dict = kwargs["c"]
            cond_or_uncond = kwargs["cond_or_uncond"]

            if not isinstance(c_condition_dict, dict): c_condition_dict = {}
            if "transformer_options" not in c_condition_dict or not isinstance(c_condition_dict["transformer_options"], dict):
                c_condition_dict["transformer_options"] = {}

            for key, value in new_model.model_options["transformer_options"].items():
                if key not in c_condition_dict["transformer_options"]:
                    c_condition_dict["transformer_options"][key] = value

            current_step_index = 0
            if "sample_sigmas" not in c_condition_dict["transformer_options"] or \
               c_condition_dict["transformer_options"]["sample_sigmas"] is None:
                print("warning: TeaCache - 'sample_sigmas' not found in c.transformer_options.TeaCache might not work correctly.")
                c_condition_dict["transformer_options"]["enable_teacache"] = False
                c_condition_dict["transformer_options"]["num_steps"] = 1
                if hasattr(diffusion_model, 'teacache_state'):
                     delattr(diffusion_model, 'teacache_state')
            else:
                sigmas = c_condition_dict["transformer_options"]["sample_sigmas"]
                total_sampler_steps = len(sigmas)
                if total_sampler_steps > 0:
                   c_condition_dict["transformer_options"]["num_steps"] = total_sampler_steps
                else:
                   c_condition_dict["transformer_options"]["num_steps"] = 1
                   c_condition_dict["transformer_options"]["enable_teacache"] = False


                if hasattr(diffusion_model, 'teacache_state') and diffusion_model.teacache_state is not None:
                    if diffusion_model.teacache_state.get("num_steps") != total_sampler_steps and total_sampler_steps > 0:
                        delattr(diffusion_model, 'teacache_state')
                        c_condition_dict["transformer_options"]["cache"] = {}

                current_timestep = timestep[0].to(device=sigmas.device, dtype=sigmas.dtype)
                # 使用 torch.isclose 和 any() 来处理浮点数比较
                close_mask = torch.isclose(sigmas, current_timestep, atol=1e-6)
                if close_mask.any():
                     matched_step_index = torch.nonzero(close_mask, as_tuple=True)[0]
                     current_step_index = matched_step_index[0].item()
                else:
                    current_step_index = 0 
                    if total_sampler_steps > 1:
                         try:
                            indices = torch.where(sigmas >= current_timestep)[0]
                            if len(indices) > 0:
                                 current_step_index = indices[-1].item()
                            else:
                                 current_step_index = total_sampler_steps -1
                         except:
                            for i in range(total_sampler_steps - 1):
                                if (sigmas[i] - current_timestep) * (sigmas[i + 1] - current_timestep) <= 0:
                                    current_step_index = i
                                    break

                current_percent = 0.0
                if total_sampler_steps > 1:
                    current_percent = current_step_index / max(1, (total_sampler_steps - 1))
                # elif total_sampler_steps == 1:
                #    current_percent = 0.0
                elif total_sampler_steps <= 0 :
                    # current_percent = 0.0
                    c_condition_dict["transformer_options"]["enable_teacache"] = False

                if start_percent <= current_percent <= end_percent and total_sampler_steps > 0:
                    c_condition_dict["transformer_options"]["enable_teacache"] = True
                else:
                    c_condition_dict["transformer_options"]["enable_teacache"] = False

            if current_step_index == 0:
                 # print("Debug: TeaCache - Resetting state at step 0")
                 if hasattr(diffusion_model, 'teacache_state'):
                     delattr(diffusion_model, 'teacache_state')
                 c_condition_dict["transformer_options"]["cache"] = {}


            with context_patch_manager:
                return model_function(input_val, timestep, **c_condition_dict)

        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)
