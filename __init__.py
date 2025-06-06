from .TeaCache_Lumina2 import TeaCache_Lumina2
from .tc_test import TeaCache_Patcher, TeaCache_Result_Collector, LPIPS_Model_Loader, Store_Baseline_Image, TeaCache_LPIPS_Evaluator

NODE_CLASS_MAPPINGS = {
    "TeaCache_Lumina2": TeaCache_Lumina2,
    "TeaCache_Patcher": TeaCache_Patcher,
    "TeaCache_Result_Collector": TeaCache_Result_Collector,
    "LPIPS_Model_Loader": LPIPS_Model_Loader,
    "Store_Baseline_Image": Store_Baseline_Image,
    "TeaCache_LPIPS_Evaluator": TeaCache_LPIPS_Evaluator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TeaCache_Lumina2": "Lumina2 TeaCahe",
    "TeaCache_Patcher": "TeaCache Patcher 🍵",
    "TeaCache_Result_Collector": "TeaCache Result Collector 📊",
    "LPIPS_Model_Loader": "LPIPS Model Loader 🧠",
    "Store_Baseline_Image": "Store Baseline Image 🖼️",
    "TeaCache_LPIPS_Evaluator": "TeaCache LPIPS Evaluator 🧐",
}
