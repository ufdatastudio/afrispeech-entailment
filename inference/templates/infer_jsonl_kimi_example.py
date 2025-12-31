# ========== KIMI-SPECIFIC CUSTOMIZATION EXAMPLE ==========
# 
# Copy infer_jsonl.py to your Kimi project folder, then replace
# the init_model() and generate_with_model() functions with these:

def init_model(model_path: str):
    """Initialize Kimi model."""
    from kimia_infer.api.kimia import KimiAudio
    return KimiAudio(model_path=model_path, load_detokenizer=True)


def generate_with_model(model, messages: List[Dict], sampling_params: Dict, max_new_tokens: int) -> str:
    """Generate with Kimi model."""
    # Kimi-specific sampling parameters
    sampling_params_kimi = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": sampling_params.get("temperature", 0.0),
        "text_top_k": sampling_params.get("top_k", 5),
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    _, text = model.generate(
        messages,
        **sampling_params_kimi,
        output_type="text",
        max_new_tokens=max_new_tokens
    )
    return text

