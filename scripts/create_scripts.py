import os

OUTPUT_BASE = "/orange/ufdatastudios/c.okocha/afrispeech-entailment/outputs/CaptionBeforeReasoningV2"
NLI_JSONL = "/orange/ufdatastudios/c.okocha/afrispeech-entailment/result/Entailment/Medical/Llama/nli/medical_nli.jsonl"
AUDIO_DIR = "/orange/ufdatastudios/c.okocha/afrispeech-entailment/Audio/medical"

MODELS = [
    {
        "name": "AudioFlamingo3",
        "base_script": "/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3/run_consistency_original.sh",
        "output_script": "/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_3/run_medical_nli_caption.sh",
        "type": "af3"
    },
    {
        "name": "AudioFlamingo2",
        "base_script": "/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_2/slurm_scripts/run_medical_nli.sh",
        "output_script": "/orange/ufdatastudios/c.okocha/audio-flamingo-audio_flamingo_2/slurm_scripts/run_medical_nli_caption.sh",
        "type": "generic"
    },
    {
        "name": "Kimi",
        "base_script": "/orange/ufdatastudios/c.okocha/Kimi-Audio/slurm_scripts/run_medical_nli.sh",
        "output_script": "/orange/ufdatastudios/c.okocha/Kimi-Audio/slurm_scripts/run_medical_nli_caption.sh",
        "type": "generic"
    },
    {
        "name": "Qwen2.5Omni",
        "base_script": "/orange/ufdatastudios/c.okocha/AfroBust/models/Qwen/Qwen2.5-Omni-7B/slurm_scripts/run_medical_nli.sh",
        "output_script": "/orange/ufdatastudios/c.okocha/AfroBust/models/Qwen/Qwen2.5-Omni-7B/slurm_scripts/run_medical_nli_caption.sh",
        "type": "generic"
    },
    {
        "name": "Qwen2AudioInstruct",
        "base_script": "/orange/ufdatastudios/c.okocha/AfroBust/models/Qwen/Qwen2-Audio-7B-Instruct/slurm_scripts/run_medical_nli.sh",
        "output_script": "/orange/ufdatastudios/c.okocha/AfroBust/models/Qwen/Qwen2-Audio-7B-Instruct/slurm_scripts/run_medical_nli_caption.sh",
        "type": "generic"
    }
]

def modify_af3(content, model_name):
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('PROMPT_VARIANT='):
            new_lines.append('PROMPT_VARIANT="caption_reasoning"')
        elif line.strip().startswith('OUTPUT_BASE='):
            new_lines.append(f'OUTPUT_BASE="{OUTPUT_BASE}"')
        elif line.strip().startswith('#SBATCH --output='):
             new_lines.append(f'#SBATCH --output={OUTPUT_BASE}/{model_name}/logs/medical_nli_caption_%j.out')
        elif line.strip().startswith('#SBATCH --error='):
             new_lines.append(f'#SBATCH --error={OUTPUT_BASE}/{model_name}/logs/medical_nli_caption_%j.err')
        elif line.strip().startswith('#SBATCH --job-name'):
             new_lines.append(f'#SBATCH --job-name {model_name}-caption')
        elif line.strip().startswith('declare -A DATASETS=('):
            new_lines.append('declare -A DATASETS=(')
            # Add single medical dataset
            new_lines.append(f'    ["medical_nli"]="{NLI_JSONL}|{AUDIO_DIR}"')
            # Skip subsequent lines until closing parenthesis
            pass 
        elif line.strip().startswith('['): 
            # Skip old dataset entries like ["general"]=...
            continue
        elif '--task consistency' in line:
            # Also need to make sure log dir exists in the script or pre-create it?
            # The original script might create directories based on variables. 
            # Let's ensure the log dir variable is updated too if it exists.
            new_lines.append(line.replace('--task consistency', '--task nli'))
        else:
            new_lines.append(line)
            
    # Add a mkdir for safety at the start of script execution logic?
    # Hard to insert safely. Assuming standard script creates dirs.
    return '\n'.join(new_lines)

def modify_generic(content, model_name):
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('OUTPUT_BASE='):
            new_lines.append(f'OUTPUT_BASE="{OUTPUT_BASE}"')
        elif line.strip().startswith('TASK='):
            new_lines.append('TASK="nli_caption"')
        elif line.strip().startswith('OUTPUT_PREFIX='):
            new_lines.append('OUTPUT_PREFIX="medical_nli_caption"')
        elif line.strip().startswith('#SBATCH --job-name'):
            new_lines.append(f'#SBATCH --job-name {model_name}-medical_nli_caption')
        elif line.strip().startswith('#SBATCH --output='):
             new_lines.append(f'#SBATCH --output={OUTPUT_BASE}/{model_name}/logs/medical_nli_caption_%j.out')
        elif line.strip().startswith('#SBATCH --error='):
             new_lines.append(f'#SBATCH --error={OUTPUT_BASE}/{model_name}/logs/medical_nli_caption_%j.err')
        elif line.strip().startswith('JSONL_PATH='):
             new_lines.append(f'JSONL_PATH="{NLI_JSONL}"')
        elif line.strip().startswith('AUDIO_DIR='):
             new_lines.append(f'AUDIO_DIR="{AUDIO_DIR}"')
        else:
            new_lines.append(line)
            
    return '\n'.join(new_lines)

for m in MODELS:
    print(f"Processing {m['name']}...")
    try:
        with open(m['base_script'], 'r') as f:
            content = f.read()
            
        if m['type'] == 'af3':
            new_content = modify_af3(content, m['name'])
        else:
            new_content = modify_generic(content, m['name'])
            # Explicit fix for Qwen venv path which is relative in the layout
            if "Qwen" in m['name']:
                new_content = new_content.replace('source .venv/bin/activate', 'source /orange/ufdatastudios/c.okocha/AfroBust/.venv/bin/activate')
            
        with open(m['output_script'], 'w') as f:
            f.write(new_content)
            
        print(f"Created {m['output_script']}")
        
    except Exception as e:
        print(f"Error processing {m['name']}: {e}")

print("Done.")
