import os
import time
import inspect
from configs import config

def record_config(meta_dir, script_name):
    """Record non-path config parameters to meta/config_log.txt."""
    os.makedirs(meta_dir, exist_ok=True)
    log_file = os.path.join(meta_dir, 'config_log.txt')
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: {script_name}\n")
        f.write(f"{'-'*50}\n")
        
        for name, value in vars(config).items():
            if name.startswith('__') or inspect.ismodule(value) or inspect.isfunction(value) or inspect.isclass(value):
                continue

            is_path = False
            name_upper = name.upper()
            if any(k in name_upper for k in ['PATH', 'ROOT', 'DIR', 'FILE', 'CKPT']):
                is_path = True
            if isinstance(value, str) and (os.sep in value or '/' in value):
                is_path = True
                
            if not is_path:
                f.write(f"{name}: {value}\n")
        f.write(f"{'='*50}\n")
