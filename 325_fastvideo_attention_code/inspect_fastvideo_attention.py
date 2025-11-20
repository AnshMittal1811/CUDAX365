
import importlib.util
import os
import pkgutil

def main():
    spec = importlib.util.find_spec('fastvideo')
    if spec is None or spec.submodule_search_locations is None:
        print('fastvideo not installed. Install it to inspect attention kernels.')
        return
    base = list(spec.submodule_search_locations)[0]
    attention_dir = os.path.join(base, 'attention')
    if not os.path.isdir(attention_dir):
        print(f'No attention directory found at {attention_dir}')
        return
    files = []
    for root, _, filenames in os.walk(attention_dir):
        for name in filenames:
            if name.endswith(('.py', '.cu', '.cpp', '.cuh')):
                files.append(os.path.join(root, name))
    print('Attention-related files:')
    for path in files:
        print(path)

if __name__ == '__main__':
    main()
