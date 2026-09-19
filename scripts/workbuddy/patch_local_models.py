"""Stage/apply a checked update to the existing WorkBuddy 5.5.6 local-only patch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
BUNDLES = ('codebuddy.js', 'codebuddy-headless.js')

def patch(source, policy):
    marker = '/* WORKBUDDY_LOCAL_QWEN_ONLY_V1 */'
    if not source.startswith(marker):
        raise ValueError('Expected the reviewed V1 patch; refusing an unknown bundle')
    end = source.index('\n})();') + len('\n})();')
    result = policy.rstrip() + source[end:]
    replacements = {
        'currentModelId:__wbQwenOnly.id': 'currentModelId:__wbLocalOnly.currentId(eu,el)',
        'currentModelId:eh.length?__wbQwenOnly.id:void 0': 'currentModelId:__wbLocalOnly.currentId(eh,em)',
        'return __wbQwenOnly.select(this);': 'return __wbLocalOnly.select(this,eA);',
        'let ec=await this.agentManager.getModel(eA.name);__wbQwenOnly.assertModel(ec.id)': 'let ec=await __wbLocalOnly.select(this.agentManager,eA.name,el);__wbLocalOnly.assertModel(ec.id)',
        'return __wbQwenOnly.base;': 'return __wbLocalOnly.resolveBaseURL(eA);',
    }
    for old, new in replacements.items():
        if result.count(old) != 1:
            raise ValueError(f'Unexpected patch anchor count: {old}')
        result = result.replace(old, new)
    result = result.replace('__wbQwenOnly.', '__wbLocalOnly.')
    assert '__wbQwenOnly' not in result
    assert 'configured local Qwen model' not in result
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    stage = ROOT / 'installer_files/workbuddy-local-models-v2'
    stage.mkdir(parents=True, exist_ok=True)
    policy = (Path(__file__).parent / 'local-model-policy.js').read_text(encoding='utf-8-sig')
    manifest = {}
    for name in BUNDLES:
        data = (args.dist / name).read_bytes()
        updated = patch(data.decode('utf-8'), policy)
        target = stage / name
        target.write_text(updated, encoding='utf-8', newline='')
        subprocess.run(['node', '--check', str(target)], check=True)
        manifest[name] = {'before_sha256': hashlib.sha256(data).hexdigest(),
                          'after_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    subprocess.run(['node', str(ROOT / 'tests/test_workbuddy_local_policy.cjs'), str(stage)], check=True)
    (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if args.apply:
        # Recheck all originals before modifying either bundle.
        for name in BUNDLES:
            assert hashlib.sha256((args.dist / name).read_bytes()).hexdigest() == manifest[name]['before_sha256']
        backup = ROOT / 'backups' / ('workbuddy-local-models-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
        backup.mkdir(parents=True)
        for name in BUNDLES:
            shutil.copy2(args.dist / name, backup / name)
        installed = []
        try:
            for name in BUNDLES:
                installed.append(name)
                shutil.copyfile(stage / name, args.dist / name)
                assert hashlib.sha256((args.dist / name).read_bytes()).hexdigest() == manifest[name]['after_sha256']
        except Exception:
            for name in installed:
                shutil.copy2(backup / name, args.dist / name)
            raise
        print('Installed both bundles. Backup:', backup)
    else:
        print('Validated staged bundles:', stage)

if __name__ == '__main__':
    main()
