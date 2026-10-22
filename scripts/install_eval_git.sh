#!/bin/bash
# Invoked inside the pinned Ubuntu 22.04 vLLM image on a preparation node.
set -eo pipefail
project_root=${1:?project root}
eval_venv=/home/jwj/venvs/gemma4-eval-v31
apt_root="$project_root/cache/eval-apt"
mkdir -p "$apt_root/lists/partial" "$apt_root/debs" "$eval_venv/vendor/git"
cat > "$apt_root/sources.list" <<'SOURCES'
deb https://archive.ubuntu.com/ubuntu jammy main
deb https://archive.ubuntu.com/ubuntu jammy-updates main
deb https://security.ubuntu.com/ubuntu jammy-security main
SOURCES
apt_options=(-o "Dir::State::lists=$apt_root/lists"
             -o "Dir::Etc::sourcelist=$apt_root/sources.list"
             -o Dir::Etc::sourceparts=- -o Debug::NoLocking=1
             -o APT::Get::List-Cleanup=0 -o APT::Get::AllowUnauthenticated=false)
apt-get "${apt_options[@]}" update
cd "$apt_root/debs"
apt-get "${apt_options[@]}" download git libpcre2-8-0
for package in ./*.deb; do
    dpkg-deb --extract "$package" "$eval_venv/vendor/git"
done
cat > "$eval_venv/bin/git" <<'WRAPPER'
#!/bin/sh
git_root=/home/jwj/venvs/gemma4-eval-v31/vendor/git
export GIT_EXEC_PATH="$git_root/usr/lib/git-core"
export GIT_TEMPLATE_DIR="$git_root/usr/share/git-core/templates"
export LD_LIBRARY_PATH="$git_root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$git_root/usr/bin/git" "$@"
WRAPPER
chmod 755 "$eval_venv/bin/git"
"$eval_venv/bin/git" --version
"$eval_venv/bin/python" - "$project_root" "$apt_root/debs" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, subprocess, sys, tempfile
root, debs = map(Path, sys.argv[1:])
git='/home/jwj/venvs/gemma4-eval-v31/bin/git'
with tempfile.TemporaryDirectory(dir=root/'cache',prefix='git-check-') as tmp:
    def run(*args):
        return subprocess.check_output([git,'-C',tmp,*args],text=True,stderr=subprocess.STDOUT)
    run('init','-q')
    (Path(tmp)/'sample.txt').write_text('before\n')
    run('add','sample.txt')
    run('-c','user.name=Runtime check','-c','user.email=runtime-check@invalid','commit','-qm','Check Git operations')
    (Path(tmp)/'sample.txt').write_text('after\n')
    diff=run('diff','HEAD')
    assert '-before' in diff and '+after' in diff
result={'checked_at':datetime.now(timezone.utc).isoformat(),
        'git_version':subprocess.check_output([git,'--version'],text=True).strip(),
        'source':'Ubuntu jammy/main, jammy-updates/main, jammy-security/main; apt signature validation enabled',
        'packages':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(debs.glob('*.deb'))],
        'verified':['init','add','commit with temporary test identity','diff HEAD'],
        'scope':'Offline local repository operations in the pinned container; not a new image build'}
(root/'resource/manifests/eval-git.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
