"""CPU proc fixtures; witness consistency is never a GPU authorization grant."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/collect_hitskt_pid_identity.py'
spec = importlib.util.spec_from_file_location('hitskt_pid_witness', SCRIPT)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
NONCE = 'a'*32


@pytest.fixture
def proc(tmp_path):
    p = tmp_path/'7'
    (p/'ns').mkdir(parents=True)
    binary = tmp_path/'binary'
    binary.write_bytes(b'UNIT')
    (p/'exe').symlink_to(binary)
    (p/'status').write_text('Name: UNIT_SECRET\nPid: 7\nTgid: 7\nPPid: 1\nTracerPid: 0\n'
        'NSpid: 101 7\nNStgid: 101 7\nUid: 0 0 0 0\n')
    (p/'stat').write_text('7 (worker ) name )) R '+' '.join(['0']*18+['123']+['0']*5))
    (p/'cgroup').write_text('3:memory:/kubepods/UNIT_PRIVATE_PATH/cri-containerd-'+'b'*64+'.scope\n')
    for name in w.NAMESPACES:
        (p/'ns'/name).symlink_to(name+':[123]')
    for name in ('uid_map', 'gid_map'):
        (p/name).write_text('0 0 4294967295\n')
    (p/'timens_offsets').write_text('monotonic 0 0\nboottime 0 0\n')
    (p/'maps').write_text('UNIT_NO_LIBRARIES\n')
    (tmp_path/'self').symlink_to(p)
    (tmp_path/'1').symlink_to(p)
    boot = tmp_path/'sys/kernel/random'
    boot.mkdir(parents=True)
    (boot/'boot_id').write_text('11111111-1111-1111-1111-111111111111\n')
    return tmp_path


def test_live_pair_metadata_is_selected_and_stat_comm_is_parsed(proc):
    r = w.snapshot_pid(7, proc)
    assert r['status'] == 'STABLE_SNAPSHOT' and not r['mapping_proven']
    assert r['identity']['start_time_ticks'] == 123
    assert r['identity']['selected_status']['NSpid'] == [101, 7]
    assert r['identity']['container_ids'] == ['b'*64]
    assert 'UNIT_SECRET' not in json.dumps(r) and 'UNIT_PRIVATE_PATH' not in json.dumps(r)


@pytest.mark.parametrize('change', ['pid', 'start', 'exe', 'namespace', 'cgroup', 'uid', 'time_offset'])
def test_reuse_exec_namespace_cgroup_and_time_races_are_not_stable(proc, change):
    def mutate(p):
        if change == 'pid':
            (p/'status').write_text((p/'status').read_text().replace('Pid: 7', 'Pid: 8'))
        elif change == 'start':
            (p/'stat').write_text((p/'stat').read_text().replace(' 123 ', ' 124 '))
        elif change == 'exe':
            new = proc/'other'
            new.write_text('OTHER')
            (p/'exe').unlink()
            (p/'exe').symlink_to(new)
        elif change == 'namespace':
            (p/'ns/pid').unlink()
            (p/'ns/pid').symlink_to('pid:[124]')
        elif change == 'cgroup':
            (p/'cgroup').write_text('0::/moved\n')
        elif change == 'uid':
            (p/'status').write_text((p/'status').read_text().replace('Uid: 0 0 0 0', 'Uid: 1 1 1 1'))
        else:
            (p/'timens_offsets').write_text('monotonic 1 0\nboottime 1 0\n')
    r = w.snapshot_pid(7, proc, after_read=mutate)
    assert r['status'] in ('CHANGED_OR_REUSED', 'INCOMPLETE_OR_EXITED')
    assert not r['mapping_proven']


@pytest.mark.parametrize('value', [-1, 0, True, 7.0])
def test_invalid_pid_rejected(value):
    with pytest.raises(ValueError):
        w.snapshot_pid(value)


def test_missing_or_oversize_metadata_is_incomplete(proc):
    assert w.snapshot_pid(8, proc)['status'] == 'INCOMPLETE_OR_EXITED'
    (proc/'7/status').write_text('x'*65537)
    assert w.snapshot_pid(7, proc)['status'] == 'INCOMPLETE_OR_EXITED'


def test_missing_required_namespace_is_incomplete(proc):
    (proc/'7/ns/pid').unlink()
    assert w.snapshot_pid(7, proc)['status'] == 'INCOMPLETE_OR_EXITED'


def test_unsupported_time_namespace_remains_explicit(proc):
    for name in ('time', 'time_for_children'):
        (proc/'7/ns'/name).unlink()
    (proc/'7/timens_offsets').unlink()
    r = w.snapshot_pid(7, proc)
    assert r['status'] == 'STABLE_SNAPSHOT'
    assert r['identity']['namespaces']['time'] is None
    assert r['identity']['timens_offsets'] is None


@pytest.mark.parametrize('nonce', ['', 'a'*31, 'a'*65, 'z'*32, '../secret'])
def test_invalid_nonce_rejected(proc, nonce):
    with pytest.raises(ValueError):
        w.collect([7], nonce, 'host', proc)


def test_host_label_and_consistent_identity_never_authorize(proc):
    r = w.collect([7], NONCE, 'host', proc)
    assert r['source_requires_administrator_attestation']
    assert not r['mapping_proven'] and not r['gpu_execution_allowed']
    assert r['processes'][0]['status'] == 'STABLE_SNAPSHOT'


def test_pid_count_is_bounded(proc):
    for pids in ([], list(range(1, 66))):
        with pytest.raises(ValueError):
            w.collect(pids, NONCE, 'host', proc)


def test_cli_is_cpu_only_and_does_not_overwrite(tmp_path):
    output = tmp_path/'evidence.json'
    args = [sys.executable, str(SCRIPT), '--view', 'container', '--nonce', NONCE, '--output', str(output)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    first = subprocess.run(args, capture_output=True, text=True, env=env, timeout=15)
    assert first.returncode == 0, first.stderr
    saved = output.read_bytes()
    result = json.loads(saved)
    assert result['no_torch_import'] and result['no_cuda_driver_import']
    assert result['optimizer_calls'] == result['signals_sent'] == 0
    assert not result['gpu_execution_allowed']
    second = subprocess.run(args, capture_output=True, text=True, env=env, timeout=15)
    assert second.returncode != 0 and output.read_bytes() == saved


@pytest.mark.parametrize('args', [['--hold-seconds', '-1'], ['--hold-seconds', '46'], ['--hold-seconds', '1', '--pid', '7']])
def test_hold_limits_cannot_become_watchers(args):
    with pytest.raises(SystemExit):
        w.main(['--view', 'container', '--nonce', NONCE, *args])


def test_optional_hold_is_one_foreground_pause(monkeypatch):
    slept = []
    monkeypatch.setattr(w, 'collect', lambda *args: {'processes': [{'status': 'STABLE_SNAPSHOT'}]})
    monkeypatch.setattr(w.time, 'sleep', slept.append)
    assert w.main(['--view', 'container', '--nonce', NONCE, '--hold-seconds', '45']) == 0
    assert slept == [45]


def test_standalone_script_has_no_gpu_runtime_or_process_control_imports():
    tree = ast.parse(SCRIPT.read_text())
    imports = [n.names[0].name for n in ast.walk(tree) if isinstance(n, ast.Import)]
    assert set(imports) <= {'argparse', 'datetime', 'hashlib', 'json', 'os', 're', 'sys', 'time'}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ('kill', 'system', 'popen', 'setns', 'unshare') for n in ast.walk(tree))
