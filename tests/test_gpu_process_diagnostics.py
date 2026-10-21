"""CPU-only attribution evidence/redaction/ABI tests; never grant GPU use."""
import ctypes
import errno
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest
from ktbench import gpu_process_diagnostics as d
from ktbench import favor_plus_resource_telemetry as t


def test_status_exposes_selected_namespace_fields_without_secrets():
    r=d.parse_status('Pid:\t7\nTgid:\t7\nPPid:\t1\nTracerPid:\t0\nNSpid:\t101 7\nNStgid:\t101 7\nUid:\t0 0 0 0\nCapEff:\t0\nSecret: UNIT-TOKEN\nName: UNIT-SECRET\n')
    assert r['NSpid']==[101,7] and not r['cap_sys_admin']
    assert 'UNIT-' not in json.dumps(r) and 'Secret' not in r and 'Name' not in r


@pytest.mark.parametrize('line',['Pid: -1','Pid: 0','Tgid: 0','NSpid: 0','NSpid: 7 secret','NSpid:','Uid: 0 token','CapEff: garbage'])
def test_invalid_metadata_cannot_be_promoted_to_identity(line):
    with pytest.raises(ValueError):d.parse_status(line)


def test_cgroup_records_fingerprints_not_raw_paths_or_env():
    text='3:memory:/kubepods.slice/UNIT_PRIVATE_PATH\n0::/\n'
    r=d.cgroup_summary(text)
    assert r['hierarchies'][0]['kubepods'] and r['hierarchies'][1]['root_only']
    assert 'UNIT_PRIVATE_PATH' not in json.dumps(r)


@pytest.mark.parametrize('text',['bad','3:memory:relative','x:cpu:/safe'])
def test_malformed_cgroup_is_incomplete(text):
    with pytest.raises(ValueError):d.cgroup_summary(text)


def process_fixture(root):
    proc=root/'7';(proc/'fd').mkdir(parents=True);(proc/'ns').mkdir()
    binary=root/'python';binary.write_bytes(b'UNIT_BINARY');binary.chmod(0o755)
    (proc/'exe').symlink_to(binary);(proc/'cwd').symlink_to(root)
    (proc/'comm').write_text('python\n')
    (proc/'cmdline').write_bytes(b'python\0--token\0UNIT_SECRET\0')
    (proc/'status').write_text('Pid: 7\nTgid: 7\nPPid: 1\nState: R\nUid: 0 0 0 0\nNSpid: 101 7\n')
    (proc/'stat').write_text('7 (python) R '+' '.join(['0']*18+['10']+['0']*5))
    (proc/'cgroup').write_text('0::/\n')
    for name in ('pid','pid_for_children','mnt','cgroup','user'):
        target=root/('namespace_'+name);target.touch();(proc/'ns'/name).symlink_to(target)
    return proc


def test_multi_level_namespace_chain_is_evidence_not_alias_permission(tmp_path):
    process_fixture(tmp_path);r=d.process_evidence(7,tmp_path)
    assert r['status']=='VISIBLE_IDENTITY_SNAPSHOT' and r['selected_status']['NSpid']==[101,7]
    assert r['mapping_proven'] is False and 'UNIT_SECRET' not in json.dumps(r)


@pytest.mark.parametrize('change',['pid_reuse','exec','namespace','cgroup','uid'])
def test_raced_identity_is_not_reliable(tmp_path,change):
    proc=process_fixture(tmp_path)
    def mutate(p):
        if change=='pid_reuse':(p/'stat').write_text((p/'stat').read_text().replace(' 10 ',' 11 '))
        elif change=='exec':
            other=tmp_path/'other_binary';other.touch();(p/'exe').unlink();(p/'exe').symlink_to(other)
        elif change=='namespace':
            other=tmp_path/'other_namespace';other.touch();(p/'ns/pid').unlink();(p/'ns/pid').symlink_to(other)
        elif change=='cgroup':(p/'cgroup').write_text('0::/changed\n')
        else:(p/'status').write_text((p/'status').read_text().replace('Uid: 0 0 0 0','Uid: 1 1 1 1'))
    r=d.process_evidence(7,tmp_path,after_read=mutate)
    assert r['status']=='CHANGED_OR_REUSED' and not r['mapping_proven']


def test_missing_host_process_does_not_mean_inactive(tmp_path):
    assert d.process_evidence(7,tmp_path)['status']=='NOT_VISIBLE'


def test_parent_namespace_denial_preserves_unknown(monkeypatch):
    monkeypatch.setattr(os,'open',lambda *a:123)
    monkeypatch.setattr(os,'close',lambda *a:None)
    def deny(*args):raise OSError(errno.EPERM,'UNIT permission')
    monkeypatch.setattr(d.fcntl,'ioctl',deny)
    r=d.namespace_parent()
    assert r['status']=='UNAVAILABLE' and r['error']=='EPERM' and not r['mapping_proven']


@pytest.mark.parametrize('kind',['empty','nvml_compute','holder','inaccessible','light','no_nvml_visibility'])
def test_no_diagnostic_snapshot_grants_gpu_execution(kind):
    snap=dict(conflicts=[],inaccessible_fd_pids=[],light_processes=[]);nvml=dict(compute_processes=[])
    if kind=='nvml_compute':nvml['compute_processes']=[{'pid':101}]
    elif kind=='holder':snap['conflicts']=[{'pid':7}]
    elif kind=='inaccessible':snap['inaccessible_fd_pids']=[7]
    elif kind=='light':snap['light_processes']=[{'pid':7}]
    elif kind=='no_nvml_visibility':nvml={}
    r=d.diagnostic_decision(snap,nvml)
    assert not r['gpu_execution_allowed'] and not r['pid_mapping_proven']
    assert (r['status']=='BLOCKED_ACTIVE_OR_UNATTRIBUTED_GPU_USE')==(kind!='empty')


def test_import_cannot_initialize_torch_or_cuda():
    code='import sys;import ktbench.gpu_process_diagnostics;assert "torch" not in sys.modules;assert not any("libcuda.so" in x for x in open("/proc/self/maps"))'
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=15,env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
    assert p.returncode==0,p.stderr


def test_existing_nvml_ctypes_layout_matches_official_c_header(tmp_path,monkeypatch):
    compiler=shutil.which('cc');header=Path('/usr/local/cuda/include/nvml.h')
    if compiler is None or not header.exists():pytest.skip('official header/compiler unavailable')
    root=Path(__file__).resolve().parents[1]
    source=root/'reports/favor_plus_v1/gpu_pid_attribution_20261021_v1/nvml_process_abi.c';binary=tmp_path/'cpu_abi'
    subprocess.run([compiler,'-I'+str(header.parent),str(source),'-o',str(binary)],check=True,capture_output=True,timeout=20)
    expected=json.loads(subprocess.run([str(binary)],capture_output=True,text=True,check=True,timeout=5).stdout)
    captured={}
    class Fake:
        def nvmlInit_v2(self):return 0
        def nvmlShutdown(self):return 0
        def nvmlDeviceGetHandleByUUID(self,*args):return 0
        def nvmlDeviceGetMemoryInfo(self,handle,memory):
            memory._obj.total=40;memory._obj.free=39;memory._obj.used=1;return 0
        def nvmlDeviceGetComputeRunningProcesses_v2(self,handle,count,data):
            count._obj.value=1
            if data is None:return 7
            captured['type']=data._type_
            data[0].pid=101;data[0].usedGpuMemory=1;data[0].gpuInstanceId=2;data[0].computeInstanceId=0
            return 0
    monkeypatch.setattr(t.ctypes,'CDLL',lambda *args:Fake())
    result=t.nvml_snapshot('UNIT_FAKE_NVML_ONLY');ctype=captured['type']
    observed={'size':ctypes.sizeof(ctype),**{name:getattr(ctype,name).offset for name in expected if name!='size'}}
    assert observed==expected
    assert result['compute_processes'][0]['pid']==101
    print('Official-header CPU ABI:',json.dumps(observed,sort_keys=True))
