#!/usr/bin/env python3
"""CPU preparation or separately authorized microbatch profiling/training."""
import argparse
import json
from pathlib import Path
from ktbench import favor_plus_micro_profile as profile
from ktbench import favor_plus_micro_epochs as epochs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','profile-campaign','profile-worker','recover-profile',
        'estimate','training-template','campaign','train-epoch','verify-epoch','validate-best'))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--scope', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', choices=profile.pair.ARMS)
    parser.add_argument('--physical-batch', type=int, choices=profile.PHYSICAL_BATCHES)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--clock', type=Path)
    parser.add_argument('--execute-cuda', action='store_true')
    parser.add_argument('--execute-training', action='store_true')
    args = parser.parse_args()
    if args.action=='prepare':
        if not args.parent or not args.run_id:
            parser.error('parent plan and fresh run ID required')
        return profile.prepare(args.parent, args.run_id)
    if not args.plan:
        parser.error('plan required')
    if args.action=='estimate':
        profile.cpu_only(); return profile.estimate(args.plan)
    if args.action=='training-template':
        if not args.output:
            parser.error('output required')
        profile.cpu_only(); return profile.training_template(args.plan, args.output)
    if not args.scope:
        parser.error('separate actual execution scope required')
    if args.action in ('profile-campaign','profile-worker','recover-profile'):
        if not args.execute_cuda:
            parser.error('explicit CUDA flag and new profile approval required')
        if args.action=='profile-campaign':
            return profile.profile_campaign(args.plan,args.scope)
        if not args.arm or args.physical_batch is None:
            parser.error('fixed arm and physical batch required')
        record = json.loads(args.scope.read_text())
        return (profile.profile if args.action=='profile-worker' else profile.recover_profile)(args.plan,record,args.arm,args.physical_batch)
    if not args.execute_training:
        parser.error('explicit training flag and separate actual training/runtime approval required')
    if args.action=='campaign':
        return epochs.campaign(args.plan,args.scope)
    if not args.arm or args.epoch is None or not args.clock:
        parser.error('epoch, arm and immutable arm clock required')
    return epochs.worker(args.plan,json.loads(args.scope.read_text()),args.arm,args.epoch,args.action,json.loads(args.clock.read_text()))


if __name__=='__main__':
    print(json.dumps(main(), indent=2))
