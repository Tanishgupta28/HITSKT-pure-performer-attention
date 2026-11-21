"""Provision or deactivate a non-demo administrator account as the MongoDB operator."""
import argparse
import sys
from datetime import datetime, timezone
from getpass import getpass
from pathlib import Path
from uuid import uuid4

from email_validator import validate_email
from pwdlib import PasswordHash
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('action',choices=['create','deactivate','activate'])
parser.add_argument('email')
parser.add_argument('--name')
parser.add_argument('--role',choices=['admin'])
args=parser.parse_args()
email=validate_email(args.email,check_deliverability=False).normalized.lower()
if email.endswith('@demo.lumen.example.com'):
    raise SystemExit('The demo domain is reserved.')
cfg=Settings()
with MongoClient(cfg.mongodb_uri,serverSelectionTimeoutMS=5000) as c:
    db=c[cfg.mongodb_database]
    if args.action=='create':
        if not args.role or not args.name or not 2<=len(args.name.strip())<=80: parser.error('Supply --role and --name for a new administrator.')
        if db.users.find_one({'email':email}): raise SystemExit('An account already exists. Roles are never elevated by this create command.')
        password=getpass('Set a password (at least 10 characters): ')
        if not 10<=len(password)<=128 or password!=getpass('Confirm the password: '): raise SystemExit('Passwords must match the length rules.')
        db.users.create_index('email',unique=True)
        db.users.insert_one({'_id':str(uuid4()),'email':email,'name':args.name.strip(),'role':args.role,'demo':False,
                             'onboarding_complete':True,'created_at':datetime.now(timezone.utc),'password_hash':PasswordHash.recommended().hash(password)})
        print('Administrator account created.')
    else:
        user=db.users.find_one_and_update({'email':email,'demo':{'$ne':True}},{'$set':{'inactive':args.action=='deactivate'}})
        if not user: raise SystemExit('Account not found.')
        if args.action=='deactivate': db.sessions.delete_many({'user_id':user['_id']})
        print('Account status updated.')
