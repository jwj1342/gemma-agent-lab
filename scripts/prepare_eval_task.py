"""Create an inference task file without gold patches, tests or hidden labels."""
import argparse
import json
from pathlib import Path

PUBLIC_FIELDS = {'instance_id','repo','base_commit','problem_statement','hints_text','created_at'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,required=True)
    p.add_argument('--task-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    matches=[json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    matches=[t for t in matches if t['instance_id']==args.task_id]
    if len(matches)!=1:raise ValueError('Expected exactly one matching task')
    task={k:v for k,v in matches[0].items() if k in PUBLIC_FIELDS}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(task)+'\n')
    print(json.dumps({'task_id':args.task_id,'public_fields':sorted(task),'output':str(args.output)}))


if __name__=='__main__':main()
