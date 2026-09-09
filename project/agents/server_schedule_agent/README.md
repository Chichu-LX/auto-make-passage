# Server Schedule Agent

This Agent owns server inventory validation, immutable workspace/environment staging, and main-benchmark launch. The later Experiment Code Run Agent may re-probe only the approved main servers and launch reviewed supplementary jobs through the same safe backend. Both require `project/configs/servers.local.json`; that credential-free inventory stores a `credential_prefix`, workspace and resource policy, never a host password or private-key path. SSH host/user/password or private-key settings live only in `project/agents/server_schedule_agent/.env`.

Create the private file from `.env.example`, fill either password or key authentication, and set exact owner-only permissions:

```bash
cp project/agents/server_schedule_agent/.env.example project/agents/server_schedule_agent/.env
chmod 600 project/agents/server_schedule_agent/.env
```

The loader rejects symlinks, files other than mode `600`, malformed hosts/users/ports, missing authentication values, and unsafe private-key permissions. It never falls back to the root `.env` or process environment. Passwords and encrypted-key passphrases are passed only through a temporary `SSH_ASKPASS` child environment; they are never placed in argv, JSON, reports or logs. SSH always requires strict host-key checking, so the server's independently verified host key must already exist in the user's `known_hosts`.

Execution order is fixed: validate approved artifacts, probe the fleet, calculate eligible GPUs and predicted makespan, create a project/run/attempt directory, create a fresh Conda environment, copy verified data and code/model files, install the local code, run the CUDA preflight, and execute Code Make's reviewed minimum demo in the foreground on every selected server. A failed or timed-out demo blocks the schedule before any persistent job begins. This stage runs only `main_benchmark` experiment/seed jobs; ablations, case studies, module analyses and other designed supplementary experiments are reserved for Experiment Code Run after the main result gate passes. Independent main experiment/seed jobs are spread over all usable GPUs. A distributed command may instead use all GPUs assigned to one server.

After installation the backend runs a fixed PyTorch CUDA preflight inside the new environment and verifies that every selected GPU is visible. Jobs assigned to different GPUs run concurrently; jobs sharing one GPU use a run-scoped file lock and execute as a queue, avoiding accidental same-card oversubscription while retaining full fleet parallelism.

After every server demo passes, each long job is detached with `nohup`. The generated remote runner uses `subprocess.Popen(..., shell=False)`, writes bootstrap and full job logs plus one JSONL metric stream per job, persists PID-linked `status.json`, and copies the best checkpoint file. A dropped SSH connection does not terminate the job. No directory-removal operation is generated or allowed. Every schedule sets a 12-hour result-report interval.

The default live registry will not launch work until a valid local config explicitly marks a profile with `execution_enabled: true`. A real server was not contacted during the local test suite.
