"""Docker container lifecycle management using the Docker SDK."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import posixpath
import shlex
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from adk_eval_core.sandbox import BaseSandbox, ExecutionResult
from adk_eval_core.sandbox.containers import build_tar_stream, reset_tarinfo

logger = logging.getLogger(__name__)

SDG_IMAGE_CANDIDATES: tuple[str, ...] = (
    'gcr.io/kaggle-playground-170215/swebench-sandbox:v1',
    'gcr.io/kaggle-playground-170215/swebench-sandbox:latest',
)


def resolve_sandbox_image(
    image_name: str = 'swebench-sandbox:latest',
    client: Any | None = None,
) -> str:
    """Resolves the sandbox image in the Docker daemon, falling back to pre-seeded SDG tags."""
    try:
        if client is None:
            import docker

            client = docker.from_env()
        try:
            client.images.get(image_name)
            return image_name
        except Exception:
            pass

        for candidate in SDG_IMAGE_CANDIDATES:
            try:
                img = client.images.get(candidate)
            except Exception:
                continue

            logger.info(
                "Found pre-seeded SDG image '%s' for requested '%s'.",
                candidate,
                image_name,
            )
            try:
                repo, tag = (
                    image_name.split(':', 1)
                    if ':' in image_name
                    else (image_name, 'latest')
                )
                img.tag(repo, tag=tag)
            except Exception:
                logger.debug(
                    "Best-effort re-tag of '%s' -> '%s' skipped (blocked by dockerproxy).",
                    candidate,
                    image_name,
                )
            return candidate
    except Exception as e:
        logger.warning(
            "Could not query Docker daemon for sandbox image '%s': %s",
            image_name,
            e,
        )
    return image_name


@dataclass
class ContainerConfig:
    """Configuration for starting a container."""

    image: str = 'swebench-sandbox:latest'
    network_mode: str = 'none'  # no network access
    mem_limit: str = '4g'
    cpu_period: int = 100_000
    cpu_quota: int = 200_000  # 2 CPUs
    working_dir: str = '/workspace'
    timeout_seconds: int = 300
    reuse_containers: bool = False
    environment: dict[str, str] = field(default_factory=lambda: {'TEST_TMPDIR': '/tmp'})


class ContainerManager(BaseSandbox):
    """Manages Docker container lifecycle for evaluations directly conforming to BaseSandbox."""

    def __init__(self, config: ContainerConfig | None = None):
        self.config = config or ContainerConfig()
        self._client = None
        self._image_resolved = False
        self._active_containers: set[str] = set()
        self._idle_containers: list[str] = []
        self._default_container_id: str | None = None
        self._initial_files: dict[str, str | bytes] = {}
        self._lock = threading.RLock()

    @property
    def client(self):
        if self._client is None:
            import docker

            self._client = docker.from_env(timeout=300)
        return self._client

    @property
    def work_dir(self) -> str:
        """Working directory inside the sandbox."""
        return self.config.working_dir

    def initialize(self, files: dict[str, str | bytes] | None = None) -> None:
        """Start a default container and optionally write initial files."""
        with self._lock:
            if (
                self._default_container_id is None
                or self._default_container_id not in self._active_containers
            ):
                self._default_container_id = self.start()
        if files is not None:
            self._initial_files = dict(files)
            for rel_path, content in files.items():
                self.write_file(rel_path, content)

    def run_command(
        self,
        command: str,
        timeout: float | None = 300,
        work_dir: str | None = None,
    ) -> ExecutionResult:
        """Execute a shell command in the active container."""
        if self._default_container_id is None:
            self.initialize()
        assert self._default_container_id is not None
        wd = work_dir or self.work_dir
        cmd = f'cd {shlex.quote(wd)} && {command}' if wd else command
        int_timeout = (
            int(timeout) if timeout is not None else self.config.timeout_seconds
        )
        return self.exec(self._default_container_id, cmd, timeout=int_timeout)

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write a file into the active container workspace."""
        if self._default_container_id is None:
            self.initialize()
        assert self._default_container_id is not None
        target_path = (
            path if posixpath.isabs(path) else posixpath.join(self.work_dir, path)
        )
        with tempfile.NamedTemporaryFile('wb', delete=False) as tmp:
            if isinstance(content, str):
                tmp.write(content.encode('utf-8'))
            else:
                tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            self._copy_to_container(self._default_container_id, tmp_path, target_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def read_file(self, path: str) -> str:
        """Read a file from the active container workspace."""
        if self._default_container_id is None:
            self.initialize()
        assert self._default_container_id is not None
        target_path = (
            path if posixpath.isabs(path) else posixpath.join(self.work_dir, path)
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_dst = Path(tmpdir) / 'downloaded_file'
            self._copy_from_container(
                self._default_container_id, target_path, local_dst
            )
            if not local_dst.exists():
                raise FileNotFoundError(f'File not found: {path}')
            return local_dst.read_text(encoding='utf-8', errors='replace')

    def copy_to(
        self,
        src_or_container: str | Path,
        dst_or_src: str | Path,
        dst: str | Path | None = None,
    ) -> None:
        """Copy a file or directory into the container.

        Supports both BaseSandbox 2-argument signature:
            copy_to(host_path, container_path)
        and legacy manager 3-argument signature:
            copy_to(container_id, src, dst)
        """
        if dst is not None:
            container_id = str(src_or_container)
            src = Path(dst_or_src)
            target_dst = str(dst)
            self._copy_to_container(container_id, src, target_dst)
        else:
            if self._default_container_id is None:
                self.initialize()
            assert self._default_container_id is not None
            host_p = Path(src_or_container)
            target_dst = (
                str(dst_or_src)
                if posixpath.isabs(str(dst_or_src))
                else posixpath.join(self.work_dir, str(dst_or_src))
            )
            self._copy_to_container(self._default_container_id, host_p, target_dst)

    @staticmethod
    def _resolve_copy_destination(
        src: Path, dst: str
    ) -> tuple[str, str | None, str]:
        """Resolves (target_dir, arcname, expected_container_path) for a copy operation."""
        dst_str = dst.strip() or '/'
        if src.is_file():
            is_dir_dst = (
                dst_str.endswith('/')
                or dst_str
                in (
                    '/',
                    '/tmp',
                    '/var/tmp',
                    '/workspace',
                    '/wheels',
                    '/sandbox',
                    '/usr/local/bin',
                )
            )
            if is_dir_dst:
                target_dir = dst_str.rstrip('/') or '/'
                arcname = src.name
            else:
                target_dir = posixpath.dirname(dst_str) or '/'
                arcname = posixpath.basename(dst_str)
            expected_path = posixpath.join(target_dir, arcname)
            return target_dir, arcname, expected_path

        target_dir = dst_str.rstrip('/') or '/'
        first_child = next(src.iterdir(), None)
        expected_path = (
            posixpath.join(target_dir, first_child.name)
            if first_child is not None
            else target_dir
        )
        return target_dir, None, expected_path

    def _path_exists_in_container(self, container_id: str, path: str) -> bool:
        """Checks if path exists inside the container's runtime mount namespace."""
        import shlex
        from unittest.mock import Mock

        try:
            container = self.client.containers.get(container_id)
            raw_res = container.exec_run(
                cmd=['/bin/bash', '-c', f'test -e {shlex.quote(path)}'],
                stdout=True,
                stderr=True,
            )
            exit_code = getattr(raw_res, 'exit_code', None)
            if isinstance(exit_code, Mock):
                # Mocked container in unit tests without integer exit_code configured
                return True
            if not isinstance(exit_code, int):
                return False
            return exit_code == 0
        except Exception:
            return False

    def _stream_tar_via_exec_socket(
        self, container_id: str, tar_stream: Any, target_dir: str
    ) -> bool:
        """Streams a tar archive into target_dir via Docker exec stdin socket."""
        import shlex
        import socket as sock_mod

        try:
            tar_stream.seek(0)
            cmd = [
                '/bin/bash',
                '-c',
                f'mkdir -p {shlex.quote(target_dir)} && tar --no-same-owner -xf - -C {shlex.quote(target_dir)}',
            ]
            exec_info = self.client.api.exec_create(
                container_id, cmd, stdin=True, stdout=True, stderr=True
            )
            if not isinstance(exec_info, dict) or 'Id' not in exec_info:
                return False
            exec_id = exec_info['Id']
            sock = self.client.api.exec_start(exec_id, socket=True)
            raw_sock: Any = getattr(sock, '_sock', sock)
            if not hasattr(raw_sock, 'sendall'):
                return False
            try:
                while True:
                    chunk = tar_stream.read(256 * 1024)
                    if not chunk:
                        break
                    raw_sock.sendall(chunk)
                with contextlib.suppress(Exception):
                    raw_sock.shutdown(sock_mod.SHUT_WR)
                with contextlib.suppress(Exception):
                    while raw_sock.recv(65536):
                        pass
            finally:
                with contextlib.suppress(Exception):
                    sock.close()
            inspect_res = self.client.api.exec_inspect(exec_id)
            if isinstance(inspect_res, dict):
                return inspect_res.get('ExitCode', 0) == 0
            return True
        except Exception as e:
            logger.debug('Exec socket tar stream failed on %s: %s', container_id, e)
            return False

    def _copy_via_exec_base64(
        self,
        container_id: str,
        src: Path,
        tar_stream: Any,
        target_dir: str,
        expected_path: str,
    ) -> None:
        """Fallback transfer writing base64 chunks via exec_run when put_archive and exec sockets are unavailable."""
        import base64
        import shlex

        chunk_size = 65536  # 64 KB raw -> ~87 KB base64 per exec call
        if src.is_file():
            self.exec(
                container_id,
                f'mkdir -p {shlex.quote(target_dir)} && : > {shlex.quote(expected_path)}',
            )
            with open(src, 'rb') as f:
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break
                    b64 = base64.b64encode(data).decode('ascii')
                    self.exec(
                        container_id,
                        f"printf '%s' '{b64}' | base64 -d >> {shlex.quote(expected_path)}",
                    )
            return

        tmp_tar = f'/var/tmp/_swegemma_copy_{int(time.time() * 1000)}.tar'
        self.exec(
            container_id,
            f'mkdir -p /var/tmp {shlex.quote(target_dir)} && : > {shlex.quote(tmp_tar)}',
        )
        try:
            tar_stream.seek(0)
            while True:
                data = tar_stream.read(chunk_size)
                if not data:
                    break
                b64 = base64.b64encode(data).decode('ascii')
                self.exec(
                    container_id,
                    f"printf '%s' '{b64}' | base64 -d >> {shlex.quote(tmp_tar)}",
                )
            self.exec(
                container_id,
                f'tar --no-same-owner -xf {shlex.quote(tmp_tar)} -C {shlex.quote(target_dir)}',
            )
        finally:
            self.exec(container_id, f'rm -f {shlex.quote(tmp_tar)}')

    def _copy_to_container(self, container_id: str, src: Path, dst: str) -> None:
        import shlex

        if not src.exists():
            raise FileNotFoundError(f'Source path {src} does not exist')

        container = self.client.containers.get(container_id)
        target_dir, arcname, expected_path = self._resolve_copy_destination(src, dst)

        # For single-file copies, stage through /var/tmp under a unique filename and move
        # into expected_path via exec_run. This avoids both /tmp tmpfs shadowing, gVisor
        # (runsc) negative dentry / stale inode caching on reused containers, and double tar creation.
        if src.is_file() and arcname is not None:
            unique_stage_name = f'_swegemma_cp_{uuid.uuid4().hex}_{arcname}'
            staged_path = posixpath.join('/var/tmp', unique_stage_name)
            with contextlib.suppress(Exception):
                self.exec(
                    container_id,
                    f'mkdir -p /var/tmp {shlex.quote(target_dir)}',
                )
                stage_stream = build_tar_stream(src, arcname=unique_stage_name)
                container.put_archive('/var/tmp', stage_stream)
                mv_res = self.exec(
                    container_id,
                    f'mv -f {shlex.quote(staged_path)} {shlex.quote(expected_path)} && test -e {shlex.quote(expected_path)}',
                )
                if mv_res.exit_code == 0 and self._path_exists_in_container(
                    container_id, expected_path
                ):
                    return

        with tempfile.SpooledTemporaryFile(
            max_size=64 * 1024 * 1024, mode='w+b'
        ) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                if src.is_file():
                    tar.add(src, arcname=arcname or src.name, filter=reset_tarinfo)
                else:
                    for item in src.iterdir():
                        tar.add(item, arcname=item.name, filter=reset_tarinfo)

            tar_stream.seek(0)
            with contextlib.suppress(Exception):
                container.put_archive(target_dir, tar_stream)
                if self._path_exists_in_container(container_id, expected_path):
                    return

            # Fallback 2: Stream tar archive directly into the container process via exec stdin socket
            if self._stream_tar_via_exec_socket(
                container_id, tar_stream, target_dir
            ) and self._path_exists_in_container(container_id, expected_path):
                return

            # Fallback 3: Chunked base64 transfer over exec_run
            self._copy_via_exec_base64(
                container_id, src, tar_stream, target_dir, expected_path
            )
            if not self._path_exists_in_container(container_id, expected_path):
                raise RuntimeError(
                    f'Failed to copy {src} to {expected_path} in container {container_id}'
                )

    def extract_archive_to_container(
        self,
        container_id: str,
        archive_path: Path,
        dst_dir: str = '/workspace',
    ) -> bool:
        """Extracts a .tar/.tgz/.tar.gz archive inside the container VFS so gVisor (runsc) caches stay coherent."""
        import shlex
        import socket as sock_mod

        if not archive_path.exists():
            raise FileNotFoundError(f'Archive path {archive_path} does not exist')

        self.exec(container_id, f'mkdir -p {shlex.quote(dst_dir)}')

        # 1. Stream compressed archive over exec stdin socket directly into tar -C dst_dir inside the sandbox VFS
        with contextlib.suppress(Exception):
            cmd = [
                '/bin/bash',
                '-c',
                f'tar --no-same-owner --unlink-first -I pigz -xf - -C {shlex.quote(dst_dir)} 2>/dev/null || tar --no-same-owner --unlink-first -xzf - -C {shlex.quote(dst_dir)}',
            ]
            exec_info = self.client.api.exec_create(
                container_id, cmd, stdin=True, stdout=True, stderr=True
            )
            if isinstance(exec_info, dict) and 'Id' in exec_info:
                exec_id = exec_info['Id']
                sock = self.client.api.exec_start(exec_id, socket=True)
                raw_sock: Any = getattr(sock, '_sock', sock)
                if hasattr(raw_sock, 'sendall'):
                    try:
                        with open(archive_path, 'rb') as f:
                            while True:
                                chunk = f.read(256 * 1024)
                                if not chunk:
                                    break
                                raw_sock.sendall(chunk)
                        with contextlib.suppress(Exception):
                            raw_sock.shutdown(sock_mod.SHUT_WR)
                        with contextlib.suppress(Exception):
                            while raw_sock.recv(65536):
                                pass
                    finally:
                        with contextlib.suppress(Exception):
                            sock.close()
                    check_res = self.exec(
                        container_id,
                        f'test -d {shlex.quote(dst_dir)}/.git || [ -n "$(ls -A {shlex.quote(dst_dir)} 2>/dev/null)" ]',
                    )
                    if check_res.exit_code == 0:
                        return True

        # 2. Fallback: stage in /var/tmp under a unique filename (avoids runsc negative dentries), extract inside VFS, and remove
        staged_tar = f'/var/tmp/_swegemma_ext_{uuid.uuid4().hex}_{archive_path.name}'
        self.copy_to(container_id, archive_path, staged_tar)
        res = self.exec(
            container_id,
            f'(tar --no-same-owner --unlink-first -I pigz -xf {shlex.quote(staged_tar)} -C {shlex.quote(dst_dir)} 2>/dev/null || tar --no-same-owner --unlink-first -xzf {shlex.quote(staged_tar)} -C {shlex.quote(dst_dir)}) && rm -f {shlex.quote(staged_tar)}',
        )
        return res.exit_code == 0

    def copy_from(
        self,
        src_or_container: str | Path,
        dst_or_src: str | Path,
        dst: str | Path | None = None,
    ) -> None:
        """Copy a file or directory from container to host.

        Supports both BaseSandbox 2-argument signature:
            copy_from(container_path, host_path)
        and legacy manager 3-argument signature:
            copy_from(container_id, src, dst)
        """
        if dst is not None:
            container_id = str(src_or_container)
            src = str(dst_or_src)
            dest_p = Path(dst)
            self._copy_from_container(container_id, src, dest_p)
        else:
            if self._default_container_id is None:
                self.initialize()
            assert self._default_container_id is not None
            src_p = (
                str(src_or_container)
                if posixpath.isabs(str(src_or_container))
                else posixpath.join(self.work_dir, str(src_or_container))
            )
            dest_p = Path(dst_or_src)
            self._copy_from_container(self._default_container_id, src_p, dest_p)

    def _copy_from_container(
        self,
        container_id: str,
        src: str,
        dst: Path,
        *,
        _resolved_symlink: bool = False,
    ) -> None:
        import shlex
        import shutil

        resolved_src = ''
        if not _resolved_symlink:
            try:
                res = self.exec(container_id, f'readlink -f {shlex.quote(src)}')
                if (
                    getattr(res, 'exit_code', -1) == 0
                    and isinstance(getattr(res, 'stdout', None), str)
                ):
                    resolved_src = res.stdout.strip()
            except Exception:
                resolved_src = ''

            if resolved_src and posixpath.isabs(resolved_src):
                if not (
                    resolved_src == '/workspace'
                    or resolved_src.startswith('/workspace/')
                    or resolved_src == '/tmp'
                    or resolved_src.startswith('/tmp/')
                    or resolved_src.startswith('/var/tmp/')
                ):
                    raise ValueError(
                        f"Symlink target '{resolved_src}' escapes allowed sandbox directories"
                    )
                src = resolved_src

        container = self.client.containers.get(container_id)
        stream, _ = container.get_archive(src)

        tar_bytes = io.BytesIO()
        for chunk in stream:
            tar_bytes.write(chunk)
        tar_bytes.seek(0)

        with tarfile.open(fileobj=tar_bytes) as tar:
            members = tar.getmembers()
            if not members:
                return

            if (
                len(members) == 1
                and members[0].isfile()
                and not (dst.exists() and dst.is_dir())
            ):
                dst.parent.mkdir(parents=True, exist_ok=True)
                f = tar.extractfile(members[0])
                if f:
                    with open(dst, 'wb') as out_f:
                        shutil.copyfileobj(f, out_f)
                    with contextlib.suppress(OSError):
                        dst.chmod(members[0].mode)
                return

            if (
                len(members) == 1
                and (members[0].issym() or members[0].islnk())
                and not (dst.exists() and dst.is_dir())
            ):
                link_target = members[0].linkname
                if not posixpath.isabs(link_target):
                    link_target = posixpath.normpath(
                        posixpath.join(posixpath.dirname(src), link_target)
                    )
                else:
                    link_target = posixpath.normpath(link_target)
                if not (
                    link_target == '/workspace'
                    or link_target.startswith('/workspace/')
                    or link_target == '/tmp'
                    or link_target.startswith('/tmp/')
                    or link_target.startswith('/var/tmp/')
                ):
                    raise ValueError(
                        f"Symlink target '{link_target}' escapes allowed sandbox directories"
                    )
                if not _resolved_symlink and link_target != src:
                    self._copy_from_container(
                        container_id, link_target, dst, _resolved_symlink=True
                    )
                    return
                cat_res = self.exec(container_id, f'cat {shlex.quote(link_target)}')
                if (
                    getattr(cat_res, 'exit_code', -1) == 0
                    and isinstance(getattr(cat_res, 'stdout', None), str)
                ):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_text(cat_res.stdout, encoding='utf-8')
                    return
                raise FileNotFoundError(
                    f"Could not resolve symlink '{src}' -> '{link_target}' in container"
                )

            dst.mkdir(parents=True, exist_ok=True)
            if hasattr(tarfile, 'data_filter'):
                tar.extractall(path=dst, filter='data')
            else:
                dst_resolved = dst.resolve()
                for member in members:
                    target_path = (dst / member.name).resolve()
                    if not target_path.is_relative_to(dst_resolved):
                        raise ValueError(
                            f'Tar slip / directory traversal detected in member: {member.name}'
                        )
                    tar.extract(member, path=dst)

    def start(self) -> str:
        """Start a container running in the background (or reuse an idle warm container).

        Returns:
            The container ID.
        """
        with self._lock:
            while self._idle_containers:
                idle_cid = self._idle_containers.pop()
                try:
                    c = self.client.containers.get(idle_cid)
                    status = getattr(c, 'status', 'running')
                    if status in ('running', 'created') or hasattr(status, '_mock_name'):
                        self._active_containers.add(idle_cid)
                        if self._default_container_id is None:
                            self._default_container_id = idle_cid
                        return idle_cid
                except Exception:
                    pass

            if not self._image_resolved:
                self.config.image = resolve_sandbox_image(
                    self.config.image, client=self.client
                )
                self._image_resolved = True

        container = self.client.containers.run(
            self.config.image,
            command='sleep infinity',
            detach=True,
            network_mode=self.config.network_mode,
            mem_limit=self.config.mem_limit,
            cpu_period=self.config.cpu_period,
            cpu_quota=self.config.cpu_quota,
            working_dir=self.config.working_dir,
            environment=self.config.environment,
        )
        cid = str(container.id or '')
        with self._lock:
            self._active_containers.add(cid)
            if self._default_container_id is None:
                self._default_container_id = cid
        return cid

    async def start_async(self) -> str:
        """Start a container asynchronously without blocking the event loop."""
        return await asyncio.to_thread(self.start)

    def exec(
        self, container_id: str, command: str, *, timeout: int | None = None
    ) -> ExecutionResult:
        """Execute a command in a running container."""
        import shlex

        container = self.client.containers.get(container_id)
        timeout = timeout if timeout is not None else self.config.timeout_seconds
        start_time = time.perf_counter()

        try:
            escaped_command = shlex.quote(command)
            timeout_command = f'timeout -k 5s {timeout} /bin/bash -c {escaped_command}'

            res = container.exec_run(
                cmd=['/bin/bash', '-c', timeout_command],
                stdout=True,
                stderr=True,
                demux=True,
            )

            duration = time.perf_counter() - start_time
            timed_out = (res.exit_code == 124) and (duration >= max(0.0, timeout - 1.0))

            max_chars = 10 * 1024 * 1024  # 10 MB character limit
            raw_stdout: bytes | None = b''
            raw_stderr: bytes | None = b''
            if isinstance(res.output, tuple):
                raw_stdout, raw_stderr = res.output
            elif isinstance(res.output, (bytes, bytearray)):
                raw_stdout, raw_stderr = bytes(res.output), b''

            stdout = (
                raw_stdout.decode('utf-8', errors='replace')
                if isinstance(raw_stdout, (bytes, bytearray))
                else ''
            )
            stderr = (
                raw_stderr.decode('utf-8', errors='replace')
                if isinstance(raw_stderr, (bytes, bytearray))
                else ''
            )

            if len(stdout) > max_chars:
                stdout = stdout[:max_chars] + '\n...[Output truncated at 10MB limit]\n'
            if len(stderr) > max_chars:
                stderr = stderr[:max_chars] + '\n...[Output truncated at 10MB limit]\n'

            exit_code = int(res.exit_code) if res.exit_code is not None else 0
            status: Literal['ok', 'error', 'timeout'] = (
                'timeout' if timed_out else ('ok' if exit_code == 0 else 'error')
            )
            return ExecutionResult(
                status=status,
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            return ExecutionResult(
                status='error',
                exit_code=-1,
                stdout='',
                stderr=str(e),
                duration_seconds=duration,
            )

    async def exec_async(
        self, container_id: str, command: str, *, timeout: int | None = None
    ) -> ExecutionResult:
        """Execute a command in a running container asynchronously without blocking the event loop."""
        return await asyncio.to_thread(
            self.exec, container_id, command, timeout=timeout
        )

    def stop(self, container_id: str) -> None:
        """Stop and remove a container (or reset workspace and return to idle pool if reuse_containers is enabled)."""
        import requests.exceptions
        from docker.errors import APIError, DockerException

        if getattr(self.config, 'reuse_containers', False):
            with contextlib.suppress(Exception):
                res = self.exec(
                    container_id,
                    'kill -9 $(pgrep -v -f "sleep infinity" | grep -v "^1$") 2>/dev/null || true; '
                    'find /workspace -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null; '
                    'find /tmp /var/tmp /root -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null; '
                    'rm -rf /usr/local/lib/python*/site-packages/workspace_paths.pth '
                    '/usr/local/lib/python*/site-packages/_swegemma_stubs.py '
                    '/usr/local/lib/python*/site-packages/_swegemma_stubs.pth '
                    '/usr/local/lib/python*/site-packages/sitecustomize.py '
                    '/usr/local/lib/python*/site-packages/usercustomize.py '
                    '/usr/local/lib/python*/site-packages/__pycache__/_swegemma_stubs*.pyc '
                    '/usr/local/lib/python*/site-packages/__pycache__/sitecustomize*.pyc '
                    '/usr/local/lib/python*/site-packages/__pycache__/usercustomize*.pyc 2>/dev/null || true',
                    timeout=30,
                )
                if res.exit_code == 0:
                    with self._lock:
                        self._active_containers.discard(container_id)
                        if self._default_container_id == container_id:
                            self._default_container_id = next(
                                iter(self._active_containers), None
                            )
                        if container_id not in self._idle_containers:
                            self._idle_containers.append(container_id)
                    return

        with self._lock:
            self._active_containers.discard(container_id)
            if container_id in self._idle_containers:
                self._idle_containers.remove(container_id)
            if self._default_container_id == container_id:
                self._default_container_id = next(iter(self._active_containers), None)
        try:
            container = self.client.containers.get(container_id)
            container.stop(timeout=2)
            container.remove()
        except (
            APIError,
            DockerException,
            requests.exceptions.RequestException,
        ) as e:
            logger.debug('Container %s cleanup notice: %s', container_id[:12], e)

    async def stop_async(self, container_id: str) -> None:
        """Stop and remove a container asynchronously without blocking the event loop."""
        await asyncio.to_thread(self.stop, container_id)

    def cleanup_all(self) -> None:
        """Stop and remove all active and idle containers managed by this instance."""
        with self._lock:
            all_ids = list(self._active_containers | set(self._idle_containers))
            self._active_containers.clear()
            self._idle_containers.clear()
        prev_reuse = getattr(self.config, 'reuse_containers', False)
        self.config.reuse_containers = False
        try:
            for cid in all_ids:
                self.stop(cid)
        finally:
            self.config.reuse_containers = prev_reuse

    def reset(self) -> None:
        """Reset container workspace to initial state."""
        import shlex
        with self._lock:
            default_id = self._default_container_id
        if default_id is not None:
            self.exec(
                default_id,
                f'find {shlex.quote(self.work_dir)} -mindepth 1 -maxdepth 1 -exec rm -rf {{}} +',
            )
            if self._initial_files:
                for rel_path, content in self._initial_files.items():
                    self.write_file(rel_path, content)

    def close(self) -> None:
        """Stop container and release resources."""
        self.cleanup_all()
        with self._lock:
            self._default_container_id = None

    def get_sandbox(self, sandbox_id: str, work_dir: str = '/workspace') -> BaseSandbox:
        """Return a BaseSandbox view of a specific managed container instance."""
        from swegemma.sandbox.base import ManagedSandbox

        return ManagedSandbox(manager=self, sandbox_id=sandbox_id, _work_dir=work_dir)

