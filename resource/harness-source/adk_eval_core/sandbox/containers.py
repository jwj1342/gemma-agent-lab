"""Low-level Docker container management.

Provides a context-manager wrapper around the Docker SDK for running
commands and transferring files via tarball archives.
"""

from __future__ import annotations

import contextlib
import io
import logging
import posixpath
import shlex
import tarfile
import time
import types
from typing import Any, Self

import docker
import docker.errors

from adk_eval_core.errors import (
    SandboxExecutionError,
    SandboxFileNotFoundError,
    SandboxIsADirectoryError,
    SandboxNotStartedError,
    SandboxTimeoutError,
)

logger = logging.getLogger(__name__)

_MAX_READ_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_RUN_READ_BYTES = 10 * 1024 * 1024  # 10 MB


class DockerContainer:
    """Manages a single Docker container lifecycle.

    Usage::

        with DockerContainer("gcr.io/kaggle-images/python") as c:
            exit_code, stdout, stderr = c.run_command("echo hello")
            c.write_file("/work/data.csv", "a,b\\n1,2")
            content = c.read_file("/work/data.csv")

    Attributes:
        image: The Docker image string.
        client: The Docker client from the environment.
        container: The underlying docker container object.
    """

    def __init__(
        self,
        image: str,
        work_dir: str | None = None,
        environment: dict[str, str] | None = None,
        mem_limit: str | None = None,
        cpu_period: int = 100_000,
        cpu_quota: int = 200_000,
        **kwargs: Any,
    ) -> None:
        """Initialize the DockerContainer.

        Args:
            image: The Docker image to run.
            work_dir: Working directory inside container.
            environment: Environment variables.
            mem_limit: Memory limit.
            cpu_period: CPU period.
            cpu_quota: CPU quota.
            **kwargs: Extra arguments for forward compatibility.
        """
        self.image = image
        self.work_dir = work_dir if work_dir is not None else "/work"
        self.environment = environment or {}
        self.mem_limit = mem_limit if mem_limit is not None else "4g"
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota
        self.extra_kwargs = kwargs
        self.client = docker.from_env()
        self.container: Any | None = None

    def start(self) -> Self:
        """Start container if not started."""
        if self.container is not None:
            return self
        return self.__enter__()

    def stop(self) -> None:
        """Stop and remove container if started."""
        if self.container is not None:
            self.__exit__(None, None, None)

    def __enter__(self) -> Self:
        logger.debug("Checking for image %s...", self.image)
        try:
            self.client.images.get(self.image)
            logger.debug("Image %s found locally.", self.image)
        except docker.errors.ImageNotFound:
            logger.info(
                "Image %s not found locally. Pulling (this may take a while)...",
                self.image,
            )
            self.client.images.pull(self.image)
            logger.info("Image %s pulled successfully.", self.image)

        logger.debug("Creating container from %s...", self.image)
        run_kwargs: dict[str, Any] = {
            "command": "tail -f /dev/null",
            "detach": True,
            "tty": True,
            "network_disabled": True,
            "environment": self.environment,
        }
        if self.mem_limit:
            run_kwargs["mem_limit"] = self.mem_limit
        if self.cpu_period:
            run_kwargs["cpu_period"] = self.cpu_period
        if self.cpu_quota:
            run_kwargs["cpu_quota"] = self.cpu_quota
        run_kwargs.update(self.extra_kwargs)

        container: Any = self.client.containers.run(
            self.image,
            **run_kwargs,
        )
        self.container = container
        logger.debug(
            "Container %s created, waiting for it to start...", container.short_id
        )

        start_time = time.time()
        while container.status != "running":
            container.reload()
            if time.time() - start_time > 10:
                with contextlib.suppress(Exception):
                    container.stop()
                with contextlib.suppress(Exception):
                    container.remove()
                raise SandboxTimeoutError("Container failed to start within 10 seconds", timeout=10.0)
            time.sleep(0.1)

        logger.debug("Container %s is running.", container.short_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self.container:
            with contextlib.suppress(docker.errors.APIError, Exception):
                self.container.stop(timeout=1)
            with contextlib.suppress(docker.errors.APIError, Exception):
                self.container.remove(force=True)
            self.container = None

    def run_command(
        self,
        command: str,
        timeout: float | None = 300,
        workdir: str | None = None,
        work_dir: str | None = None,
    ) -> tuple[int, str, str]:
        """Run a command and return (exit_code, stdout, stderr).

        Args:
            command: Shell command to execute.
            timeout: Optional execution timeout.
            workdir: Working directory inside the container.
            work_dir: Alias for workdir.

        Returns:
            Tuple of (exit_code, decoded_stdout, decoded_stderr).

        Raises:
            RuntimeError: If the container is not started.
        """
        if not self.container:
            raise SandboxNotStartedError("Container not started")

        effective_workdir = workdir or work_dir or self.work_dir
        effective_command = (
            f"timeout -k 5s {max(1, int(timeout))} sh -c {shlex.quote(command)}"
            if timeout is not None and timeout > 0
            else command
        )

        exec_id = None
        if hasattr(self.client, "api"):
            original_exec_create = self.client.api.exec_create

            def hook(*args: Any, **kwargs: Any) -> Any:
                nonlocal exec_id
                res = original_exec_create(*args, **kwargs)
                exec_id = res.get("Id")
                return res

            self.client.api.exec_create = hook
            try:
                res: Any = self.container.exec_run(
                    ["sh", "-c", effective_command],
                    workdir=effective_workdir,
                    stream=True,
                    demux=True,
                )
                exit_code, output = res
            finally:
                self.client.api.exec_create = original_exec_create
        else:
            res: Any = self.container.exec_run(
                ["sh", "-c", effective_command],
                workdir=effective_workdir,
                stream=True,
                demux=True,
            )
            exit_code, output = res

        stdout_bytes = io.BytesIO()
        stderr_bytes = io.BytesIO()
        total_bytes = 0
        limit_exceeded = False

        for out_chunk, err_chunk in output:
            if out_chunk:
                if total_bytes + len(out_chunk) > _MAX_RUN_READ_BYTES:
                    allowed = _MAX_RUN_READ_BYTES - total_bytes
                    if allowed > 0:
                        stdout_bytes.write(out_chunk[:allowed])
                        total_bytes += allowed
                    limit_exceeded = True
                    break
                stdout_bytes.write(out_chunk)
                total_bytes += len(out_chunk)
            if err_chunk:
                if total_bytes + len(err_chunk) > _MAX_RUN_READ_BYTES:
                    allowed = _MAX_RUN_READ_BYTES - total_bytes
                    if allowed > 0:
                        stderr_bytes.write(err_chunk[:allowed])
                        total_bytes += allowed
                    limit_exceeded = True
                    break
                stderr_bytes.write(err_chunk)
                total_bytes += len(err_chunk)

        if limit_exceeded:
            logger.warning(
                "Command output exceeded maximum read size of %d bytes. Truncating.",
                _MAX_RUN_READ_BYTES,
            )

        if exit_code is None and exec_id is not None:
            for _ in range(10):
                inspect_data = self.client.api.exec_inspect(exec_id)
                if inspect_data.get("Running") is False:
                    exit_code = inspect_data.get("ExitCode")
                    break
                time.sleep(0.1)
            if exit_code is None:
                exit_code = -1

        if exit_code is None:
            exit_code = 0

        stdout_str = stdout_bytes.getvalue().decode("utf-8", errors="replace").rstrip("\r\n")
        stderr_str = stderr_bytes.getvalue().decode("utf-8", errors="replace").rstrip("\r\n")

        if exit_code in (124, 137) and timeout is not None and timeout > 0:
            raise SandboxTimeoutError(
                f"Command timed out after {timeout}s",
                command=command,
                timeout=float(timeout),
                stdout=stdout_str,
                stderr=stderr_str,
            )

        return exit_code, stdout_str, stderr_str

    def write_file(self, path: str, content: str | bytes) -> None:
        """Write content to a file inside the container.

        Uses tarball transfer since bind mounts are not available.

        Args:
            path: Absolute path inside the container.
            content: File content as string (UTF-8 encoded) or bytes.

        Raises:
            RuntimeError: If the container is not started.
        """
        if not self.container:
            raise SandboxNotStartedError("Container not started")

        dir_name = posixpath.dirname(path)
        file_name = posixpath.basename(path)

        if not dir_name:
            dir_name = "/"

        exit_code, output = self.container.exec_run(f"mkdir -p {shlex.quote(dir_name)}")
        if exit_code != 0:
            err_msg = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
            raise SandboxExecutionError(
                f"Failed to create directory {dir_name}: {err_msg}",
                exit_code=exit_code,
                command=f"mkdir -p {shlex.quote(dir_name)}",
                stderr=err_msg,
            )

        if isinstance(content, str):
            data = content.encode("utf-8")
        else:
            data = content

        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(data)
            tarinfo.mtime = int(time.time())
            tar.addfile(tarinfo, io.BytesIO(data))

        tar_stream.seek(0)
        self.container.put_archive(dir_name, tar_stream)

    def read_file(self, path: str) -> str:
        """Read a file from the container and return it as a string.

        Args:
            path: Absolute path inside the container.

        Returns:
            File content as a UTF-8 string.

        Raises:
            FileNotFoundError: If the file does not exist.
            RuntimeError: If the container is not started or on other read errors.
        """
        if not self.container:
            raise SandboxNotStartedError("Container not started")

        try:
            stream, _ = self.container.get_archive(path)

            file_obj = io.BytesIO()
            total_read = 0
            for chunk in stream:
                total_read += len(chunk)
                if total_read > _MAX_READ_BYTES:
                    raise SandboxExecutionError(
                        f"File {path} exceeds maximum read size of "
                        f"{_MAX_READ_BYTES // (1024 * 1024)} MB"
                    )
                file_obj.write(chunk)
            file_obj.seek(0)

            with tarfile.open(fileobj=file_obj, mode="r") as tar:
                members = tar.getmembers()
                if not members:
                    raise SandboxExecutionError(f"File {path} is empty or invalid tar.")

                member = members[0]
                if member.isdir():
                    raise SandboxIsADirectoryError(f"{path} is a directory", path=path)
                f = tar.extractfile(member)
                if f is None:
                    raise SandboxExecutionError(f"Could not extract file {path}.")

                return f.read().decode("utf-8", errors="replace")

        except docker.errors.NotFound:
            raise SandboxFileNotFoundError(f"File not found: {path}", path=path)
        except (FileNotFoundError, IsADirectoryError, RuntimeError):
            raise
        except Exception as e:
            raise SandboxExecutionError(f"Error reading file {path}: {e}") from e

    def copy_to(self, host_path: str | Any, container_path: str) -> None:
        """Copy a file or directory tree from host into container with /var/tmp staging fallback."""
        import uuid
        from pathlib import Path

        if not self.container:
            raise SandboxNotStartedError("Container not started")

        host_p = Path(host_path).resolve()
        if not host_p.exists():
            raise SandboxFileNotFoundError(
                f"Host path not found: {host_path}", path=str(host_path)
            )

        work_dir = getattr(self, "work_dir", None)
        if not isinstance(work_dir, str) or not work_dir:
            work_dir = "/work"
        norm_work = posixpath.normpath(work_dir)
        norm_container = posixpath.normpath(container_path)
        if host_p.is_file():
            if container_path.endswith("/") or norm_container in (norm_work, "/tmp", "/var/tmp", "/"):
                arcname = host_p.name
                dest_dir = norm_container
                container_path = posixpath.join(norm_container, host_p.name)
            else:
                arcname = posixpath.basename(container_path) or host_p.name
                dest_dir = posixpath.dirname(container_path) or work_dir
        else:
            arcname = None
            dest_dir = container_path

        tar_stream = build_tar_stream(host_p, arcname=arcname)
        self.run_command(f"mkdir -p {shlex.quote(dest_dir)}")
        try:
            self.container.put_archive(dest_dir, tar_stream)
        except Exception:
            if host_p.is_file() and arcname:
                unique_stage_name = f"_adk_cp_{uuid.uuid4().hex}_{arcname}"
                staged_path = posixpath.join("/var/tmp", unique_stage_name)
                stage_stream = build_tar_stream(host_p, arcname=unique_stage_name)
                self.run_command(f"mkdir -p /var/tmp {shlex.quote(dest_dir)}")
                self.container.put_archive("/var/tmp", stage_stream)
                self.run_command(
                    f"mv -f {shlex.quote(staged_path)} {shlex.quote(container_path)}"
                )
            else:
                raise

    def copy_from(self, container_path: str, host_path: str | Any) -> None:
        """Copy a file or directory tree from container to host with Tar Slip protection."""
        import copy
        import shutil
        from pathlib import Path

        from adk_eval_core.errors import SandboxPathTraversalError

        if not self.container:
            raise SandboxNotStartedError("Container not started")

        host_p = Path(host_path).resolve()

        try:
            stream, _stat = self.container.get_archive(container_path)
        except docker.errors.NotFound:
            raise SandboxFileNotFoundError(
                f"Container path not found: {container_path}", path=str(container_path)
            )

        tar_bytes = io.BytesIO()
        total_read = 0
        for chunk in stream:
            total_read += len(chunk)
            if total_read > _MAX_READ_BYTES:
                raise SandboxExecutionError(
                    f"Archive {container_path} exceeds maximum read size of "
                    f"{_MAX_READ_BYTES // (1024 * 1024)} MB"
                )
            tar_bytes.write(chunk)
        tar_bytes.seek(0)

        with tarfile.open(fileobj=tar_bytes, mode="r") as tar:
            members = tar.getmembers()
            if not members:
                raise SandboxFileNotFoundError(
                    f"Container archive empty: {container_path}",
                    path=str(container_path),
                )

            for member in members:
                if (
                    member.name.startswith(("/", "\\"))
                    or ".." in member.name.replace("\\", "/").split("/")
                ):
                    raise SandboxPathTraversalError(
                        f"Tar slip / path traversal detected in member: {member.name}",
                        path=member.name,
                    )
                if member.issym() or member.islnk():
                    linkname = member.linkname
                    if (
                        linkname.startswith(("/", "\\"))
                        or ".." in linkname.replace("\\", "/").split("/")
                    ):
                        raise SandboxPathTraversalError(
                            f"Tar slip / path traversal detected in member linkname: {member.linkname}",
                            path=member.linkname,
                        )

            # Single regular file extraction
            if len(members) == 1 and members[0].isfile():
                if str(host_path).endswith(("/", "\\")) or (host_p.exists() and host_p.is_dir()):
                    host_p.mkdir(parents=True, exist_ok=True)
                    target_file = host_p / (posixpath.basename(container_path.rstrip("/")) or members[0].name)
                else:
                    host_p.parent.mkdir(parents=True, exist_ok=True)
                    target_file = host_p
                f = tar.extractfile(members[0])
                if f is not None:
                    with open(target_file, "wb") as out_f:
                        shutil.copyfileobj(f, out_f)
                return

            # Directory extraction with traversal defense
            host_p.mkdir(parents=True, exist_ok=True)
            host_p_resolved = host_p.resolve()
            arc_root = posixpath.basename(container_path.rstrip("/"))
            strip_prefix = bool(
                arc_root
                and all(
                    m.name.rstrip("/") == arc_root or m.name.startswith(arc_root + "/")
                    for m in members
                )
            )
            for member in members:
                rel_name = member.name
                if strip_prefix:
                    if rel_name.rstrip("/") == arc_root:
                        continue
                    if rel_name.startswith(arc_root + "/"):
                        rel_name = rel_name[len(arc_root) + 1 :]
                if not rel_name:
                    continue
                target_member = (host_p / rel_name).resolve()
                if not target_member.is_relative_to(host_p_resolved):
                    raise SandboxPathTraversalError(
                        f"Tar slip / path traversal detected in member: {member.name}",
                        path=member.name,
                    )
                if member.issym() or member.islnk():
                    link_target = (target_member.parent / member.linkname).resolve()
                    if not link_target.is_relative_to(host_p_resolved):
                        raise SandboxPathTraversalError(
                            f"Tar slip / path traversal detected in member linkname: {member.linkname}",
                            path=member.linkname,
                        )
                extracted_member = copy.copy(member)
                extracted_member.name = rel_name
                tar.extract(extracted_member, path=host_p, filter="data")


def reset_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    """Normalize TarInfo ownership metadata to root:root (0:0)."""
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = "root"
    tarinfo.gname = "root"
    return tarinfo


def build_tar_stream(
    host_path: str | Any,
    arcname: str | None = None,
) -> io.BytesIO:
    """Package a host file or directory contents into an in-memory tar stream."""
    from pathlib import Path

    host_p = Path(host_path).resolve()
    if not host_p.exists():
        raise SandboxFileNotFoundError(
            f"Host path not found: {host_path}", path=str(host_path)
        )

    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        if host_p.is_file():
            eff_arcname = arcname or host_p.name
            tar.add(host_p, arcname=eff_arcname, filter=reset_tarinfo)
        else:
            for item in host_p.iterdir():
                tar.add(item, arcname=item.name, filter=reset_tarinfo)
    tar_stream.seek(0)
    return tar_stream


__all__ = ["DockerContainer", "build_tar_stream", "reset_tarinfo"]
