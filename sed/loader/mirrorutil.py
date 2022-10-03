"""
module sed.loader.mirrorutil, code for transparently mirroring file system trees to a
second (local) location.
Mostly ported from https://github.com/mpes-kit/mpes.
@author: L. Rettig
"""
import errno
import os
import shutil
from datetime import datetime

import dask as d
from dask.diagnostics import ProgressBar


class CopyTool:
    """File collecting and sorting class."""

    def __init__(self, source="/", dest="/", ntasks=None, **kwds):

        self.source = source
        self.dest = dest
        self.safety_margin = kwds.pop(
            "safetyMargin",
            1 * 2**30,
        )  # Default 500 GB safety margin
        self.pbenv = kwds.pop("pbenv", "classic")
        self.gid = kwds.pop("gid", 5050)

        if (ntasks is None) or (ntasks < 0):
            # Default to 25 concurrent copy tasks
            self.ntasks = 25
        else:
            self.ntasks = int(ntasks)

    def copy(  # pylint: disable=R0912, R0914
        self,
        source,
        force_copy=False,
        scheduler="threads",
        **compute_kwds,
    ):
        """Local file copying method."""

        if not os.path.exists(source):
            raise FileNotFoundError("Source not found!")

        filenames = []
        dirnames = []

        if os.path.isfile(source):
            # Single file
            sdir = os.path.dirname(os.path.realpath(source))
            ddir = get_target_dir(
                sdir,
                self.source,
                self.dest,
                gid=self.gid,
                mode=0o775,
                create=True,
            )
            filenames.append(source)

        elif os.path.isdir(source):
            sdir = source
            ddir = get_target_dir(
                sdir,
                self.source,
                self.dest,
                gid=self.gid,
                mode=0o775,
                create=True,
            )
            # dirs.append(sdir)
            for path, dirs, files in os.walk(sdir):
                for file in files:
                    filenames.append(os.path.join(path, file))
                for directory in dirs:
                    dirnames.append(os.path.join(path, directory))

        if not filenames:
            raise FileNotFoundError("No files found at path!")

        # actual copy loop
        # Check space left
        size_src = 0
        size_dst = 0
        for sfile in filenames:
            size_src += os.path.getsize(sfile)
            if os.path.exists(sfile.replace(sdir, ddir)):
                size_dst += os.path.getsize(sfile.replace(sdir, ddir))
        if size_src == 0 and not force_copy:
            # nothing to copy, just return directory
            return ddir
        free = shutil.disk_usage(ddir).free
        if size_src - size_dst > free - self.safety_margin:
            raise OSError(
                errno.ENOSPC,
                "Target disk full, only "
                + str(free / 2**30)
                + " GB free, but "
                + str((size_src - size_dst) / 2**30)
                + " GB needed!",
            )

        # make directories
        for directory in dirnames:
            dest_dir = directory.replace(sdir, ddir)
            mymakedirs(dest_dir, gid=self.gid, mode=0o775)

        copy_tasks = []  # Core-level jobs
        for src_file in filenames:
            dest_file = src_file.replace(sdir, ddir)
            size_src = os.path.getsize(src_file)
            if os.path.exists(dest_file):
                size_dst = os.path.getsize(dest_file)
            else:
                size_dst = 0
            if (
                not os.path.exists(dest_file)
                or size_dst != size_src
                or force_copy
            ):
                if os.path.exists(dest_file):
                    # delete existing file, to fix permission issue
                    copy_tasks.append(
                        d.delayed(mycopy)(
                            src_file,
                            dest_file,
                            gid=self.gid,
                            mode=0o664,
                            replace=True,
                        ),
                    )
                else:
                    copy_tasks.append(
                        d.delayed(mycopy)(
                            src_file,
                            dest_file,
                            gid=self.gid,
                            mode=0o664,
                        ),
                    )

        # run the copy tasks
        if len(copy_tasks) > 0:
            print("Copy Files...")
            with ProgressBar():
                d.compute(
                    *copy_tasks,
                    scheduler=scheduler,
                    num_workers=self.ntasks,
                    **compute_kwds,
                )
            print("Copy finished!")

        if os.path.isdir(source):
            return ddir

        return dest_file

    def size(self, sdir):
        """Calculate file size."""

        for path, dirs, filenames in os.walk(  # pylint: disable=W0612
            sdir,
        ):
            # Check space left
            size = 0
            for sfile in filenames:
                size += os.path.getsize(os.path.join(sdir, sfile))
            return size

    def cleanup_oldest_scan(  # pylint: disable=R0912
        self,
        force=False,
        report=False,
    ):
        """Remove scans in old directories."""

        # get list of all Scan directories (leaf directories)
        scan_dirs = []
        for root, dirs, files in os.walk(  # pylint: disable=W0612
            self.dest,
        ):
            if not dirs:
                scan_dirs.append(root)

        scan_dirs = sorted(scan_dirs, key=os.path.getctime)
        if report:
            print(
                "Last accessed                                Size          Path",
            )
            total_size = 0
            for scan in scan_dirs:
                size = 0
                for path, dirs, filenames in os.walk(  # pylint: disable=W0612
                    scan,
                ):
                    for sfile in filenames:
                        size += os.path.getsize(os.path.join(scan, sfile))
                total_size += size
                if size > 0:
                    print(
                        f"{datetime.fromtimestamp(os.path.getctime(scan))},        ",
                        f"{(size/2**30):.2f} GB,     {scan}",
                    )
            print(f"Total size: {(total_size/2**30):.2f} GB.")
        oldest_scan = None
        for scan in scan_dirs:
            size = 0
            for path, dirs, filenames in os.walk(  # pylint: disable=W0612
                scan,
            ):
                for sfile in filenames:
                    size += os.path.getsize(os.path.join(scan, sfile))
            if size > 0:
                oldest_scan = scan
                break
        if oldest_scan is None:
            raise FileNotFoundError("No scan with data found to remove!")

        print(
            f'Oldest scan is "{oldest_scan}", removing it will free ',
            f"{(size/2**30):.2f} GB space.",
        )
        if force:
            proceed = "y"
        else:
            print("Proceed (y/n)?")
            proceed = input()
        if proceed == "y":
            shutil.rmtree(oldest_scan)
            print("Removed sucessfully!")
        else:
            print("Aborted.")


# private Functions
def get_target_dir(
    sdir,
    source,
    dest,
    gid,
    mode,
    create=False,
):  # pylint: disable=R0913
    """Retrieve target directories."""

    if not os.path.isdir(sdir):
        raise NotADirectoryError("Only works for directories!")

    dirs = []
    head, tail = os.path.split(sdir)
    dirs.append(tail)
    while not os.path.samefile(head, source):
        if os.path.samefile(head, "/"):
            raise ValueError("sdir needs to be inside of source!")

        head, tail = os.path.split(head)
        dirs.append(tail)

    dirs.reverse()
    ddir = dest
    for directory in dirs:
        ddir = os.path.join(ddir, directory)
        if create and not os.path.exists(ddir):
            mymakedirs(ddir, mode, gid)
    return ddir


# replacement for os.makedirs, which is independent of umask
def mymakedirs(path, mode, gid):
    """Create directories."""

    if not path or os.path.exists(path):
        return []
    (head, tail) = os.path.split(path)  # pylint: disable=W0612
    res = mymakedirs(head, mode, gid)
    os.mkdir(path)
    os.chmod(path, mode)
    os.chown(path, -1, gid)
    res += [path]
    return res


def mycopy(source, dest, gid, mode, replace=False):
    """Copy function with option to delete the target file firs (to take ownership)."""

    if replace:
        if os.path.exists(dest):
            os.remove(dest)
    shutil.copy2(source, dest)
    # fix permissions and group ownership:
    os.chown(dest, -1, gid)
    os.chmod(dest, mode)
