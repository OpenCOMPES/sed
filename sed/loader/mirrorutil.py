"""
module sed.loader.mirrorutil, code for transparently mirroring file system trees to a
second (local) location. This is speeds up binning of data stored on network drives
tremendiously.
Mostly ported from https://github.com/mpes-kit/mpes.
@author: L. Rettig
"""
import errno
import os
import shutil
from datetime import datetime
from typing import List

import dask as d
from dask.diagnostics import ProgressBar


class CopyTool:
    """File collecting and sorting class.

    Args:
        source (str): Dource path for the copy tool.
        dest (str): Destination path for the copy tool.
    """

    def __init__(
        self,
        source: str,
        dest: str,
        **kwds,
    ):
        self.source = source
        self.dest = dest
        self.safety_margin = kwds.pop(
            "safetyMargin",
            1 * 2**30,
        )  # Default 500 GB safety margin
        self.gid = kwds.pop("gid", 5050)
        self.scheduler = kwds.pop("scheduler", "threads")

        # Default to 25 concurrent copy tasks
        self.ntasks = int(kwds.pop("ntasks", 25))

    def copy(
        self,
        source: str,
        force_copy: bool = False,
        **compute_kwds,
    ) -> str:
        """Local file copying method.

        Args:
            source (str): source path
            force_copy (bool, optional): re-copy all files. Defaults to False.

        Raises:
            FileNotFoundError: Raised if the source path is not found or empty.
            OSError: Raised if the target disk is full.

        Returns:
            str: Path of the copied source directory mapped into the target tree
        """

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
            filenames.append(os.path.realpath(source))

        elif os.path.isdir(source):
            sdir = os.path.realpath(source)
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
                f"Target disk full, only {free / 2**30} GB free, "
                + f"but {(size_src - size_dst) / 2**30} GB needed!",
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
            if not os.path.exists(dest_file) or size_dst != size_src or force_copy:
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
                    scheduler=self.scheduler,
                    num_workers=self.ntasks,
                    **compute_kwds,
                )
            print("Copy finished!")

        if os.path.isdir(source):
            return ddir

        return dest_file

    def size(self, sdir: str) -> int:
        """Calculate file size.

        Args:
            sdir (str): Path to source directory

        Returns:
            int: Size of files in source path
        """

        size = 0
        for path, dirs, filenames in os.walk(  # pylint: disable=W0612
            sdir,
        ):
            # Check space left
            for sfile in filenames:
                size += os.path.getsize(os.path.join(sdir, sfile))

        return size

    def cleanup_oldest_scan(
        self,
        force: bool = False,
        report: bool = False,
    ):
        """Remove scans in old directories. Looks for the directory with the oldest
        ctime and queries the user to confirm for its deletion.

        Args:
            force (bool, optional): Forces to automatically remove the oldest scan.
                Defaults to False.
            report (bool, optional): Print a report with all directories in dest,
                sorted by age. Defaults to False.

        Raises:
            FileNotFoundError: Raised if no scans to remove are found.
        """

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
    sdir: str,
    source: str,
    dest: str,
    gid: int,
    mode: int,
    create: bool = False,
) -> str:
    """Retrieve target directory.

    Args:
        sdir (str): Source directory to copy
        source (str): source root path
        dest (str): destination root path
        gid (int): Group id
        mode (int): Unix mode
        create (bool, optional): Wether to create directories. Defaults to False.

    Raises:
        NotADirectoryError: Raised if sdir is not a directory
        ValueError: Raised if sdir not inside of source

    Returns:
        str: The mapped targed directory inside dest
    """

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
def mymakedirs(path: str, mode: int, gid: int) -> List[str]:
    """Creates a directory path iteratively from its root

    Args:
        path (str): Path of the directory to create
        mode (int): Unix access mode of created directories
        gid (int): Group id of created directories

    Returns:
        str: Path of created directories
    """

    if not path or os.path.exists(path):
        return []
    head, tail = os.path.split(path)  # pylint: disable=W0612
    res = mymakedirs(head, mode, gid)
    os.mkdir(path)
    os.chmod(path, mode)
    os.chown(path, -1, gid)
    res.append(path)
    return res


def mycopy(source: str, dest: str, gid: int, mode: int, replace: bool = False):
    """Copy function with option to delete the target file firs (to take ownership).

    Args:
        source (str): Path to the source file
        dest (str): Path to the destination file
        gid (int): Group id to be set for the destination file
        mode (int): Unix access mode to be set for the destination file
        replace (bool, optional): Option to replace an existing file.
            Defaults to False.
    """

    if replace:
        if os.path.exists(dest):
            os.remove(dest)
    shutil.copy2(source, dest)
    # fix permissions and group ownership:
    os.chown(dest, -1, gid)
    os.chmod(dest, mode)
