#!/usr/bin/env python3
"""Put a Meridian launcher on your Desktop (Windows, macOS or Linux).

    python3 install_shortcut.py            create the shortcut
    python3 install_shortcut.py --remove   delete it again

The shortcut runs meridian.sh / meridian.bat from this folder, so keep the folder
where it is (re-run this script if you move it). Standard library only.
"""
import os
import shlex
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ICON_ICO = os.path.join(HERE, "static", "meridian.ico")
ICON_PNG = os.path.join(HERE, "static", "icon-512.png")


def _windows_desktop():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "[Environment]::GetFolderPath('Desktop')"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return out or os.path.join(os.path.expanduser("~"), "Desktop")


def _unix_desktop():
    try:                                  # honours a localised / relocated Desktop folder
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True).stdout.strip()
        if out and os.path.isdir(out):
            return out
    except OSError:
        pass
    return os.path.join(os.path.expanduser("~"), "Desktop")


def _ps_quote(s):
    return "'" + s.replace("'", "''") + "'"


def windows(remove):
    path = os.path.join(_windows_desktop(), "Meridian.lnk")
    if remove:
        return [path]
    bat = os.path.join(HERE, "meridian.bat")
    ps = ("$s=(New-Object -ComObject WScript.Shell).CreateShortcut(%s);"
          "$s.TargetPath=%s;$s.WorkingDirectory=%s;$s.IconLocation=%s;"
          "$s.Description='Meridian Terminal';$s.Save()"
          % (_ps_quote(path), _ps_quote(bat), _ps_quote(HERE), _ps_quote(ICON_ICO)))
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    return [path]


def macos(remove):
    # A .command file opens in Terminal, so you can see the login and stop it with Ctrl+C.
    path = os.path.join(os.path.expanduser("~"), "Desktop", "Meridian.command")
    if remove:
        return [path]
    with open(path, "w") as f:
        f.write("#!/bin/bash\nexec %s\n" % shlex.quote(os.path.join(HERE, "meridian.sh")))
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return [path]


def linux(remove):
    apps = os.path.join(os.path.expanduser("~"), ".local", "share", "applications", "meridian.desktop")
    paths = [os.path.join(_unix_desktop(), "Meridian.desktop"), apps]   # Desktop icon + app menu entry
    if remove:
        return paths
    # Exec values quote with double quotes; backslash-escape the characters the spec reserves.
    quoted = '"' + "".join("\\" + c if c in '"`$\\' else c for c in os.path.join(HERE, "meridian.sh")) + '"'
    entry = ("[Desktop Entry]\nType=Application\nName=Meridian Terminal\n"
             "Comment=Quant engine dashboard\nExec=%s\nPath=%s\nIcon=%s\n"
             "Terminal=true\nCategories=Office;Finance;\n" % (quoted, HERE, ICON_PNG))
    for p in paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(entry)
        os.chmod(p, 0o755)
    try:                                   # GNOME needs the Desktop file marked trusted to launch it
        subprocess.run(["gio", "set", paths[0], "metadata::trusted", "true"],
                       capture_output=True, check=False)
    except OSError:
        pass
    return paths


def main():
    remove = "--remove" in sys.argv[1:]
    handler = {"win32": windows, "darwin": macos}.get(sys.platform, linux)
    paths = handler(remove)
    for p in paths:
        if remove:
            if os.path.exists(p):
                os.remove(p)
                print("Removed", p)
        else:
            print("Created", p)
    if not remove:
        print("Double-click it to start Meridian (the first launch installs dependencies).")


if __name__ == "__main__":
    main()
