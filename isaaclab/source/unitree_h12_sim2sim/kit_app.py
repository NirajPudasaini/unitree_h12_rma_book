# Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import ctypes
import os
import sys


def check_eula():
    package_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    # check for environment variable
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", default="N").lower() in ["y", "yes", "1"]:
        return

    # check for prior acceptance
    eula_accepted_path = os.path.join(package_path, "EULA_ACCEPTED")
    if os.path.isfile(eula_accepted_path):
        with open(eula_accepted_path, "r") as f:
            if f.readline().strip().lower() in ["y", "yes", "1"]:
                return

    # show notice
    print("\nBy installing or using Omniverse Kit, I agree to the terms of NVIDIA OMNIVERSE LICENSE AGREEMENT (EULA)")
    print("in https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html\n")

    # prompt for confirmation
    user_choice = input("Do you accept the EULA? (Yes/No): ")
    if user_choice.lower() not in ["y", "yes"]:
        print("The EULA was not accepted.\n")
        exit()

    # save confirmation
    with open(os.path.join(package_path, "EULA_ACCEPTED"), "w") as f:
        f.write("yes")

    print("The EULA was accepted.\n")


def bootstrap_kernel():
    root_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    # preload libcarb.so
    carb_library = "carb.dll" if sys.platform == "win32" else "libcarb.so"
    ctypes.CDLL(os.path.join(root_path, carb_library), mode=ctypes.RTLD_GLOBAL)

    # preload more libraries on linux. In normal kit run they are loaded implicitly using RPATH from kit executable
    if sys.platform != "win32":
        for file in ["libre2.so", "libcares.so"]:
            ctypes.CDLL(os.path.join(root_path, file), mode=ctypes.RTLD_GLOBAL)

    # set environment variables
    if not os.environ.get("CARB_APP_PATH", None):
        os.environ["CARB_APP_PATH"] = os.path.join(root_path)

    # add library path for windows
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(root_path)

    # load libpython on linux (requried for bindings), prefer system over pre-packaged
    if sys.platform != "win32":
        # determine which python version we are supposed to be using
        # by examinming what python executable is running this script
        kernel_plugins_path = os.path.join(root_path, "kernel/plugins")
        python_lib = f"libpython{sys.version_info.major}.{sys.version_info.minor}.so"

        try:
            ctypes.CDLL(python_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            ctypes.CDLL(os.path.join(kernel_plugins_path, python_lib), mode=ctypes.RTLD_GLOBAL)

    # set PYTHONPATH
    paths = [
        os.path.join(root_path, "kernel", "py"),
    ]
    for path in paths:
        if not path in sys.path:
            if not os.path.exists(path):
                print(f"PYTHONPATH: path doesn't exist ({path})")
                continue
            sys.path.insert(0, path)


check_eula()
bootstrap_kernel()

import carb
import omni.kit.app


class KitApp:
    def __init__(self) -> None:
        self._app = None
        self._framework = None

    @property
    def extension_manager(self):
        if not self._app:
            raise RuntimeError("The Omniverse app is not launched")
        return self._app.get_extension_manager()

    def startup(self, argv) -> None:
        # load the omniverse application plugins
        self._framework = carb.get_framework()
        self._framework.load_plugins(
            loaded_file_wildcards=["omni.kit.app.plugin"], search_paths=["${CARB_APP_PATH}/kernel/plugins"]
        )

        # start app
        self._app = omni.kit.app.get_app()

        argv = [sys.argv[0]] + argv if argv else sys.argv
        self._app.startup("kit", os.environ["CARB_APP_PATH"], argv)

    def shutdown(self) -> int:
        app_obj = self._app
        self._app = None
        self._framework = None

        return app_obj.shutdown_and_release_framework()

    def is_running(self):
        return self._app.is_running()

    def update(self):
        self._app.update()

    def run(self) -> int:
        self.startup(None)
        while self.is_running():
            self.update()
        return self.shutdown()


if __name__ == "__main__":
    sys.exit(KitApp().run())
