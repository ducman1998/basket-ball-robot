import os
from glob import glob

from setuptools import find_packages, setup

package_name = "basket_robot_bringup"

launch_files = glob(os.path.join("launch", "*.launch.py"))  # or '*.py' if you prefer
params_files = glob(os.path.join("params", "*.yaml"))

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", launch_files),
        ("share/" + package_name + "/params", params_files),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Duc-Man Vo",
    maintainer_email="ducman@ut.ee",
    description="Bringup package for the Basket Robot",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
