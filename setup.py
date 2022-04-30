import setuptools
packages = ["SOM","Tests","helpers","libsom"]


module1 = setuptools.Extension('libsom',
                   sources = ['helpers/c_helper.c'],
                   include_dirs = [],
                   extra_compile_args=['-fPIC',"-fopenmp"]
                    )

setuptools.setup(
    name="SOM",
    version="0.1",
    packages=packages,
    include_package_data=True,
    description="Self-Organizing-Maps",
    long_description=open("README.md", "r").read(),
    package_dir={pkg: f"./{pkg.replace('.', '/')}" for pkg in packages},
    install_requires=[line.strip() for line in open('requirements.txt', 'r').readlines()],
    extras_require={},
    classifiers=[],
    package_data={'':['*.so']},
    # ext_modules = [module1]
)