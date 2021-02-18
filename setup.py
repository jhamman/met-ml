from setuptools import setup, find_packages


setup(name='met-ml',
      version='0.0.1',
	  packages=find_packages(),
	  license='Apache License 2.0',
      include_package_data=True,
	  long_description=open('README.md').read()
)
