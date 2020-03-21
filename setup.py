from setuptools import setup

setup(name='carla_rllib',
      version='1.0',
      description='Reinforcement Library for the CARLA Simulator',
      classifiers=[
          'License :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Topic :: Reinforcement Learning',
      ],
      url='https://ids-git.fzi.de/svmuelle/carla_rllib.git',
      author='Sven Müller',
      author_email='svmuelle@fzi.de',
      license='MIT',
      packages=['carla_rllib'],
      install_requires=[
          'gym', 'numpy', 'pygame', 'tensorflow-gpu'
      ],
      zip_safe=False
      )

# add used packages
